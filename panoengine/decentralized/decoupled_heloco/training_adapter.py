"""Token accounting and island data shards, independent of TorchTitan imports."""

from dataclasses import dataclass


@dataclass
class IslandDataLoaderConfig:
    """Keep single-rank training but shard its data across learner islands."""

    source: object
    islands: int
    learner_id: int

    def build(self, **kwargs):
        kwargs.update(dp_world_size=self.islands, dp_rank=self.learner_id)
        return self.source.build(**kwargs)

    def to_dict(self):
        return {"source": self.source.to_dict(), "islands": self.islands, "learner_id": self.learner_id}


class _CountedBatches:
    def __init__(self, iterator):
        self.iterator = iterator
        self.tokens = 0

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.iterator)
        # TorchTitan labels are on CPU here. Count real loss-bearing tokens,
        # including every microbatch consumed for gradient accumulation.
        labels = batch[1]
        tokens = int((labels != -100).sum().item())
        if tokens < 1:
            raise ValueError("training batch has no valid target tokens")
        self.tokens += tokens
        return batch


def train_one_step(trainer, iterator, learner):
    """Wrap one ordinary optimizer step; exceptions count no completed work.

    This adapter targets the pinned base Trainer's plain AdamW path, which has
    one optimizer update per train_step and no GradScaler skipped updates.
    It does not synchronize with the server or touch optimizer moments.
    """
    batches = _CountedBatches(iterator)
    learner.begin_step()
    try:
        trainer.train_step(batches)
        if batches.tokens < 1:
            raise ValueError("trainer completed a step without consuming data")
    except BaseException:
        learner.end_step(0, completed=False)
        raise
    learner.end_step(batches.tokens)
    return batches.tokens
