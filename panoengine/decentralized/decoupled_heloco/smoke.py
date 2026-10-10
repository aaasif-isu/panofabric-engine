"""Deterministic CPU token-model simulation; no production trainer/network."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from .config import ExperimentConfig
from .learner import DecoupledLearner
from .syncer import DecoupledSyncer


class _TinyTokenModel(nn.Module):
    def __init__(self, num_fragments: int):
        super().__init__()
        self.embedding = nn.Embedding(8, 8)
        # Provide enough whole parameter tensors for the requested toy layout.
        self.hidden = nn.ModuleList(nn.Linear(8, 8) for _ in range(max(1, (num_fragments - 2) // 2)))
        self.head = nn.Linear(8, 8)

    def forward(self, token_ids):
        hidden = self.embedding(token_ids)
        for layer in self.hidden:
            hidden = torch.tanh(layer(hidden))
        return self.head(hidden)


def run_smoke(config: ExperimentConfig, *, ticks: int = 30, outer_method: str = "heloco") -> int:
    """Use YAML topology/fragment controls with a fixed synthetic CPU recipe.

    Paces are simulated tick intervals, not measured wall-clock speedups.
    Real recipe, dataset, GPUs, quantization, grace window, and token-budget
    options are not exercised here. Choose HeLoCo math or the prior SGD baseline.
    """
    if type(ticks) is not int or ticks < 1:
        raise ValueError("smoke ticks must be a positive integer")
    islands = config.run.get("islands", 2)
    previous_threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.run.get("seed", 42))
        torch.set_num_threads(1)
        try:
            return _simulate(config, islands, ticks, outer_method)
        finally:
            torch.set_num_threads(previous_threads)


def _simulate(config, islands, ticks, outer_method):
    decoupled = config.decoupled
    model = _TinyTokenModel(decoupled.num_fragments)
    local_models = [copy.deepcopy(model) for _ in range(islands)]
    learners = [DecoupledLearner(m, decoupled.num_fragments, learner_id=i) for i, m in enumerate(local_models)]
    optimizers = [torch.optim.SGD(m.parameters(), lr=0.1) for m in local_models]
    syncer = DecoupledSyncer(
        model, learners, decoupled.num_fragments,
        min_quorum=decoupled.min_quorum, overlap_steps=decoupled.overlap_steps,
        outer_lr=config.run.get("outer_lr", 0.7),
        outer_method=outer_method, outer_momentum=config.run.get("outer_momentum", 0.9),
        heloco=decoupled.heloco,
    )
    # Each input entry is a token; the next-token target is a simple cycle.
    tokens = torch.arange(8).repeat(2, 1)
    targets = (tokens + 1) % 8
    with torch.no_grad():
        initial_loss = float(F.cross_entropy(model(tokens).reshape(-1, 8), targets.flatten()))
    paces = list(range(1, islands + 1))
    print(f"CPU smoke simulation: learners={islands}, fragments={decoupled.num_fragments}, quorum={decoupled.min_quorum}, overlap={decoupled.overlap_steps}")
    print(f"Simulated paces (ticks per local step): {paces}")
    if outer_method == "heloco":
        correction = "tensorwise" if decoupled.heloco.correction_enabled else "off"
        lookahead = "on" if decoupled.heloco.lookahead else "off"
        print(f"Merge: token-weighted average; outer: HeLoCo; correction: {correction}; look-ahead: {lookahead}; rho={decoupled.heloco.rho}")
    elif outer_method == "diloco":
        print("Merge: token-weighted average; outer: DiLoCo Nesterov; correction: off; look-ahead: off")
    else:
        print("Merge: token-weighted average; outer: SGD; correction: off; look-ahead: off")
    for tick in range(1, ticks + 1):
        for i, pace in enumerate(paces):
            if tick % pace:
                continue
            with learners[i].training_step(tokens=tokens.numel()):
                optimizers[i].zero_grad()
                loss = F.cross_entropy(local_models[i](tokens).reshape(-1, 8), targets.flatten())
                loss.backward()
                optimizers[i].step()
        if not syncer.has_active_sync:
            syncer.begin_sync()
        for learner in learners:
            learner.boundary()
        result = syncer.poll()
        if result is not None:
            rounded_weights = [round(weight, 3) for weight in result.weights]
            print(f"tick={tick:02d} sync={result.sync_step} fragment={result.fragment_id} revision={result.fragment_revision} learners={list(result.learner_ids)} local_steps={list(result.local_steps)} weights={rounded_weights}")
        for learner in learners:
            learner.boundary()
    syncer.cancel_sync()
    model.load_state_dict(syncer.optimizer.model_snapshot())
    with torch.no_grad():
        final_loss = float(F.cross_entropy(model(tokens).reshape(-1, 8), targets.flatten()))
    revisions = syncer.fragment_revisions
    delivered = all(
        metadata.last_applied_revision == revisions[metadata.fragment_id]
        for learner in learners for metadata in learner.metadata().fragments
    )
    print(f"Final fragment revisions: {list(revisions)}")
    print(f"Local optimizer steps: {[learner.metadata().total_local_steps for learner in learners]}")
    print(f"Global toy loss: {initial_loss:.6f} -> {final_loss:.6f}")
    if min(revisions) > 0 and delivered:
        print("SMOKE TEST PASSED: every fragment synchronized and its final revision applied by all learners.")
        return 0
    print("SMOKE TEST INCOMPLETE: increase --smoke-ticks for the configured quorum/overlap.")
    return 2
