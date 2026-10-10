"""Round-robin fragment order and revision-aware quorum planning.

The syncer will own polling, grace windows, fragment revisions, networking,
and committing updates. This planner does not sleep or mutate any model.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .fragment_manager import FragmentManager
from .schedule_spec import resolve_paper_schedule

if TYPE_CHECKING:
    from .learner import LearnerMetadata


@dataclass(frozen=True)
class SyncPlan:
    sync_step: int
    fragment_id: int
    fragment_revision: int
    layout_signature: str
    learner_ids: tuple[int, ...]
    global_step: int = 0


class RoundRobinScheduler:
    kind = "round_robin"

    def __init__(self, manager: FragmentManager, *, min_quorum: int, overlap_steps: int, min_local_steps: int | None = None):
        for name, value in (("min_quorum", min_quorum), ("overlap_steps", overlap_steps)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.manager = manager
        self.min_quorum = min_quorum
        self.overlap_steps = overlap_steps
        self.min_local_steps = overlap_steps if min_local_steps is None else min_local_steps
        if type(self.min_local_steps) is not int or self.min_local_steps < 1:
            raise ValueError("min_local_steps must be a positive integer")
        self._sync_step = 0

    @property
    def sync_step(self) -> int:
        return self._sync_step

    @property
    def fragment_id(self) -> int:
        return self._sync_step % self.manager.num_fragments

    @property
    def global_step(self) -> int:
        return self.sync_step

    def plan(self, learners: Iterable[LearnerMetadata], *, fragment_revision: int) -> SyncPlan | None:
        """Include every currently ready learner; wait if quorum is missing.

        Readiness requires enough work since the requested global revision and
        no newer received replacement waiting to be applied. Metadata can
        change after planning, so pull requests must recheck the same revision.
        """
        if type(fragment_revision) is not int or fragment_revision < 0:
            raise ValueError("fragment_revision must be a nonnegative integer")
        seen = set()
        ready = []
        for learner in learners:
            if type(learner.learner_id) is not int or learner.learner_id < 0:
                raise ValueError("learner_id must be a nonnegative integer")
            if learner.learner_id in seen:
                raise ValueError("learner metadata contains a duplicate learner_id")
            seen.add(learner.learner_id)
            if learner.layout_signature != self.manager.layout_signature:
                raise ValueError("learner/scheduler fragment layout signatures differ")
            if tuple(f.fragment_id for f in learner.fragments) != tuple(range(self.manager.num_fragments)):
                raise ValueError("learner metadata must contain every fragment in canonical order")
            fragment = learner.fragments[self.fragment_id]
            if fragment.layout_signature != self.manager.layout_signature:
                raise ValueError("fragment metadata has an incompatible layout signature")
            if (
                fragment.local_steps >= self.min_local_steps
                and fragment.tokens > 0
                and fragment.last_applied_revision == fragment_revision
                and fragment.last_server_revision == fragment_revision
            ):
                ready.append(learner.learner_id)
        if len(ready) < self.min_quorum:
            return None
        return SyncPlan(self.sync_step, self.fragment_id, fragment_revision, self.manager.layout_signature, tuple(sorted(ready)), self.global_step)

    def validate_commit(self, plan: SyncPlan) -> None:
        """Check a plan before allocating/committing its outer update."""
        if (
            plan.sync_step != self.sync_step
            or plan.global_step != self.global_step
            or plan.fragment_id != self.fragment_id
            or plan.layout_signature != self.manager.layout_signature
        ):
            raise ValueError("cannot commit a stale or incompatible sync plan")
        if len(set(plan.learner_ids)) != len(plan.learner_ids) or len(plan.learner_ids) < self.min_quorum:
            raise ValueError("sync plan does not contain a distinct learner quorum")

    def commit(self, plan: SyncPlan) -> None:
        """Advance only after the syncer commits this fragment's outer update."""
        self.validate_commit(plan)
        self._sync_step += 1




class PaperOffsetScheduler(RoundRobinScheduler):
    """Algorithm 2's t mod H = t_p selection, starting at global t=1.

    sync_step counts committed fragment updates; global_step counts schedule
    slots, including slots without a fragment. Retries retain the current slot.
    Tau is the communication budget, independent of contribution readiness.
    This planner alone does not implement concurrent fragment transfers.
    """

    kind = "paper_offsets"

    def __init__(self, manager, *, min_quorum, overlap_steps, sync_period=None, fragment_offsets=None, min_local_steps=None):
        super().__init__(manager, min_quorum=min_quorum, overlap_steps=overlap_steps,
                         min_local_steps=1 if min_local_steps is None else min_local_steps)
        self.sync_period, self.fragment_offsets = resolve_paper_schedule(manager.num_fragments, sync_period, fragment_offsets)
        self._offset_fragments = {offset: i for i, offset in enumerate(self.fragment_offsets)}
        self._global_step = self._next_after(0)

    def _next_after(self, step):
        return min(step + ((offset - step % self.sync_period) % self.sync_period or self.sync_period) for offset in self.fragment_offsets)

    @property
    def global_step(self):
        return self._global_step

    @property
    def fragment_id(self):
        return self._offset_fragments[self.global_step % self.sync_period]

    def commit(self, plan):
        super().commit(plan)
        self._global_step = self._next_after(self._global_step)


def build_scheduler(manager, *, scheduler="round_robin", min_quorum, overlap_steps,
                    sync_period=None, fragment_offsets=None, min_local_steps=None):
    if scheduler == "paper_offsets":
        return PaperOffsetScheduler(manager, min_quorum=min_quorum, overlap_steps=overlap_steps,
                                    sync_period=sync_period, fragment_offsets=fragment_offsets,
                                    min_local_steps=min_local_steps)
    if scheduler == "round_robin":
        if sync_period is not None or fragment_offsets is not None:
            raise ValueError("sync_period/fragment_offsets apply only to paper_offsets")
        return RoundRobinScheduler(manager, min_quorum=min_quorum, overlap_steps=overlap_steps, min_local_steps=min_local_steps)
    raise ValueError("scheduler must be round_robin or paper_offsets")
