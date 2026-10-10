"""Round-robin fragment order and revision-aware quorum planning.

The syncer will own polling, grace windows, fragment revisions, networking,
and committing updates. This planner does not sleep or mutate any model.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .fragment_manager import FragmentManager

if TYPE_CHECKING:
    from .learner import LearnerMetadata


@dataclass(frozen=True)
class SyncPlan:
    sync_step: int
    fragment_id: int
    fragment_revision: int
    layout_signature: str
    learner_ids: tuple[int, ...]


class RoundRobinScheduler:
    def __init__(self, manager: FragmentManager, *, min_quorum: int, overlap_steps: int):
        for name, value in (("min_quorum", min_quorum), ("overlap_steps", overlap_steps)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.manager = manager
        self.min_quorum = min_quorum
        self.overlap_steps = overlap_steps
        self._sync_step = 0

    @property
    def sync_step(self) -> int:
        return self._sync_step

    @property
    def fragment_id(self) -> int:
        return self._sync_step % self.manager.num_fragments

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
                fragment.local_steps >= self.overlap_steps
                and fragment.tokens > 0
                and fragment.last_applied_revision == fragment_revision
                and fragment.last_server_revision == fragment_revision
            ):
                ready.append(learner.learner_id)
        if len(ready) < self.min_quorum:
            return None
        return SyncPlan(self.sync_step, self.fragment_id, fragment_revision, self.manager.layout_signature, tuple(sorted(ready)))

    def validate_commit(self, plan: SyncPlan) -> None:
        """Check a plan before allocating/committing its outer update."""
        if (
            plan.sync_step != self.sync_step
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
