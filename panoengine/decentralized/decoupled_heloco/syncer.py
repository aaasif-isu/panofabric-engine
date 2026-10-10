"""Single-control-thread dense fragment syncer with HeLoCo, DiLoCo or toy SGD.

begin_sync requests captures without waiting. Learners service captures only
at their own training boundaries. poll commits once a captured quorum exists,
then broadcasts to all registered learners. Remote endpoints supply localhost
TCP; timing.py owns interval/grace control. Production FSDP/trainer integration
is still absent. HeLoCo corrects after merging.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass, replace
import threading
from typing import Protocol

import torch

from .fragment_manager import FragmentManager
from .learner import LearnerMetadata
from .optimizer import FragmentDiLoCo, FragmentHeLoCo, FragmentSGD
from .config import HeLoCoConfig
from .scheduler import RoundRobinScheduler, SyncPlan
from .state import FragmentSnapshot, _dense


class LearnerEndpoint(Protocol):
    """Shared nonblocking interface for a local adapter or remote endpoint."""

    def metadata(self) -> LearnerMetadata: ...
    def request_snapshot(self, request_id: str, fragment_id: int, expected_revision: int, *, min_local_steps: int = 1) -> Future[FragmentSnapshot]: ...
    def release_snapshot(self, request_id: str) -> bool: ...
    def queue_update(self, fragment_id: int, parameters: Mapping[str, torch.Tensor], revision: int, *, layout_signature: str) -> bool: ...


class SyncQuorumLost(RuntimeError):
    """Too few requested captures remain valid; no outer step was committed."""


@dataclass(frozen=True)
class SyncResult:
    sync_step: int
    fragment_id: int
    fragment_revision: int
    learner_ids: tuple[int, ...]
    local_steps: tuple[int, ...]
    tokens: tuple[int, ...]
    weights: tuple[float, ...]
    merged_norm: float
    pending_broadcast: tuple[int, ...]


@dataclass
class _PullAttempt:
    request_id: str
    plan: SyncPlan
    futures: dict[int, Future[FragmentSnapshot]]


@dataclass
class _Broadcast:
    fragment_id: int
    revision: int
    parameters: dict[str, torch.Tensor]
    pending: set[int]


class DecoupledSyncer:
    """Own independent global FP32 weights and per-fragment revisions.

    The initial global model and learners must start from identical weights.
    Layouts are checked here; transport/checkpoint initialization comes later.
    Own these learners' request sequence exclusively, from their initial state.
    All control methods run on the thread that constructed this syncer.
    """

    def __init__(
        self, global_model, learners: Iterable[LearnerEndpoint], num_fragments: int,
        *, min_quorum: int, overlap_steps: int, outer_lr: float = 1.0,
        outer_method: str = "heloco", outer_momentum: float = 0.9,
        heloco: HeLoCoConfig | None = None,
    ):
        self.manager = FragmentManager.from_model(global_model, num_fragments)
        self.scheduler = RoundRobinScheduler(self.manager, min_quorum=min_quorum, overlap_steps=overlap_steps)
        parameters = dict(global_model.named_parameters())
        if outer_method == "heloco":
            self.optimizer = FragmentHeLoCo(self.manager, parameters, lr=outer_lr, momentum=outer_momentum, config=heloco)
        elif outer_method == "diloco":
            self.optimizer = FragmentDiLoCo(self.manager, parameters, lr=outer_lr, momentum=outer_momentum)
        elif outer_method == "sgd":
            self.optimizer = FragmentSGD(self.manager, parameters, lr=outer_lr)
        else:
            raise ValueError("outer_method must be heloco, diloco or sgd for this syncer")
        self.outer_method = outer_method
        self._learners = {}
        for learner in learners:
            metadata = learner.metadata()
            if metadata.learner_id in self._learners:
                raise ValueError("duplicate registered learner_id")
            if metadata.layout_signature != self.manager.layout_signature:
                raise ValueError("global/learner fragment layout signatures differ")
            self._learners[metadata.learner_id] = learner
        if len(self._learners) < min_quorum:
            raise ValueError("registered learner count is smaller than min_quorum")
        self._revisions = [0] * num_fragments
        self._request_sequence = 0
        self._active: _PullAttempt | None = None
        self._broadcast: _Broadcast | None = None
        self._broadcast_errors: dict[int, str] = {}
        self._owner = threading.get_ident()

    def _control_thread(self) -> None:
        if threading.get_ident() != self._owner:
            raise RuntimeError("syncer control must run on its owning thread")

    @property
    def fragment_revisions(self) -> tuple[int, ...]:
        return tuple(self._revisions)

    @property
    def has_active_sync(self) -> bool:
        return self._active is not None

    @property
    def broadcast_errors(self) -> dict[int, str]:
        return dict(self._broadcast_errors)

    def begin_sync(self) -> SyncPlan | None:
        self._control_thread()
        if self._active is not None:
            raise RuntimeError("a fragment pull attempt is already active")
        if self.retry_broadcast():
            return None
        fragment = self.scheduler.fragment_id
        plan = self.scheduler.plan(
            (learner.metadata() for learner in self._learners.values()),
            fragment_revision=self._revisions[fragment],
        )
        if plan is None:
            return None
        request_id = str(self._request_sequence)
        self._request_sequence += 1
        futures = {}
        try:
            for learner_id in plan.learner_ids:
                futures[learner_id] = self._learners[learner_id].request_snapshot(
                    request_id, fragment, plan.fragment_revision,
                    min_local_steps=self.scheduler.overlap_steps,
                )
        except Exception:
            for learner_id in futures:
                self._learners[learner_id].release_snapshot(request_id)
            raise
        self._active = _PullAttempt(request_id, plan, futures)
        return plan

    def _validate_snapshot(self, snapshot: FragmentSnapshot, plan: SyncPlan) -> None:
        if (
            snapshot.fragment_id != plan.fragment_id
            or snapshot.layout_signature != plan.layout_signature
            or snapshot.base_revision != plan.fragment_revision
        ):
            raise ValueError("capture does not match the requested fragment/revision/layout")
        if type(snapshot.local_steps) is not int or snapshot.local_steps < self.scheduler.overlap_steps:
            raise ValueError("capture has insufficient local optimizer work")
        if type(snapshot.tokens) is not int or snapshot.tokens < 1:
            raise ValueError("capture must contain positive token work")
        for values in (snapshot.baseline, snapshot.current):
            self.manager.validate_update(plan.fragment_id, values)
            for tensor in values.values():
                _dense(tensor)
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError("capture contains nonfinite parameters")

    def extend_sync(self) -> tuple[int, ...]:
        """Add learners that become ready while the timed grace window is open."""
        self._control_thread()
        attempt = self._active
        if attempt is None:
            return ()
        ready = self.scheduler.plan(
            (learner.metadata() for learner in self._learners.values()),
            fragment_revision=attempt.plan.fragment_revision,
        )
        if ready is None:
            return ()
        added = []
        try:
            for learner_id in ready.learner_ids:
                if learner_id in attempt.futures:
                    continue
                attempt.futures[learner_id] = self._learners[learner_id].request_snapshot(
                    attempt.request_id, attempt.plan.fragment_id,
                    attempt.plan.fragment_revision, min_local_steps=self.scheduler.overlap_steps,
                )
                added.append(learner_id)
        except Exception:
            self._release_attempt()
            raise
        attempt.plan = replace(attempt.plan, learner_ids=tuple(sorted(attempt.futures)))
        return tuple(added)

    def capture_quorum_ready(self) -> bool:
        """Inspect completed captures without committing an outer update."""
        self._control_thread()
        if self._active is None:
            return False
        valid = 0
        for future in self._active.futures.values():
            if not future.done():
                continue
            try:
                self._validate_snapshot(future.result(), self._active.plan)
            except Exception:
                continue
            valid += 1
        return valid >= self.scheduler.min_quorum

    def _release_attempt(self) -> None:
        if self._active is not None:
            for learner_id in self._active.futures:
                self._learners[learner_id].release_snapshot(self._active.request_id)
            self._active = None

    def cancel_sync(self) -> bool:
        self._control_thread()
        if self._active is None:
            return False
        self._release_attempt()
        return True

    def retry_broadcast(self) -> tuple[int, ...]:
        """Retry queued delivery only, never the committed outer step."""
        self._control_thread()
        broadcast = self._broadcast
        if broadcast is None:
            return ()
        for learner_id in sorted(broadcast.pending):
            learner = self._learners[learner_id]
            try:
                accepted = learner.queue_update(
                    broadcast.fragment_id, broadcast.parameters, broadcast.revision,
                    layout_signature=self.manager.layout_signature,
                )
                received = learner.metadata().fragments[broadcast.fragment_id].last_server_revision
                if not accepted and received < broadcast.revision:
                    raise RuntimeError("learner did not accept the broadcast revision")
            except Exception as exc:
                self._broadcast_errors[learner_id] = str(exc)
            else:
                broadcast.pending.remove(learner_id)
                self._broadcast_errors.pop(learner_id, None)
        remaining = tuple(sorted(broadcast.pending))
        if not remaining:
            self._broadcast = None
        return remaining

    def poll(self) -> SyncResult | None:
        """Inspect completed captures without waiting on pending futures."""
        self._control_thread()
        attempt = self._active
        if attempt is None:
            return None
        valid = {}
        failures = {}
        unfinished = 0
        for learner_id, future in attempt.futures.items():
            if not future.done():
                unfinished += 1
                continue
            try:
                snapshot = future.result()  # already complete; never waits
                self._validate_snapshot(snapshot, attempt.plan)
            except Exception as exc:
                failures[learner_id] = str(exc)
            else:
                valid[learner_id] = snapshot
        if len(valid) < self.scheduler.min_quorum:
            if len(valid) + unfinished >= self.scheduler.min_quorum:
                return None
            self._release_attempt()
            raise SyncQuorumLost(f"fragment capture quorum lost; failures: {failures}")

        plan = attempt.plan
        ids = tuple(sorted(valid))
        total_tokens = sum(valid[i].tokens for i in ids)
        weights = tuple(valid[i].tokens / total_tokens for i in ids)
        # One merged buffer plus one contributor delta at a time; avoid
        # storing K additional full fragment pseudo-gradient dictionaries.
        merged = {
            name: torch.zeros_like(valid[ids[0]].baseline[name], device="cpu", dtype=torch.float32)
            for name in self.manager.fragment(plan.fragment_id).parameter_names
        }
        for learner_id, weight in zip(ids, weights):
            snapshot = valid[learner_id]
            for name in merged:
                delta = snapshot.baseline[name].detach().float().cpu() - snapshot.current[name].detach().float().cpu()
                merged[name].add_(delta, alpha=weight)
        # Validate the plan and allocate dispatch before changing global weights.
        norm = sum(float(tensor.double().square().sum()) for tensor in merged.values()) ** 0.5
        try:
            self.scheduler.validate_commit(plan)
            dispatch = self.optimizer.step(plan.fragment_id, merged)
        except Exception:
            self._release_attempt()
            raise
        self._revisions[plan.fragment_id] += 1
        revision = self._revisions[plan.fragment_id]
        self.scheduler.commit(plan)
        # Retain the preallocated updated fragment in an outbox until all
        # learners have queued it. Transport retries cannot repeat the update.
        self._broadcast = _Broadcast(
            plan.fragment_id, revision, dispatch, set(self._learners),
        )
        self._release_attempt()
        remaining = self.retry_broadcast()
        return SyncResult(
            plan.sync_step, plan.fragment_id, revision, ids,
            tuple(valid[i].local_steps for i in ids), tuple(valid[i].tokens for i in ids),
            weights, norm, remaining,
        )
