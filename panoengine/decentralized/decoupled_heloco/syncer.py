"""Single-control-thread dense fragment syncer with HeLoCo, DiLoCo or toy SGD.

begin_sync requests captures without waiting. Learners service captures only
at their own training boundaries. poll commits once a captured quorum exists,
then broadcasts to all registered learners. Remote endpoints supply localhost
TCP; timing.py owns interval/grace control. Production FSDP/trainer integration
is still absent. HeLoCo defaults to correcting after merging.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from concurrent.futures import Future
import copy
from dataclasses import dataclass, replace
import threading
from typing import Protocol

import torch

from .fragment_manager import FragmentManager
from .learner import LearnerMetadata
from .optimizer import FragmentDiLoCo, FragmentHeLoCo, FragmentSGD
from .config import HeLoCoConfig
from .scheduler import build_scheduler, SyncPlan
from .state import FragmentSnapshot, _dense
from .merging import merge_gradients, contribution_weights


class LearnerEndpoint(Protocol):
    """Shared nonblocking interface for a local adapter or remote endpoint."""

    def metadata(self) -> LearnerMetadata: ...
    def request_snapshot(self, request_id: str, fragment_id: int, expected_revision: int, *, min_local_steps: int = 1) -> Future[FragmentSnapshot]: ...
    def release_snapshot(self, request_id: str) -> bool: ...
    def queue_update(self, fragment_id: int, parameters: Mapping[str, torch.Tensor], revision: int, *, layout_signature: str, global_step: int = 0) -> bool: ...


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
    global_step: int = 0


@dataclass
class _PullAttempt:
    request_id: str
    plan: SyncPlan
    futures: dict[int, Future[FragmentSnapshot]]
    request_ids: dict[int, str]
    sealed: bool = False


@dataclass
class _Broadcast:
    fragment_id: int
    revision: int
    parameters: dict[str, torch.Tensor]
    pending: set[int]
    global_step: int = 0


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
        weighting: str = "tokens_squared_per_step", merge: str = "weighted_average",
        scheduler: str = "round_robin", sync_period=None, fragment_offsets=None, min_local_steps=None,
        max_inflight_captures: int = 1, syncer_shards: int = 1, syncer_timeout: float = 60.0,
    ):
        self.manager = FragmentManager.from_model(global_model, num_fragments)
        self.scheduler = build_scheduler(self.manager, scheduler=scheduler, min_quorum=min_quorum,
                                         overlap_steps=overlap_steps, sync_period=sync_period,
                                         fragment_offsets=fragment_offsets, min_local_steps=min_local_steps)
        parameters = dict(global_model.named_parameters())
        if outer_method == "heloco":
            self.optimizer = FragmentHeLoCo(self.manager, parameters, lr=outer_lr, momentum=outer_momentum, config=heloco)
        elif outer_method == "diloco":
            self.optimizer = FragmentDiLoCo(self.manager, parameters, lr=outer_lr, momentum=outer_momentum)
        elif outer_method == "sgd":
            self.optimizer = FragmentSGD(self.manager, parameters, lr=outer_lr)
        else:
            raise ValueError("outer_method must be heloco, diloco or sgd for this syncer")
        if type(syncer_shards) is not int or syncer_shards < 1:
            raise ValueError("syncer_shards must be a positive integer")
        import math
        if isinstance(syncer_timeout, bool) or not isinstance(syncer_timeout, (int, float)) or not math.isfinite(syncer_timeout) or syncer_timeout <= 0:
            raise ValueError("syncer_timeout must be positive and finite")
        self.syncer_shards = syncer_shards
        self.outer_method = outer_method
        if weighting not in {"tokens", "tokens_squared_per_step"}:
            raise ValueError("unknown contribution weighting")
        if merge not in {"weighted_average", "rda", "paper_rda"}:
            raise ValueError("unknown fragment merge")
        self.weighting, self.merge = weighting, merge
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
        self._fragment_steps = [0] * num_fragments
        self.global_step = 0
        self._request_sequence = 0
        if type(max_inflight_captures) is not int or not 1 <= max_inflight_captures <= num_fragments:
            raise ValueError("max_inflight_captures must be in 1..num_fragments")
        self.max_inflight_captures = max_inflight_captures
        self.peak_inflight_captures = 0
        self._attempts: list[_PullAttempt] = []
        # Reservation advances independently; only the real scheduler commits.
        self._reservation_scheduler = copy.copy(self.scheduler)
        # Latest revision per fragment. Older missed revisions can be replaced:
        # learners overwrite parameters rather than replaying outer updates.
        self._broadcasts: dict[int, _Broadcast] = {}
        self._broadcast_errors: dict[int, str] = {}
        self._owner = threading.get_ident()
        if syncer_shards > 1:
            from .sharding import ShardedOptimizer
            options = {"lr": outer_lr}
            if outer_method != "sgd":
                options["momentum"] = outer_momentum
            if outer_method == "heloco":
                options["config"] = self.optimizer.config
            # Discard the central outer state; only replicas retain it.
            self.optimizer = ShardedOptimizer(self.optimizer.model_snapshot(), self.manager,
                                               outer_method, options, syncer_shards, syncer_timeout)
            self.optimizer.progress = self._pump_communication

    def _pump_communication(self):
        self._control_thread()
        for learner in tuple(self._learners.values()):
            pump = getattr(learner, "pump", None)
            if pump is not None:
                pump()

    def _control_thread(self) -> None:
        if threading.get_ident() != self._owner:
            raise RuntimeError("syncer control must run on its owning thread")

    @property
    def fragment_revisions(self) -> tuple[int, ...]:
        return tuple(self._revisions)

    @property
    def _active(self):
        """Compatibility view of the oldest reserved capture."""
        return self._attempts[0] if self._attempts else None

    @property
    def has_active_sync(self) -> bool:
        return bool(self._attempts)

    @property
    def capture_attempts(self):
        return tuple(self._attempts)

    @property
    def next_capture_step(self):
        retry = next((a for a in self._attempts if not a.futures), None)
        return retry.plan.global_step if retry else self._reservation_scheduler.global_step

    def _attempt(self, sync_step=None):
        if sync_step is None:
            return self._active
        return next((a for a in self._attempts if a.plan.sync_step == sync_step), None)

    def _planner(self, plan):
        planner = copy.copy(self.scheduler)
        planner._sync_step = plan.sync_step
        if planner.kind == "paper_offsets":
            planner._global_step = plan.global_step
        return planner

    @property
    def broadcast_errors(self) -> dict[int, str]:
        return dict(self._broadcast_errors)

    def learner_metadata(self):
        """Read only available endpoints; one failed peer cannot spoil a quorum."""
        reports = []
        for learner_id, learner in self._learners.items():
            if not getattr(learner, "available", True):
                continue
            try:
                reports.append(learner.metadata())
            except (ConnectionError, TimeoutError, OSError) as exc:
                self._broadcast_errors[learner_id] = str(exc)
        return reports

    def reconnect_learner(self, learner: LearnerEndpoint) -> None:
        """Attach a replacement endpoint and queue the latest dispatch to catch up.

        This restores fragment weights/baselines, not lost inner optimizer state.
        The caller owns connection authentication and trainer restart policy.
        """
        self._control_thread()
        metadata = learner.metadata()
        if metadata.learner_id not in self._learners or metadata.layout_signature != self.manager.layout_signature:
            raise ValueError("reconnected learner identity/layout mismatch")
        if any(f.last_server_revision > self._revisions[f.fragment_id] for f in metadata.fragments):
            raise ValueError("reconnected learner is ahead of the syncer")
        for attempt in self._attempts:
            request_id = attempt.request_ids.get(metadata.learner_id)
            if request_id is not None:
                self._safe_release(metadata.learner_id, request_id)
            future = attempt.futures.get(metadata.learner_id)
            if future is not None:
                # A completed snapshot from the old endpoint is also invalid
                # after reattachment; it must not enter a later merge.
                attempt.futures.pop(metadata.learner_id)
                attempt.sealed = False
                attempt.request_ids.pop(metadata.learner_id, None)
                if not future.done():
                    future.cancel()
        self._learners[metadata.learner_id] = learner
        for fragment_id, revision in enumerate(self._revisions):
            if metadata.fragments[fragment_id].last_server_revision >= revision:
                continue
            broadcast = self._broadcasts.get(fragment_id)
            if broadcast is None:
                broadcast = _Broadcast(fragment_id, revision, self.optimizer.dispatch_snapshot(fragment_id), set(), self._fragment_steps[fragment_id])
                self._broadcasts[fragment_id] = broadcast
            broadcast.pending.add(metadata.learner_id)
        self.retry_broadcast()

    def begin_sync(self, *, max_global_step=None) -> SyncPlan | None:
        """Reserve the next distinct fragment or retry a retained failed slot."""
        self._control_thread()
        if max_global_step is not None and (type(max_global_step) is not int or max_global_step < 1):
            raise ValueError("max_global_step must be a positive integer or None")
        self.retry_broadcast()
        retry = next((a for a in self._attempts if not a.futures), None)
        if retry is None:
            if len(self._attempts) >= self.max_inflight_captures:
                if self.max_inflight_captures == 1:
                    raise RuntimeError("a fragment pull attempt is already active")
                return None
            planner = self._reservation_scheduler
            fragment = planner.fragment_id
            if any(a.plan.fragment_id == fragment for a in self._attempts):
                return None  # Never reserve a fragment twice at one revision.
        else:
            planner = self._planner(retry.plan)
            fragment = retry.plan.fragment_id
        if max_global_step is not None and planner.global_step > max_global_step:
            return None
        plan = planner.plan(self.learner_metadata(), fragment_revision=self._revisions[fragment])
        if plan is None:
            return None
        request_id = str(self._request_sequence)
        self._request_sequence += 1
        futures = {}
        for learner_id in plan.learner_ids:
            try:
                futures[learner_id] = self._learners[learner_id].request_snapshot(
                    request_id, fragment, plan.fragment_revision,
                    min_local_steps=self.scheduler.min_local_steps,
                )
            except (ConnectionError, TimeoutError, OSError) as exc:
                self._broadcast_errors[learner_id] = str(exc)
        if len(futures) < self.scheduler.min_quorum:
            for learner_id in futures:
                self._safe_release(learner_id, request_id)
            return None
        plan = replace(plan, learner_ids=tuple(sorted(futures)))
        attempt = _PullAttempt(request_id, plan, futures, {i: request_id for i in futures})
        if retry is None:
            self._attempts.append(attempt)
            self._reservation_scheduler.commit(plan)
        else:
            self._attempts[self._attempts.index(retry)] = attempt
        self.peak_inflight_captures = max(self.peak_inflight_captures,
                                         sum(bool(a.futures) for a in self._attempts))
        return plan

    def _validate_snapshot(self, snapshot: FragmentSnapshot, plan: SyncPlan) -> None:
        if (
            snapshot.fragment_id != plan.fragment_id
            or snapshot.layout_signature != plan.layout_signature
            or snapshot.base_revision != plan.fragment_revision
        ):
            raise ValueError("capture does not match the requested fragment/revision/layout")
        if type(snapshot.local_steps) is not int or snapshot.local_steps < self.scheduler.min_local_steps:
            raise ValueError("capture has insufficient local optimizer work")
        if type(snapshot.tokens) is not int or snapshot.tokens < 1:
            raise ValueError("capture must contain positive token work")
        for values in (snapshot.baseline, snapshot.current):
            self.manager.validate_update(plan.fragment_id, values)
            for tensor in values.values():
                _dense(tensor)
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError("capture contains nonfinite parameters")

    def extend_sync(self, sync_step=None) -> tuple[int, ...]:
        """Add learners that become ready while the timed grace window is open."""
        self._control_thread()
        attempt = self._attempt(sync_step)
        if attempt is None or not attempt.futures or attempt.sealed:
            return ()
        ready = self._planner(attempt.plan).plan(
            self.learner_metadata(),
            fragment_revision=attempt.plan.fragment_revision,
        )
        if ready is None:
            return ()
        added = []
        for learner_id in ready.learner_ids:
            if learner_id in attempt.futures:
                continue
            try:
                # Late contributors need a fresh ID: they may already have
                # captured a younger slot, making the original ID stale.
                request_id = str(self._request_sequence)
                self._request_sequence += 1
                attempt.futures[learner_id] = self._learners[learner_id].request_snapshot(
                    request_id, attempt.plan.fragment_id,
                    attempt.plan.fragment_revision, min_local_steps=self.scheduler.min_local_steps,
                )
                attempt.request_ids[learner_id] = request_id
                added.append(learner_id)
            except (ConnectionError, TimeoutError, OSError) as exc:
                self._broadcast_errors[learner_id] = str(exc)
        attempt.plan = replace(attempt.plan, learner_ids=tuple(sorted(attempt.futures)))
        return tuple(added)

    def capture_quorum_ready(self, sync_step=None) -> bool:
        """Inspect completed captures without committing an outer update."""
        self._control_thread()
        attempt = self._attempt(sync_step)
        if attempt is None:
            return False
        valid = 0
        for future in attempt.futures.values():
            if not future.done():
                continue
            try:
                self._validate_snapshot(future.result(), attempt.plan)
            except Exception:
                continue
            valid += 1
        return valid >= self.scheduler.min_quorum

    def seal_capture(self, sync_step=None) -> bool:
        """Freeze the valid completed quorum at this slot's grace deadline."""
        self._control_thread()
        attempt = self._attempt(sync_step)
        if attempt is None:
            return False
        if attempt.sealed:
            return True
        valid = set()
        for learner_id, future in attempt.futures.items():
            if future.done():
                try:
                    self._validate_snapshot(future.result(), attempt.plan)
                except Exception:
                    continue
                valid.add(learner_id)
        if len(valid) < self.scheduler.min_quorum:
            return False
        for learner_id in tuple(attempt.futures):
            if learner_id not in valid:
                self._safe_release(learner_id, attempt.request_ids.pop(learner_id))
                attempt.futures.pop(learner_id)
        attempt.plan = replace(attempt.plan, learner_ids=tuple(sorted(valid)))
        attempt.sealed = True
        return True

    def _release_attempt(self, *, retain_slot=False, sync_step=None) -> None:
        attempt = self._attempt(sync_step)
        if attempt is not None:
            for learner_id, request_id in attempt.request_ids.items():
                self._safe_release(learner_id, request_id)
            if retain_slot:
                attempt.futures.clear()
                attempt.request_ids.clear()
                attempt.sealed = False
            else:
                self._attempts.remove(attempt)
                if not self._attempts:
                    self._reservation_scheduler = copy.copy(self.scheduler)

    def _safe_release(self, learner_id, request_id):
        try:
            self._learners[learner_id].release_snapshot(request_id)
        except (ConnectionError, TimeoutError, OSError) as exc:
            self._broadcast_errors[learner_id] = str(exc)

    def cancel_sync(self, sync_step=None, *, retain_slot=False) -> bool:
        """Cancel captures; pipeline retries retain the original scheduled slot."""
        self._control_thread()
        if sync_step is None and not retain_slot:
            had_attempts = bool(self._attempts)
            for attempt in tuple(self._attempts):
                self._release_attempt(sync_step=attempt.plan.sync_step)
            self._reservation_scheduler = copy.copy(self.scheduler)
            return had_attempts
        if self._attempt(sync_step) is None:
            return False
        if sync_step is not None and self.max_inflight_captures > 1 and not retain_slot:
            raise ValueError("individual pipeline cancellation must retain its schedule slot")
        self._release_attempt(sync_step=sync_step, retain_slot=retain_slot)
        return True

    def retry_broadcast(self) -> tuple[int, ...]:
        """Retry queued delivery only, never the committed outer step."""
        import time
        wall, cpu = time.perf_counter(), time.thread_time()
        had_pending = any(b.pending for b in self._broadcasts.values())
        self._control_thread()
        for fragment_id, broadcast in list(self._broadcasts.items()):
            for learner_id in sorted(broadcast.pending):
                learner = self._learners[learner_id]
                if not getattr(learner, "available", True):
                    continue
                try:
                    accepted = learner.queue_update(
                        broadcast.fragment_id, broadcast.parameters, broadcast.revision,
                        layout_signature=self.manager.layout_signature, global_step=broadcast.global_step,
                    )
                    received = learner.metadata().fragments[broadcast.fragment_id].last_server_revision
                    if not accepted and received < broadcast.revision:
                        continue  # Waiting for a remote queue ACK is normal.
                except Exception as exc:
                    self._broadcast_errors[learner_id] = str(exc)
                else:
                    broadcast.pending.remove(learner_id)
            if not broadcast.pending:
                del self._broadcasts[fragment_id]
        remaining = tuple(sorted({i for b in self._broadcasts.values() for i in b.pending}))
        for learner_id in list(self._broadcast_errors):
            if learner_id not in remaining:
                self._broadcast_errors.pop(learner_id)
        monitor = getattr(self, "monitor", None)
        if had_pending and monitor is not None and monitor.enabled:
            monitor.record("dispatch_enqueue", time.perf_counter()-wall, time.thread_time()-cpu, self.scheduler.sync_step)
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
            self._release_attempt(retain_slot=self.max_inflight_captures > 1)
            raise SyncQuorumLost(f"fragment capture quorum lost; failures: {failures}")

        import time
        process_wall, process_cpu = time.perf_counter(), time.thread_time()
        plan = attempt.plan
        ids = tuple(sorted(valid))
        weights = contribution_weights([valid[i].tokens for i in ids], [valid[i].local_steps for i in ids], self.weighting)
        if self.syncer_shards > 1:
            self.scheduler.validate_commit(plan)
            try:
                dispatch, norm = self.optimizer.merge_step(plan.fragment_id, valid, weights, self.merge)
            except Exception:
                self._release_attempt()
                raise
        else:
            # One merged buffer plus one contributor delta at a time; avoid
            # storing K additional full fragment pseudo-gradient dictionaries.
            merged = {}
            corrected_before_merge = self.outer_method == "heloco" and self.optimizer.config.correction_order == "correct_then_merge"
            for name in self.manager.fragment(plan.fragment_id).parameter_names:
                gradients = (valid[i].baseline[name].detach().float().cpu() - valid[i].current[name].detach().float().cpu() for i in ids)
                if corrected_before_merge:
                    gradients = (self.optimizer.correct({name: gradient})[name] for gradient in gradients)
                mode = self.merge
                if mode == "paper_rda":
                    mode = "weighted_average" if {"embedding", "tok_embeddings", "embed_tokens"}.intersection(name.split(".")) else "rda"
                merged[name] = merge_gradients(gradients, weights, mode)
            # Validate the plan and allocate dispatch before changing global weights.
            norm = sum(float(tensor.double().square().sum()) for tensor in merged.values()) ** 0.5
            try:
                self.scheduler.validate_commit(plan)
                if corrected_before_merge:
                    dispatch = self.optimizer.step(plan.fragment_id, merged, already_corrected=True)
                else:
                    dispatch = self.optimizer.step(plan.fragment_id, merged)
            except Exception:
                self._release_attempt()
                raise
        self._revisions[plan.fragment_id] += 1
        revision = self._revisions[plan.fragment_id]
        self.scheduler.commit(plan)
        self.global_step = plan.global_step
        self._fragment_steps[plan.fragment_id] = plan.global_step
        monitor = getattr(self, "monitor", None)
        if monitor is not None and monitor.enabled:
            monitor.record("merge_correction_outer_update", time.perf_counter()-process_wall,
                           time.thread_time()-process_cpu, self.scheduler.sync_step)
            monitor.trajectory.save_from(self.scheduler.sync_step, self.optimizer.model_snapshot,
                                    tokens=sum(m.total_tokens for m in self.learner_metadata()))
        # Retain the preallocated updated fragment in an outbox until all
        # learners have queued it. Transport retries cannot repeat the update.
        self._broadcasts[plan.fragment_id] = _Broadcast(
            plan.fragment_id, revision, dispatch, set(self._learners), plan.global_step,
        )
        self._release_attempt()
        remaining = self.retry_broadcast()
        return SyncResult(
            plan.sync_step, plan.fragment_id, revision, ids,
            tuple(valid[i].local_steps for i in ids), tuple(valid[i].tokens for i in ids),
            weights, norm, remaining, plan.global_step,
        )

    def close(self):
        """Release capture requests and stop persistent replica processes."""
        self.cancel_sync()
        if self.syncer_shards > 1:
            self.optimizer.close()
