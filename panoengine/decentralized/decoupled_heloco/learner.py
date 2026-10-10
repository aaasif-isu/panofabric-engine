"""Dense learner adapter with explicit training-thread boundaries.

This is independent of the existing trainer. It performs no optimizer step,
network operation, GPU collective, or wait for a server. A future trainer
integration must invoke these boundaries, including island-wide FSDP handling.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
import threading

import torch

from .fragment_manager import FragmentManager
from .state import FragmentMetadata, FragmentSnapshot, FragmentState


class StaleSnapshotRequest(RuntimeError):
    """The requested baseline revision is no longer applied locally."""


class SnapshotQueueFull(RuntimeError):
    """Release the previous request before retaining another snapshot."""


@dataclass(frozen=True)
class LearnerMetadata:
    learner_id: int
    layout_signature: str
    total_local_steps: int
    total_tokens: int
    fragments: tuple[FragmentMetadata, ...]


@dataclass(frozen=True)
class _SnapshotRequest:
    fragment_id: int
    expected_revision: int
    min_local_steps: int
    future: Future[FragmentSnapshot]


class DecoupledLearner:
    """Connect explicit successful local steps to independent fragment states.

    Construct this on the training thread. Use training_step around exactly one
    successful optimizer update; report the real island token count. For AMP
    steps that may be skipped, use begin_step/end_step with completed=False.
    Snapshots and pending replacements are serviced only at safe boundaries.

    At most one request is retained by default, limiting cached snapshots to
    one fragment. Receiver/control threads may read metadata, queue updates,
    and request/release snapshots. They must not call training boundary methods.
    """

    def __init__(self, model, num_fragments: int, *, learner_id: int, max_snapshot_requests: int = 1):
        if type(learner_id) is not int or learner_id < 0:
            raise ValueError("learner_id must be a nonnegative integer")
        if type(max_snapshot_requests) is not int or max_snapshot_requests < 1:
            raise ValueError("max_snapshot_requests must be a positive integer")
        self.learner_id = learner_id
        self.manager = FragmentManager.from_model(model, num_fragments)
        self._parameters = dict(model.named_parameters())
        self._states = tuple(
            FragmentState(self.manager, fragment.fragment_id, self._parameters)
            for fragment in self.manager.fragments
        )
        self._owner = threading.get_ident()
        self._lock = threading.RLock()
        self._step_active = False
        self._total_local_steps = 0
        self._total_tokens = 0
        self._requests: dict[str, _SnapshotRequest] = {}
        self._max_snapshot_requests = max_snapshot_requests
        # Request IDs are monotonically increasing integers chosen by the
        # syncer, encoded as strings. A watermark prevents replay with O(1)
        # bookkeeping after snapshot tensors have been released.
        self._highest_request_id = -1

    def _training_thread(self) -> None:
        if threading.get_ident() != self._owner:
            raise RuntimeError("training boundaries must run on the thread that constructed the learner")

    def metadata(self) -> LearnerMetadata:
        with self._lock:
            return LearnerMetadata(
                self.learner_id, self.manager.layout_signature,
                self._total_local_steps, self._total_tokens,
                tuple(state.metadata() for state in self._states),
            )

    def queue_update(self, fragment_id: int, parameters: Mapping[str, torch.Tensor], revision: int, *, layout_signature: str) -> bool:
        self.manager.fragment(fragment_id)
        with self._lock:
            return self._states[fragment_id].queue_update(
                parameters, revision, layout_signature=layout_signature,
            )

    def request_snapshot(
        self, request_id: str, fragment_id: int, expected_revision: int, *, min_local_steps: int = 1,
    ) -> Future[FragmentSnapshot]:
        """Queue a capture; callers may await the future outside training.

        IDs must be canonical nonnegative integer strings, increasing across
        all requests to this learner. Retry a retained ID with identical args
        to receive the same future/capture. Release it after consumption.
        """
        if not isinstance(request_id, str) or not request_id.isascii() or not request_id.isdecimal():
            raise ValueError("request_id must be a canonical nonnegative integer string")
        sequence = int(request_id)
        if str(sequence) != request_id:
            raise ValueError("request_id must not contain leading zeros")
        self.manager.fragment(fragment_id)
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be a nonnegative integer")
        if type(min_local_steps) is not int or min_local_steps < 1:
            raise ValueError("min_local_steps must be a positive integer")
        with self._lock:
            existing = self._requests.get(request_id)
            if existing is not None:
                if (existing.fragment_id, existing.expected_revision, existing.min_local_steps) != (fragment_id, expected_revision, min_local_steps):
                    raise ValueError("a retried request_id must use identical snapshot arguments")
                return existing.future
            if sequence <= self._highest_request_id:
                raise ValueError("request_id has already been released or is out of order")
            if len(self._requests) >= self._max_snapshot_requests:
                raise SnapshotQueueFull("release the retained snapshot request before requesting another")
            future: Future[FragmentSnapshot] = Future()
            self._requests[request_id] = _SnapshotRequest(fragment_id, expected_revision, min_local_steps, future)
            self._highest_request_id = sequence
            return future

    def release_snapshot(self, request_id: str) -> bool:
        """Drop the learner's cache; cancel a capture that has not occurred."""
        with self._lock:
            request = self._requests.pop(request_id, None)
            if request is None:
                return False
            if not request.future.done():
                request.future.cancel()
            return True

    def _service_boundary(self) -> tuple[int, ...]:
        # Called with the adapter lock, on the training thread, while inactive.
        applied = tuple(
            state.fragment_id for state in self._states
            if state.apply_pending(self._parameters)
        )
        for request in tuple(self._requests.values()):
            if request.future.done():
                continue
            state = self._states[request.fragment_id]
            meta = state.metadata()
            if meta.last_applied_revision == request.expected_revision and meta.local_steps < request.min_local_steps:
                continue
            # Future.cancel() may be called outside the adapter lock. Mark the
            # capture running before touching tensors so cancellation cannot
            # race with set_result/set_exception at a completed boundary.
            if not request.future.set_running_or_notify_cancel():
                continue
            if meta.last_applied_revision != request.expected_revision:
                request.future.set_exception(StaleSnapshotRequest(
                    f"fragment {request.fragment_id}: expected revision {request.expected_revision}, "
                    f"applied revision is {meta.last_applied_revision}"
                ))
            else:
                try:
                    snapshot = state.snapshot(self._parameters)
                except Exception as exc:
                    request.future.set_exception(exc)
                else:
                    request.future.set_result(snapshot)
        return applied

    def boundary(self) -> tuple[int, ...]:
        """Service queued work while idle; useful before training or shutdown."""
        self._training_thread()
        with self._lock:
            if self._step_active:
                raise RuntimeError("cannot service a boundary during an active training step")
            return self._service_boundary()

    def begin_step(self) -> None:
        self._training_thread()
        with self._lock:
            if self._step_active:
                raise RuntimeError("a training step is already active")
            self._service_boundary()
            self._step_active = True

    def end_step(self, tokens: int, *, completed: bool = True) -> tuple[int, ...]:
        """Count one successful optimizer step, then service the boundary.

        For skipped steps use completed=False and tokens=0. The caller still
        owns optimizer/gradient handling; there is no rollback on exceptions.
        """
        self._training_thread()
        if type(completed) is not bool:
            raise ValueError("completed must be true or false")
        if type(tokens) is not int or tokens < (1 if completed else 0):
            raise ValueError("tokens must be a positive integer for completed steps")
        if not completed and tokens != 0:
            raise ValueError("skipped steps must report tokens=0")
        with self._lock:
            if not self._step_active:
                raise RuntimeError("no training step is active")
            self._step_active = False
            if completed:
                for state in self._states:
                    state.record_step(tokens)
                self._total_local_steps += 1
                self._total_tokens += tokens
            return self._service_boundary()

    @contextmanager
    def training_step(self, tokens: int):
        if type(tokens) is not int or tokens < 1:
            raise ValueError("tokens must be a positive integer")
        self.begin_step()
        try:
            yield
        except BaseException:
            with self._lock:
                self._step_active = False
            raise
        else:
            self.end_step(tokens)
