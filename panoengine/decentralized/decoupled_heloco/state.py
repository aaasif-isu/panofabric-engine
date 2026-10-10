"""A fragment's baseline, work counters, and queued server replacement.

No training hook or networking is installed here. Snapshot and apply_pending
must be called by the training thread at a safe optimizer-step boundary.
The lock protects bookkeeping, not concurrent forward/backward parameter use.
Only dense, unsharded real tensors are supported by this initial adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import threading
from types import MappingProxyType

import torch

from .fragment_manager import FragmentManager


def _revision(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("revision must be a nonnegative integer")


def _dense(tensor: torch.Tensor) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.is_meta
        or tensor.layout != torch.strided
        or callable(getattr(tensor, "to_local", None))
        or not tensor.is_floating_point()
    ):
        raise ValueError("fragment state requires dense, unsharded, floating-point tensors with real storage")


def _cpu_copy(parameters: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Check the whole fragment before performing any allocation/copy.
    for tensor in parameters.values():
        _dense(tensor)
    return {
        name: tensor.detach().to(device="cpu", dtype=torch.float32, copy=True)
        for name, tensor in parameters.items()
    }


@dataclass(frozen=True)
class FragmentMetadata:
    fragment_id: int
    layout_signature: str
    local_steps: int
    tokens: int
    last_server_revision: int
    last_applied_revision: int


@dataclass(frozen=True)
class FragmentSnapshot:
    fragment_id: int
    layout_signature: str
    base_revision: int
    local_steps: int
    tokens: int
    baseline: Mapping[str, torch.Tensor]
    current: Mapping[str, torch.Tensor]

    def pseudo_gradient(self) -> dict[str, torch.Tensor]:
        """Baseline minus captured local weights, in FP32 on CPU."""
        return {name: self.baseline[name] - self.current[name] for name in self.baseline}


class FragmentState:
    """One state per learner/fragment; baselines are private CPU FP32 copies.

    A snapshot is non-consuming. This version's baseline advances only when a
    newer global fragment is actually applied. A future syncer must deduplicate
    pulls by fragment revision/request, rather than merging the same snapshot
    twice. Applying a replacement discards local work since the capture,
    matching replacement semantics; delta-preserving rebasing is not added.
    """

    def __init__(
        self,
        manager: FragmentManager,
        fragment_id: int,
        parameters: Mapping[str, torch.Tensor],
        *,
        initial_revision: int = 0,
    ):
        _revision(initial_revision)
        self.manager = manager
        self.fragment_id = fragment_id
        self._baseline = _cpu_copy(manager.select(fragment_id, parameters))
        self._local_steps = 0
        self._tokens = 0
        self._last_server_revision = initial_revision
        self._last_applied_revision = initial_revision
        self._pending: tuple[int, dict[str, torch.Tensor]] | None = None
        self._lock = threading.RLock()

    def record_step(self, tokens: int) -> None:
        """Call for every fragment after a successful local optimizer step.

        `tokens` is the island's real token count for that optimizer step,
        including gradient accumulation, not a count per microbatch/rank.
        """
        if type(tokens) is not int or tokens < 1:
            raise ValueError("tokens must be a positive integer")
        with self._lock:
            self._local_steps += 1
            self._tokens += tokens

    def metadata(self) -> FragmentMetadata:
        with self._lock:
            return FragmentMetadata(
                self.fragment_id, self.manager.layout_signature,
                self._local_steps, self._tokens,
                self._last_server_revision, self._last_applied_revision,
            )

    def snapshot(self, parameters: Mapping[str, torch.Tensor]) -> FragmentSnapshot:
        """Capture at a safe step boundary without altering baseline/counters."""
        with self._lock:
            current = _cpu_copy(self.manager.select(self.fragment_id, parameters))
            baseline = {name: tensor.clone() for name, tensor in self._baseline.items()}
            return FragmentSnapshot(
                self.fragment_id, self.manager.layout_signature,
                self._last_applied_revision, self._local_steps, self._tokens,
                MappingProxyType(baseline), MappingProxyType(current),
            )

    def queue_update(
        self,
        parameters: Mapping[str, torch.Tensor],
        revision: int,
        *,
        layout_signature: str,
    ) -> bool:
        """Queue the latest revision; never mutate model weights here.

        This can run in a receiver thread. The training thread calls
        apply_pending at its next safe step boundary. Returns False for a
        duplicate or older revision, including one older than a pending update.
        """
        _revision(revision)
        if layout_signature != self.manager.layout_signature:
            raise ValueError("server/learner fragment layout signatures differ")
        self.manager.validate_update(self.fragment_id, parameters)
        with self._lock:
            if revision <= self._last_server_revision:
                return False
            pending = _cpu_copy(parameters)
            self._pending = (revision, pending)
            self._last_server_revision = revision
            return True

    def apply_pending(self, parameters: Mapping[str, torch.Tensor]) -> bool:
        """Replace only this fragment at a safe optimizer-step boundary.

        Stage and validate every tensor before copying any model weight. The
        new baseline reflects the actual local dtype, including BF16 rounding.
        Local optimizer moments are left to the future learner's explicit
        policy; no optimizer is attached or reset by this state primitive.
        """
        with self._lock:
            if self._pending is None:
                return False
            revision, update = self._pending
            targets = self.manager.select(self.fragment_id, parameters)
            for tensor in targets.values():
                _dense(tensor)
            with torch.no_grad():
                staged = {
                    name: update[name].to(device=target.device, dtype=target.dtype, copy=True)
                    for name, target in targets.items()
                }
                new_baseline = _cpu_copy(staged)
                for name, target in targets.items():
                    target.copy_(staged[name])
            self._baseline = new_baseline
            self._local_steps = self._tokens = 0
            self._last_applied_revision = revision
            self._pending = None
            return True
