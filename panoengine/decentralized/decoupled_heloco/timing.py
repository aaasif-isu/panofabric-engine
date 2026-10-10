"""Monotonic, nonblocking interval and grace-window control for the syncer."""

import math
import time

from .syncer import DecoupledSyncer


class TimedSyncController:
    """Gate cycle starts; allow additional ready learners after a capture quorum.

    The first attempt starts after one interval. Subsequent attempts start no
    earlier than one interval after commit/timeout. An initial capture quorum
    must arrive within one interval of the request. Then a grace window of
    interval * factor admits more ready learners; its expiry commits the valid
    captured quorum and releases unfinished pulls. factor=0 commits as soon as
    the initial capture quorum arrives. Missing readiness does not skip a
    fragment, and capture timeout cancels without changing the outer state.
    """

    def __init__(self, syncer: DecoupledSyncer, *, sync_interval: float, grace_window_factor: float, clock=time.monotonic):
        for name, value in (("sync_interval", sync_interval), ("grace_window_factor", grace_window_factor)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if sync_interval <= 0 or not 0 <= grace_window_factor <= 1:
            raise ValueError("interval must be positive and grace factor must be in [0, 1]")
        self.syncer = syncer
        self.interval = sync_interval
        self.factor = grace_window_factor
        self.clock = clock
        self._last_time = clock()
        if not math.isfinite(self._last_time):
            raise ValueError("clock must return a finite monotonic time")
        self._next_start = self._last_time + self.interval
        self._capture_deadline = None
        self._grace_deadline = None
        self.timeouts = 0

    def tick(self):
        now = self.clock()
        if not math.isfinite(now) or now < self._last_time:
            raise ValueError("clock must return a finite monotonic time")
        self._last_time = now
        self.syncer.retry_broadcast()
        if not self.syncer.has_active_sync:
            self._capture_deadline = self._grace_deadline = None
            if now < self._next_start or self.syncer.begin_sync() is None:
                return None
            self._capture_deadline = now + self.interval
        self.syncer.extend_sync()
        if self._grace_deadline is None and self.syncer.capture_quorum_ready():
            self._grace_deadline = now + self.interval * self.factor
        if self._grace_deadline is not None:
            if now < self._grace_deadline:
                return None
            result = self.syncer.poll()
            if result is not None:
                self._next_start = now + self.interval
                self._capture_deadline = self._grace_deadline = None
            return result
        if now >= self._capture_deadline:
            self.syncer.cancel_sync()
            self.timeouts += 1
            self._next_start = now + self.interval
            self._capture_deadline = None
        return None
