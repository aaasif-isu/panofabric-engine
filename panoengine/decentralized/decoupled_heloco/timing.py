"""Monotonic, nonblocking interval and grace-window control for the syncer."""

import math
import time

from .syncer import DecoupledSyncer, SyncQuorumLost


class TimedSyncController:
    """Gate cycle starts; allow additional ready learners after a capture quorum.

    The first attempt starts after one interval. Subsequent attempts start no
    earlier than one interval after commit/timeout. An initial capture quorum
    must arrive within one interval of the request. By default measured slack
    (overlap_steps * step_time - quorum_time - sync_time) bounds the grace
    window. adaptive_grace=False restores interval * factor. Expiry commits the valid
    captured quorum and releases unfinished pulls. factor=0 commits as soon as
    the initial capture quorum arrives. Missing readiness does not skip a
    fragment, and capture timeout cancels without changing the outer state.
    """

    def __init__(self, syncer: DecoupledSyncer, *, sync_interval: float, grace_window_factor: float, adaptive_grace=True, ema_alpha=0.2, clock=time.monotonic, max_global_step=None):
        for name, value in (("sync_interval", sync_interval), ("grace_window_factor", grace_window_factor)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if sync_interval <= 0 or not 0 <= grace_window_factor <= 1:
            raise ValueError("interval must be positive and grace factor must be in [0, 1]")
        if type(adaptive_grace) is not bool or isinstance(ema_alpha, bool) or not isinstance(ema_alpha, (int, float)) or not math.isfinite(ema_alpha) or not 0 < ema_alpha <= 1:
            raise ValueError("adaptive_grace must be boolean and ema_alpha in (0, 1]")
        if max_global_step is not None and (type(max_global_step) is not int or max_global_step < 1):
            raise ValueError("max_global_step must be a positive integer or None")
        self.max_global_step = max_global_step
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
        self.quorum_losses = 0
        self.adaptive = adaptive_grace
        self.alpha = ema_alpha
        self.step_time_ema = None
        self.quorum_time_ema = None
        self.sync_time_ema = None
        self.last_grace_seconds = 0.0
        self._step_samples = {m.learner_id: (self._last_time, m.total_local_steps) for m in syncer.learner_metadata()}
        self._learner_step_times = {}
        self._attempt_started = None
        self._deliveries = {}
        self.paper_offsets = syncer.scheduler.kind == "paper_offsets"
        self._schedule_anchor_time = self._last_time
        self._schedule_anchor_step = 0
        self._retry_not_before = self._last_time
        self._pipeline_times = {}
        self._pipeline_retries = {}
        self.event_sink = None

    def _step_seconds(self):
        return self.step_time_ema if self.step_time_ema is not None else self.interval

    def _schedule_deadline(self):
        # Paper slots are paced by measured compute, anchored at the previous
        # attempt START. Communication/grace must not add another full interval
        # after commit. Idle H slots retain their elapsed compute-time spacing.
        distance = self.syncer.scheduler.global_step - self._schedule_anchor_step
        return max(self._retry_not_before, self._schedule_anchor_time + distance * self._step_seconds())

    def _retry_later(self, now):
        if self.paper_offsets:
            self._retry_not_before = now + self._step_seconds()
        else:
            self._next_start = now + self.interval

    def _ema(self, previous, sample):
        return sample if previous is None else (1 - self.alpha) * previous + self.alpha * sample

    def _observe(self, now):
        reports = self.syncer.learner_metadata()
        changed = False
        for metadata in reports:
            previous = self._step_samples.get(metadata.learner_id)
            if previous is None or metadata.total_local_steps < previous[1]:
                self._step_samples[metadata.learner_id] = (now, metadata.total_local_steps)
                self._learner_step_times.pop(metadata.learner_id, None)
            elif metadata.total_local_steps > previous[1] and now > previous[0]:
                sample = (now - previous[0]) / (metadata.total_local_steps - previous[1])
                self._learner_step_times[metadata.learner_id] = self._ema(self._learner_step_times.get(metadata.learner_id), sample)
                self._step_samples[metadata.learner_id] = (now, metadata.total_local_steps)
                changed = True
        samples = [self._learner_step_times[m.learner_id] for m in reports if m.learner_id in self._learner_step_times]
        if changed and len(samples) >= self.syncer.scheduler.min_quorum:
            # Estimate a quorum's pace rather than allowing one slow learner to
            # inflate the communication budget for all of its healthy peers.
            sample = sorted(samples)[self.syncer.scheduler.min_quorum - 1]
            self.step_time_ema = sample
        for key, started in list(self._deliveries.items()):
            fragment, revision = key
            acknowledged = sum(m.fragments[fragment].last_server_revision >= revision for m in reports)
            if acknowledged >= self.syncer.scheduler.min_quorum:
                self.sync_time_ema = self._ema(self.sync_time_ema, max(0.0, now - started))
                del self._deliveries[key]

    def _grace(self, quorum_time):
        if not self.adaptive:
            return self.interval * self.factor
        if self.step_time_ema is None:
            return 0.0  # No measured slack yet: do not invent a compute budget.
        slack = self.syncer.scheduler.overlap_steps * self.step_time_ema - max(quorum_time, self.quorum_time_ema or 0.0) - (self.sync_time_ema or 0.0)
        return self.factor * max(0.0, slack)

    def tick(self):
        if self.syncer.max_inflight_captures > 1:
            return self._tick_pipeline()
        now = self.clock()
        if not math.isfinite(now) or now < self._last_time:
            raise ValueError("clock must return a finite monotonic time")
        self._last_time = now
        self.syncer.retry_broadcast()
        self._observe(now)
        if not self.syncer.has_active_sync:
            self._capture_deadline = self._grace_deadline = None
            deadline = self._schedule_deadline() if self.paper_offsets else self._next_start
            if now < deadline:
                return None
            plan = self.syncer.begin_sync(max_global_step=self.max_global_step)
            if plan is None:
                return None
            self._trace("begin", plan, now)
            self._capture_deadline = now + (max(self.interval, self.syncer.scheduler.overlap_steps * self._step_seconds()) if self.paper_offsets else self.interval)
            self._attempt_started = now
            if self.paper_offsets:
                self._schedule_anchor_time = now
                self._schedule_anchor_step = self.syncer.scheduler.global_step
        self.syncer.extend_sync()
        if self._grace_deadline is None and self.syncer.capture_quorum_ready():
            quorum_time = now - self._attempt_started
            self.quorum_time_ema = self._ema(self.quorum_time_ema, quorum_time)
            self.last_grace_seconds = self._grace(quorum_time)
            self._grace_deadline = now + self.last_grace_seconds
            self._trace("quorum_ready", self.syncer._active.plan, now)
        if self._grace_deadline is not None:
            if now < self._grace_deadline:
                return None
            committing_plan = self.syncer._active.plan
            try:
                result = self.syncer.poll()
            except SyncQuorumLost:
                self.quorum_losses += 1
                self._retry_later(now)
                self._capture_deadline = self._grace_deadline = None
                return None
            if result is not None:
                self._trace("commit", committing_plan, now)
                # Bound retained observations to one latest revision per fragment.
                for key in list(self._deliveries):
                    if key[0] == result.fragment_id:
                        del self._deliveries[key]
                self._deliveries[(result.fragment_id, result.fragment_revision)] = now
                self._observe(self.clock())
                if not self.paper_offsets:
                    self._next_start = now + self.interval
                self._capture_deadline = self._grace_deadline = None
            elif now >= max(self._capture_deadline, self._grace_deadline):
                self.syncer.cancel_sync()
                self.quorum_losses += 1
                self._retry_later(now)
                self._capture_deadline = self._grace_deadline = None
            return result
        if now >= self._capture_deadline:
            self.syncer.cancel_sync()
            self.timeouts += 1
            self._retry_later(now)
            self._capture_deadline = None
        return None


    def _trace(self, event, plan, now):
        if self.event_sink is not None:
            self.event_sink({"event": event, "elapsed_s": now,
                             "sync_step": plan.sync_step, "global_step": plan.global_step,
                             "fragment_id": plan.fragment_id,
                             "inflight_captures": sum(bool(a.futures) for a in self.syncer.capture_attempts)})

    def _reset_capture(self, attempt, now, *, quorum_lost=False):
        self.syncer.cancel_sync(attempt.plan.sync_step, retain_slot=True)
        self._pipeline_times.pop(attempt.plan.sync_step, None)
        self._pipeline_retries[attempt.plan.sync_step] = now + self._step_seconds()
        if quorum_lost:
            self.quorum_losses += 1
        else:
            self.timeouts += 1
        self._trace("retry", attempt.plan, now)

    def _tick_pipeline(self):
        """Concurrent captures; serial schedule-ordered outer commits.

        A prepared younger capture waits for the head without expiring. Failed
        captures retain their slot, so retries cannot skip the shared clock.
        At most one reservation or retry and one outer commit occur per tick.
        """
        now = self.clock()
        if not math.isfinite(now) or now < self._last_time:
            raise ValueError("clock must return a finite monotonic time")
        self._last_time = now
        self.syncer.retry_broadcast()
        self._observe(now)

        # Track each slot's capture/grace deadline independently. A younger
        # transfer can complete while an older one is still being captured.
        for attempt in self.syncer.capture_attempts:
            if not attempt.futures:
                continue
            timing = self._pipeline_times.get(attempt.plan.sync_step)
            if timing is None:
                raise RuntimeError("pipeline capture has no admission timing")
            if timing["grace"] is None or now < timing["grace"]:
                self.syncer.extend_sync(attempt.plan.sync_step)
            if timing["grace"] is None and self.syncer.capture_quorum_ready(attempt.plan.sync_step):
                quorum_time = now - timing["started"]
                self.quorum_time_ema = self._ema(self.quorum_time_ema, quorum_time)
                self.last_grace_seconds = self._grace(quorum_time)
                timing["grace"] = now + self.last_grace_seconds
                self._trace("quorum_ready", attempt.plan, now)
            if timing["grace"] is None and now >= timing["deadline"]:
                self._reset_capture(attempt, now)
            elif timing["grace"] is not None and now >= timing["grace"] and not attempt.sealed:
                if not self.syncer.seal_capture(attempt.plan.sync_step):
                    self._reset_capture(attempt, now, quorum_lost=True)
                else:
                    self._trace("prepared", attempt.plan, now)

        head = self.syncer._active
        result = None
        if head is not None and head.futures:
            timing = self._pipeline_times[head.plan.sync_step]
            if timing["grace"] is not None and now >= timing["grace"]:
                try:
                    result = self.syncer.poll()
                except SyncQuorumLost:
                    self._reset_capture(head, now, quorum_lost=True)
                else:
                    if result is not None:
                        self._pipeline_times.pop(head.plan.sync_step, None)
                        self._pipeline_retries.pop(head.plan.sync_step, None)
                        for key in list(self._deliveries):
                            if key[0] == result.fragment_id:
                                del self._deliveries[key]
                        self._deliveries[(result.fragment_id, result.fragment_revision)] = now
                        self._trace("commit", head.plan, now)
                    elif now >= max(timing["deadline"], timing["grace"]):
                        self._reset_capture(head, now, quorum_lost=True)

        # Capacity counts reserved slots as well as retained completed tensors.
        # No fragment is captured a second time until its previous slot commits.
        retry = next((a for a in self.syncer.capture_attempts if not a.futures), None)
        candidate_step = self.syncer.next_capture_step
        if retry is not None:
            due = self._pipeline_retries.get(retry.plan.sync_step, now)
        elif self.paper_offsets:
            due = self._schedule_anchor_time + (candidate_step - self._schedule_anchor_step) * self._step_seconds()
        else:
            due = self._next_start
        if (now >= due and (retry is not None or len(self.syncer.capture_attempts) < self.syncer.max_inflight_captures)
                and (self.max_global_step is None or candidate_step <= self.max_global_step)):
            plan = self.syncer.begin_sync(max_global_step=self.max_global_step)
            if plan is not None:
                capture_budget = max(self.interval, self.syncer.scheduler.overlap_steps * self._step_seconds()) if self.paper_offsets else self.interval
                self._pipeline_times[plan.sync_step] = {"started": now, "deadline": now + capture_budget, "grace": None}
                self._pipeline_retries.pop(plan.sync_step, None)
                if retry is None:
                    if self.paper_offsets:
                        self._schedule_anchor_time = now
                        self._schedule_anchor_step = plan.global_step
                    else:
                        self._next_start = now + self.interval
                self._trace("begin", plan, now)
        return result
