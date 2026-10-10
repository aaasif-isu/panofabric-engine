"""Offset selection, validation and capture retries."""
from dataclasses import replace
import unittest
from test_scheduler import SchedulerTests
from panoengine.decentralized.decoupled_heloco.scheduler import PaperOffsetScheduler, build_scheduler
from panoengine.decentralized.decoupled_heloco.config import DecoupledConfig, ConfigError

class PaperSchedulerTests(SchedulerTests):
    def test_sparse_offsets_and_retry_preserve_schedule_clock(self):
        scheduler = PaperOffsetScheduler(self.manager, min_quorum=2, overlap_steps=5,
                                        sync_period=6, fragment_offsets=[0, 2])
        self.assertIsNone(scheduler.plan([], fragment_revision=0))
        self.assertEqual(scheduler.global_step, 2)
        learners = [self.metadata(i, (1, 1)) for i in range(2)]
        selected = []
        for _ in range(6):
            plan = scheduler.plan(learners, fragment_revision=0)
            self.assertIsNotNone(plan)  # tau does not impose readiness=5.
            selected.append((plan.global_step, plan.fragment_id, plan.sync_step))
            with self.assertRaises(ValueError):
                scheduler.commit(replace(plan, global_step=plan.global_step+1))
            scheduler.commit(plan)
        self.assertEqual(selected, [(2,1,0),(6,0,1),(8,1,2),(12,0,3),(14,1,4),(18,0,5)])

    def test_explicit_offsets_control_fragment_order(self):
        scheduler = build_scheduler(self.manager, scheduler='paper_offsets', min_quorum=2,
                                    overlap_steps=2, sync_period=5, fragment_offsets=[3,1])
        learners = [self.metadata(i, (3,3)) for i in range(2)]
        selected = []
        for _ in range(4):
            plan = scheduler.plan(learners, fragment_revision=0)
            selected.append((plan.global_step, plan.fragment_id))
            scheduler.commit(plan)
        self.assertEqual(selected, [(1,1),(3,0),(6,1),(8,0)])

    def test_configuration_rejects_ignored_and_invalid_settings(self):
        for settings in ({'scheduler':'unknown'}, {'scheduler':'round_robin','sync_period':4},
                         {'scheduler':'paper_offsets','num_fragments':2,'sync_period':1},
                         {'scheduler':'paper_offsets','num_fragments':2,'fragment_offsets':[0,0]},
                         {'scheduler':'paper_offsets','num_fragments':2,'fragment_offsets':[0,2]},
                         {'scheduler':'paper_offsets','num_fragments':2,'fragment_offsets':[True,1]},
                         {'scheduler':'paper_offsets','num_fragments':2,'fragment_offsets':[0]}):
            with self.subTest(settings=settings), self.assertRaises(ConfigError):
                DecoupledConfig.from_mapping(settings)
        self.assertEqual(DecoupledConfig.from_mapping({'scheduler':'paper_offsets','overlap_steps':5}).capture_min_steps, 1)
        self.assertEqual(DecoupledConfig.from_mapping({'scheduler':'round_robin','overlap_steps':5}).capture_min_steps, 5)

from test_timing import TimingTests
from panoengine.decentralized.decoupled_heloco.syncer import DecoupledSyncer
from panoengine.decentralized.decoupled_heloco.timing import TimedSyncController

class PaperTimingTests(TimingTests):
    def test_sparse_slot_spacing_and_timeout_retry(self):
        self.make(count=2, factor=0)
        self.syncer = DecoupledSyncer(self.models[0], self.learners, 1, min_quorum=2,
            overlap_steps=2, outer_method='sgd', scheduler='paper_offsets', sync_period=4,
            fragment_offsets=[2])
        self.controller = TimedSyncController(self.syncer, sync_interval=1,
            grace_window_factor=0, clock=lambda:self.now)
        for i in range(2):
            self.train(i)
        self.tick(1)
        self.assertFalse(self.syncer.has_active_sync)
        self.tick(2)
        self.assertTrue(self.syncer.has_active_sync)
        self.tick(4)  # no boundaries: capture expires without advancing slot
        self.assertEqual(self.controller.timeouts,1)
        self.assertEqual(self.syncer.scheduler.global_step,2)
        self.tick(5)
        for learner in self.learners:
            learner.boundary()
        result = self.tick(5.1)
        self.assertEqual(result.global_step,2)
        self.assertEqual(self.syncer.scheduler.global_step,6)
        for learner in self.learners:
            learner.boundary()
        for i in range(2):
            self.train(i)
        # Fix the measured compute estimate to test the schedule in isolation.
        self.controller._observe = lambda now: None
        self.controller.step_time_ema = 1
        self.tick(8.99)
        self.assertFalse(self.syncer.has_active_sync)
        self.tick(9)
        self.assertTrue(self.syncer.has_active_sync)
