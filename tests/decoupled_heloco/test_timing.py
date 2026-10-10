"""Interval/grace behavior using a fake clock and actual learner boundaries."""

import copy
import unittest

import torch

from panoengine.decentralized.decoupled_heloco.learner import DecoupledLearner
from panoengine.decentralized.decoupled_heloco.syncer import DecoupledSyncer
from panoengine.decentralized.decoupled_heloco.timing import TimedSyncController


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


class TimingTests(unittest.TestCase):
    def make(self, *, count=3, factor=0.5):
        model = _Model()
        self.models = [copy.deepcopy(model) for _ in range(count)]
        self.learners = [DecoupledLearner(m, 1, learner_id=i) for i, m in enumerate(self.models)]
        self.syncer = DecoupledSyncer(model, self.learners, 1, min_quorum=2, overlap_steps=1, outer_method="sgd")
        self.now = 0.0
        self.controller = TimedSyncController(self.syncer, sync_interval=1.0, grace_window_factor=factor, clock=lambda: self.now)

    def train(self, i):
        optimizer = torch.optim.SGD(self.models[i].parameters(), lr=0.1)
        with self.learners[i].training_step(8):
            optimizer.zero_grad()
            self.models[i].weight.sum().backward()
            optimizer.step()

    def tick(self, now):
        self.now = now
        return self.controller.tick()

    def test_interval_and_grace_admit_a_later_ready_learner(self):
        self.make()
        self.train(0)
        self.train(1)
        self.assertIsNone(self.tick(0.99))
        self.assertFalse(self.syncer.has_active_sync)
        self.assertIsNone(self.tick(1.0))
        self.learners[0].boundary()
        self.learners[1].boundary()
        self.assertIsNone(self.tick(1.1))  # Grace closes at 1.6.
        self.train(2)
        self.assertIsNone(self.tick(1.3))  # Newly ready learner joins the pull.
        self.learners[2].boundary()
        self.assertIsNone(self.tick(1.59))
        result = self.tick(1.61)
        self.assertEqual(result.learner_ids, (0, 1, 2))
        self.assertEqual(result.weights, (1 / 3, 1 / 3, 1 / 3))
        self.assertEqual(self.syncer.fragment_revisions, (1,))
        for learner in self.learners:
            learner.boundary()
        for i in range(3):
            self.train(i)
        self.assertIsNone(self.tick(2.6))
        self.assertFalse(self.syncer.has_active_sync)
        self.tick(2.62)
        self.assertTrue(self.syncer.has_active_sync)

    def test_zero_grace_waits_for_a_capture_quorum_then_commits_immediately(self):
        self.make(count=2, factor=0.0)
        self.train(0)
        self.train(1)
        self.assertIsNone(self.tick(1.0))
        self.assertEqual(self.syncer.fragment_revisions, (0,))
        for learner in self.learners:
            learner.boundary()
        self.assertEqual(self.tick(1.01).fragment_revision, 1)

    def test_capture_timeout_preserves_outer_state_and_can_retry(self):
        self.make(count=2)
        self.train(0)
        self.train(1)
        self.tick(1.0)
        old_future = self.learners[0].request_snapshot("0", 0, 0)
        self.assertIsNone(self.tick(2.0))
        self.assertTrue(old_future.cancelled())
        self.assertFalse(self.syncer.has_active_sync)
        self.assertEqual(self.controller.timeouts, 1)
        self.assertEqual(self.syncer.fragment_revisions, (0,))
        torch.testing.assert_close(self.syncer.optimizer.model_snapshot()["weight"], torch.ones(1))
        self.tick(3.0)
        for learner in self.learners:
            learner.boundary()
        self.tick(3.1)
        self.assertEqual(self.tick(3.61).fragment_revision, 1)

    def test_unfinished_extra_capture_is_released_when_grace_expires(self):
        self.make()
        for i in range(3):
            self.train(i)
        self.tick(1.0)
        unfinished = self.learners[2].request_snapshot("0", 0, 0)
        self.learners[0].boundary()
        self.learners[1].boundary()
        self.tick(1.1)
        result = self.tick(1.61)
        self.assertEqual(result.learner_ids, (0, 1))
        self.assertTrue(unfinished.cancelled())
        self.learners[2].boundary()
        self.assertEqual(self.learners[2].metadata().fragments[0].last_applied_revision, 1)

    def test_training_continues_during_grace_without_mutating_retained_capture(self):
        self.make(count=2)
        for i in range(2):
            self.train(i)
        self.tick(1.0)
        for learner in self.learners:
            learner.boundary()
        self.tick(1.1)
        for i in range(2):
            self.train(i)
        result = self.tick(1.61)
        self.assertEqual(result.local_steps, (1, 1))
        self.assertEqual([learner.metadata().total_local_steps for learner in self.learners], [2, 2])
        torch.testing.assert_close(self.syncer.optimizer.model_snapshot()["weight"], torch.tensor([0.9]))

    def test_missing_readiness_does_not_open_or_skip_a_cycle(self):
        self.make(count=2)
        self.train(0)
        for now in (1.0, 2.0, 10.0):
            self.assertIsNone(self.tick(now))
        self.assertFalse(self.syncer.has_active_sync)
        self.assertEqual(self.syncer.scheduler.sync_step, 0)
        self.assertEqual(self.controller.timeouts, 0)

    def test_invalid_timing_and_backwards_clock_are_rejected(self):
        self.make(count=2)
        for interval, factor in ((0, 0.5), (float("nan"), 0.5), (1, -0.1), (1, 1.1), (True, 0.5)):
            with self.subTest(interval=interval, factor=factor):
                with self.assertRaises(ValueError):
                    TimedSyncController(self.syncer, sync_interval=interval, grace_window_factor=factor)
        self.tick(0.5)
        with self.assertRaises(ValueError):
            self.tick(0.4)


if __name__ == "__main__":
    unittest.main()
