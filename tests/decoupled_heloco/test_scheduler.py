"""Round-robin scheduling with heterogeneous fragment metadata."""

from dataclasses import replace
import unittest

import torch

from panoengine.decentralized.decoupled_heloco.fragment_manager import FragmentManager
from panoengine.decentralized.decoupled_heloco.learner import LearnerMetadata
from panoengine.decentralized.decoupled_heloco.scheduler import RoundRobinScheduler
from panoengine.decentralized.decoupled_heloco.state import FragmentMetadata


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.parameters = {"a": torch.zeros(2), "b": torch.zeros(2)}
        self.manager = FragmentManager(self.parameters.items(), 2)
        self.scheduler = RoundRobinScheduler(self.manager, min_quorum=2, overlap_steps=3)

    def metadata(self, learner_id, steps, *, applied=0, received=None):
        if received is None:
            received = applied
        fragments = tuple(
            FragmentMetadata(i, self.manager.layout_signature, count, count * 8, received, applied)
            for i, count in enumerate(steps)
        )
        return LearnerMetadata(learner_id, self.manager.layout_signature, max(steps), max(steps) * 8, fragments)

    def test_quorum_skips_slow_workers_and_includes_all_ready_workers(self):
        learners = [self.metadata(3, (5, 0)), self.metadata(1, (1, 8)), self.metadata(0, (3, 0)), self.metadata(2, (4, 0))]
        plan = self.scheduler.plan(learners, fragment_revision=0)
        self.assertEqual(plan.learner_ids, (0, 2, 3))
        self.assertEqual(plan.fragment_id, 0)
        self.assertEqual(self.scheduler.sync_step, 0)

    def test_missing_quorum_does_not_skip_fragment_or_advance_clock(self):
        learners = [self.metadata(0, (3, 0)), self.metadata(1, (1, 8))]
        self.assertIsNone(self.scheduler.plan(learners, fragment_revision=0))
        self.assertEqual((self.scheduler.sync_step, self.scheduler.fragment_id), (0, 0))

    def test_round_robin_advances_only_on_commit_and_rejects_double_commit(self):
        learners = [self.metadata(0, (3, 3)), self.metadata(1, (3, 3))]
        selected = []
        first = None
        for _ in range(5):
            plan = self.scheduler.plan(learners, fragment_revision=0)
            selected.append(plan.fragment_id)
            if first is None:
                first = plan
            self.scheduler.commit(plan)
        self.assertEqual(selected, [0, 1, 0, 1, 0])
        with self.assertRaises(ValueError):
            self.scheduler.commit(first)

    def test_old_baselines_and_received_but_unapplied_replacements_are_not_ready(self):
        ready = self.metadata(0, (5, 0), applied=2)
        unapplied = self.metadata(1, (5, 0), applied=1, received=2)
        old = self.metadata(2, (5, 0), applied=1)
        self.assertIsNone(self.scheduler.plan([ready, unapplied, old], fragment_revision=2))
        just_reset = self.metadata(1, (0, 0), applied=2)
        self.assertIsNone(self.scheduler.plan([ready, just_reset], fragment_revision=2))
        progressed = self.metadata(1, (3, 0), applied=2)
        self.assertEqual(self.scheduler.plan([ready, progressed], fragment_revision=2).learner_ids, (0, 1))

    def test_incompatible_or_duplicate_metadata_is_rejected(self):
        ready = self.metadata(0, (3, 3))
        with self.assertRaises(ValueError):
            self.scheduler.plan([ready, ready], fragment_revision=0)
        with self.assertRaises(ValueError):
            self.scheduler.plan([replace(ready, layout_signature="other")], fragment_revision=0)
        with self.assertRaises(ValueError):
            self.scheduler.plan([replace(ready, fragments=tuple(reversed(ready.fragments)))], fragment_revision=0)
        with self.assertRaises(ValueError):
            self.scheduler.plan([replace(ready, learner_id=-1)], fragment_revision=0)

    def test_zero_tokens_do_not_satisfy_quorum(self):
        ready = self.metadata(0, (3, 3))
        invalid = self.metadata(1, (3, 3))
        invalid = replace(invalid, fragments=(replace(invalid.fragments[0], tokens=0), invalid.fragments[1]))
        self.assertIsNone(self.scheduler.plan([ready, invalid], fragment_revision=0))


if __name__ == "__main__":
    unittest.main()
