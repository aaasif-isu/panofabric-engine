"""Exact outer updates, quorum failure, and retryable broadcast delivery."""

import copy
import io
from contextlib import redirect_stdout
import unittest
from unittest.mock import patch

import torch

from panoengine.decentralized.decoupled_heloco.config import DecoupledConfig, ExperimentConfig
from panoengine.decentralized.decoupled_heloco.learner import DecoupledLearner
from panoengine.decentralized.decoupled_heloco.smoke import run_smoke
from panoengine.decentralized.decoupled_heloco.syncer import DecoupledSyncer, SyncQuorumLost


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.ones(1))
        self.b = torch.nn.Parameter(torch.ones(1))


class SyncerTests(unittest.TestCase):
    def make(self, *, count=3, fragments=2, quorum=2, outer_lr=1.0):
        global_model = _Model()
        self.models = [copy.deepcopy(global_model) for _ in range(count)]
        self.learners = [DecoupledLearner(model, fragments, learner_id=i) for i, model in enumerate(self.models)]
        self.optimizers = [torch.optim.SGD(model.parameters(), lr=0.1) for model in self.models]
        self.syncer = DecoupledSyncer(global_model, self.learners, fragments, min_quorum=quorum, overlap_steps=1, outer_lr=outer_lr, outer_method="sgd")

    def train(self, learner_id, steps, tokens=8):
        for _ in range(steps):
            with self.learners[learner_id].training_step(tokens):
                self.optimizers[learner_id].zero_grad()
                sum(p.sum() for p in self.models[learner_id].parameters()).backward()
                self.optimizers[learner_id].step()

    def boundaries(self):
        for learner in self.learners:
            learner.boundary()

    def test_single_learner_full_fragment_matches_known_outer_sgd_step(self):
        self.make(count=1, fragments=1, quorum=1, outer_lr=0.5)
        self.train(0, 2)
        self.syncer.begin_sync()
        self.boundaries()
        result = self.syncer.poll()
        for tensor in self.syncer.optimizer.model_snapshot().values():
            torch.testing.assert_close(tensor, torch.tensor([0.9]))
        self.assertEqual(result.weights, (1.0,))
        self.assertEqual(self.syncer.fragment_revisions, (1,))
        self.boundaries()
        self.assertEqual(self.learners[0].metadata().fragments[0].local_steps, 0)

    def test_token_weighted_merge_and_broadcast_include_noncontributing_learner(self):
        self.make()
        self.train(0, 1)
        self.train(1, 3)
        plan = self.syncer.begin_sync()
        self.assertEqual(plan.learner_ids, (0, 1))
        self.boundaries()
        result = self.syncer.poll()
        self.assertEqual(result.tokens, (8, 24))
        self.assertEqual(result.weights, (0.25, 0.75))
        # Delta = .25*.1 + .75*.3 = .25; global a = 1 - .25.
        global_weights = self.syncer.optimizer.model_snapshot()
        torch.testing.assert_close(global_weights["a"], torch.tensor([0.75]))
        torch.testing.assert_close(global_weights["b"], torch.ones(1))
        torch.testing.assert_close(self.models[2].a, torch.ones(1))
        self.assertEqual(self.learners[2].metadata().fragments[0].last_server_revision, 1)
        self.boundaries()
        for learner, model in zip(self.learners, self.models):
            torch.testing.assert_close(model.a, torch.tensor([0.75]))
            self.assertEqual(learner.metadata().fragments[0].last_applied_revision, 1)
        self.assertEqual(self.learners[1].metadata().fragments[1].local_steps, 3)
        self.assertIsNone(self.syncer.poll())  # A duplicate poll cannot repeat the outer step.
        torch.testing.assert_close(self.syncer.optimizer.model_snapshot()["a"], torch.tensor([0.75]))

    def test_missing_quorum_leaves_global_state_unchanged(self):
        self.make()
        self.train(0, 1)
        self.assertIsNone(self.syncer.begin_sync())
        self.assertEqual(self.syncer.scheduler.sync_step, 0)
        self.assertEqual(self.syncer.fragment_revisions, (0, 0))
        torch.testing.assert_close(self.syncer.optimizer.model_snapshot()["a"], torch.ones(1))

    def test_poll_waits_for_capture_boundaries_without_blocking_or_servicing_training(self):
        self.make(count=1, quorum=1)
        self.train(0, 1)
        self.syncer.begin_sync()
        self.assertIsNone(self.syncer.poll())
        self.assertEqual(self.syncer.fragment_revisions, (0, 0))
        self.train(0, 1)  # Beginning the next step services the queued capture.
        result = self.syncer.poll()
        self.assertEqual(result.local_steps, (1,))
        self.assertEqual(self.learners[0].metadata().total_local_steps, 2)

    def test_cancelled_quorum_cleans_up_and_can_retry_without_double_update(self):
        self.make(count=1, quorum=1)
        self.train(0, 1)
        self.syncer.begin_sync()
        future = self.learners[0].request_snapshot("0", 0, 0)
        future.cancel()
        with self.assertRaises(SyncQuorumLost):
            self.syncer.poll()
        self.assertFalse(self.syncer.has_active_sync)
        self.assertEqual(self.syncer.fragment_revisions, (0, 0))
        self.syncer.begin_sync()
        self.boundaries()
        self.assertEqual(self.syncer.poll().fragment_revision, 1)
        self.assertEqual(self.syncer.scheduler.sync_step, 1)

    def test_valid_quorum_can_commit_despite_an_extra_cancelled_capture(self):
        self.make()
        for i in range(3):
            self.train(i, 1)
        self.syncer.begin_sync()
        self.learners[2].request_snapshot("0", 0, 0).cancel()
        self.boundaries()
        result = self.syncer.poll()
        self.assertEqual(result.learner_ids, (0, 1))
        self.assertEqual(result.pending_broadcast, ())
        self.boundaries()
        self.assertEqual(self.learners[2].metadata().fragments[0].last_applied_revision, 1)

    def test_broadcast_failure_retains_outbox_and_retry_never_repeats_outer_update(self):
        self.make()
        self.train(0, 1)
        self.train(1, 1)
        self.syncer.begin_sync()
        self.boundaries()
        with patch.object(self.learners[2], "queue_update", side_effect=RuntimeError("temporary delivery failure")):
            result = self.syncer.poll()
            self.assertEqual(result.pending_broadcast, (2,))
            self.assertEqual(self.syncer.begin_sync().fragment_id, 1)
            self.syncer.cancel_sync()
        committed = self.syncer.optimizer.model_snapshot()["a"]
        self.assertEqual(self.syncer.retry_broadcast(), ())
        torch.testing.assert_close(self.syncer.optimizer.model_snapshot()["a"], committed)
        self.assertEqual(self.syncer.fragment_revisions, (1, 0))
        self.assertEqual(self.syncer.scheduler.sync_step, 1)
        self.boundaries()
        torch.testing.assert_close(self.models[2].a, committed)

    def test_nonfinite_capture_is_rejected_before_global_mutation(self):
        self.make(count=1, quorum=1)
        self.train(0, 1)
        with torch.no_grad():
            self.models[0].a.fill_(float("nan"))
        self.syncer.begin_sync()
        self.boundaries()
        with self.assertRaises(SyncQuorumLost):
            self.syncer.poll()
        torch.testing.assert_close(self.syncer.optimizer.model_snapshot()["a"], torch.ones(1))
        self.assertEqual(self.syncer.fragment_revisions, (0, 0))

    def test_two_fragments_have_independent_revisions_and_global_weights(self):
        self.make(count=1, quorum=1)
        self.train(0, 1)
        for fragment in (0, 1):
            self.assertEqual(self.syncer.begin_sync().fragment_id, fragment)
            self.boundaries()
            self.assertEqual(self.syncer.poll().fragment_id, fragment)
            self.boundaries()
        self.assertEqual(self.syncer.fragment_revisions, (1, 1))
        for tensor in self.syncer.optimizer.model_snapshot().values():
            torch.testing.assert_close(tensor, torch.tensor([0.9]))


class SmokeDemoTests(unittest.TestCase):
    def test_default_cpu_demo_uses_tensor_correction_momentum_and_lookahead(self):
        config = ExperimentConfig(
            method="decoupled_heloco", run={"islands": 3, "seed": 42, "outer_lr": 0.7},
            decoupled=DecoupledConfig(num_fragments=2, min_quorum=2, overlap_steps=2),
        )
        output = io.StringIO()
        with redirect_stdout(output):
            status = run_smoke(config, ticks=12)
        self.assertEqual(status, 0)
        self.assertIn("outer: HeLoCo", output.getvalue())
        self.assertIn("correction: tensorwise", output.getvalue())
        self.assertIn("look-ahead: on", output.getvalue())
        self.assertIn("SMOKE TEST PASSED", output.getvalue())

    def test_cpu_demo_completes_all_fragments_and_reports_incomplete_when_too_short(self):
        config = ExperimentConfig(
            method="decoupled_heloco", run={"islands": 3, "seed": 42, "outer_lr": 0.7},
            decoupled=DecoupledConfig(num_fragments=2, min_quorum=2, overlap_steps=2),
        )
        output = io.StringIO()
        with redirect_stdout(output):
            status = run_smoke(config, ticks=12, outer_method="sgd")
        self.assertEqual(status, 0)
        self.assertIn("SMOKE TEST PASSED", output.getvalue())
        self.assertIn("correction: off", output.getvalue())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(run_smoke(config, ticks=1, outer_method="sgd"), 2)


if __name__ == "__main__":
    unittest.main()
