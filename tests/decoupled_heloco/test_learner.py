"""Dense training boundaries, asynchronous requests, and bounded snapshots."""

from concurrent.futures import CancelledError
import threading
import unittest

import torch

from panoengine.decentralized.decoupled_heloco.learner import (
    DecoupledLearner, SnapshotQueueFull, StaleSnapshotRequest,
)


class LearnerTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            self.model.weight.fill_(1)
            self.model.bias.fill_(2)
        self.learner = DecoupledLearner(self.model, 2, learner_id=0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)

    def step(self, tokens=8):
        with self.learner.training_step(tokens):
            self.optimizer.zero_grad()
            sum(p.sum() for p in self.model.parameters()).backward()
            self.optimizer.step()

    def queue(self, fragment, update, revision):
        return self.learner.queue_update(fragment, update, revision, layout_signature=self.learner.manager.layout_signature)

    def test_real_steps_update_every_fragment_and_lifetime_token_counts(self):
        self.step(8)
        self.step(12)
        metadata = self.learner.metadata()
        self.assertEqual((metadata.total_local_steps, metadata.total_tokens), (2, 20))
        self.assertEqual([(f.local_steps, f.tokens) for f in metadata.fragments], [(2, 20), (2, 20)])
        future = self.learner.request_snapshot("0", 0, 0)
        self.assertFalse(future.done())
        self.learner.boundary()
        snapshot = future.result(timeout=1)
        torch.testing.assert_close(snapshot.pseudo_gradient()["weight"], torch.full((1, 2), 0.29))
        self.assertEqual(snapshot.local_steps, 2)

    def test_retry_reuses_capture_and_capacity_is_bounded_until_release(self):
        self.step()
        future = self.learner.request_snapshot("1", 0, 0)
        self.learner.boundary()
        first = future.result(timeout=1)
        self.step()
        self.assertIs(self.learner.request_snapshot("1", 0, 0), future)
        self.assertIs(future.result(), first)
        self.assertEqual(first.local_steps, 1)
        with self.assertRaises(SnapshotQueueFull):
            self.learner.request_snapshot("2", 1, 0)
        with self.assertRaises(ValueError):
            self.learner.request_snapshot("1", 1, 0)
        self.assertTrue(self.learner.release_snapshot("1"))
        with self.assertRaises(ValueError):
            self.learner.request_snapshot("1", 0, 0)
        next_capture = self.learner.request_snapshot("2", 1, 0)
        self.learner.boundary()
        self.assertEqual(next_capture.result(timeout=1).local_steps, 2)

    def test_receiver_request_during_backward_is_deferred_until_step_finishes(self):
        results = []
        with self.learner.training_step(8):
            self.optimizer.zero_grad()
            sum(p.sum() for p in self.model.parameters()).backward()

            def receiver():
                results.append(self.learner.request_snapshot("0", 0, 0))

            thread = threading.Thread(target=receiver)
            thread.start()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(results), 1)
            self.assertFalse(results[0].done())
            self.optimizer.step()
        snapshot = results[0].result(timeout=1)
        torch.testing.assert_close(snapshot.pseudo_gradient()["weight"], torch.full((1, 2), 0.1))
        self.assertEqual(snapshot.local_steps, 1)

    def test_incoming_replacement_is_deferred_and_preserves_other_fragment_and_optimizer_moments(self):
        self.step()
        moment = self.optimizer.state[self.model.weight]["momentum_buffer"].clone()
        old_weight = self.model.weight.detach().clone()
        with self.learner.training_step(16):
            thread = threading.Thread(target=lambda: self.queue(0, {"weight": torch.full((1, 2), 10.0)}, 1))
            thread.start()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            torch.testing.assert_close(self.model.weight, old_weight)
            # Complete a real second optimizer step, then the replacement wins
            # at its safe boundary. The bias's accumulated work is preserved.
            self.optimizer.zero_grad()
            sum(p.sum() for p in self.model.parameters()).backward()
            self.optimizer.step()
        torch.testing.assert_close(self.model.weight, torch.full((1, 2), 10.0))
        torch.testing.assert_close(self.optimizer.state[self.model.weight]["momentum_buffer"], moment * 0.9 + 1)
        metadata = self.learner.metadata()
        self.assertEqual((metadata.total_local_steps, metadata.total_tokens), (2, 24))
        self.assertEqual([(f.local_steps, f.tokens) for f in metadata.fragments], [(0, 0), (2, 24)])
        self.assertEqual(metadata.fragments[0].last_applied_revision, 1)

    def test_request_waits_for_local_overlap_without_stopping_training(self):
        future = self.learner.request_snapshot("0", 0, 0, min_local_steps=2)
        self.step()
        self.assertFalse(future.done())
        self.step()
        self.assertEqual(future.result(timeout=1).local_steps, 2)
        self.assertEqual(self.learner.metadata().total_local_steps, 2)

    def test_stale_request_fails_instead_of_capturing_a_different_baseline(self):
        self.step()
        future = self.learner.request_snapshot("0", 0, 0)
        self.queue(0, {"weight": torch.ones(1, 2)}, 1)
        self.learner.boundary()
        with self.assertRaises(StaleSnapshotRequest):
            future.result(timeout=1)
        self.learner.release_snapshot("0")
        future = self.learner.request_snapshot("1", 0, 1)
        self.step()
        self.assertEqual(future.result(timeout=1).base_revision, 1)

    def test_skipped_and_failed_steps_are_not_counted(self):
        self.learner.begin_step()
        self.learner.end_step(0, completed=False)
        with self.assertRaisesRegex(RuntimeError, "failed step"):
            with self.learner.training_step(8):
                raise RuntimeError("failed step")
        self.assertEqual(self.learner.metadata().total_local_steps, 0)
        self.assertEqual(self.learner.metadata().total_tokens, 0)
        self.step()
        self.assertEqual(self.learner.metadata().total_local_steps, 1)

    def test_boundary_rejects_wrong_thread_and_mid_step_access(self):
        errors = []

        def receiver():
            try:
                self.learner.boundary()
            except RuntimeError as exc:
                errors.append(str(exc))

        thread = threading.Thread(target=receiver)
        thread.start()
        thread.join(timeout=2)
        self.assertEqual(len(errors), 1)
        with self.learner.training_step(8):
            with self.assertRaises(RuntimeError):
                self.learner.boundary()
            self.optimizer.zero_grad()
            sum(p.sum() for p in self.model.parameters()).backward()
            self.optimizer.step()

    def test_cancelled_capture_is_not_serviced_and_release_allows_next_request(self):
        future = self.learner.request_snapshot("0", 0, 0)
        self.assertTrue(future.cancel())
        self.step()
        with self.assertRaises(CancelledError):
            future.result()
        self.assertTrue(self.learner.release_snapshot("0"))
        next_capture = self.learner.request_snapshot("1", 0, 0)
        self.learner.boundary()
        self.assertEqual(next_capture.result(timeout=1).local_steps, 1)


if __name__ == "__main__":
    unittest.main()
