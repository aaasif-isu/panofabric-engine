"""Baseline, work-accounting, and replacement checks without networking."""

import unittest

import torch

from panoengine.decentralized.decoupled_heloco.fragment_manager import FragmentManager
from panoengine.decentralized.decoupled_heloco.state import FragmentState


class FragmentStateTests(unittest.TestCase):
    def setUp(self):
        self.parameters = {
            "a": torch.nn.Parameter(torch.tensor([1.0, 2.0])),
            "b": torch.nn.Parameter(torch.tensor([3.0, 4.0])),
        }
        self.manager = FragmentManager(self.parameters.items(), 2)
        self.states = [FragmentState(self.manager, i, self.parameters) for i in range(2)]

    def queue(self, state, update, revision):
        return state.queue_update(update, revision, layout_signature=self.manager.layout_signature)

    def test_real_local_steps_produce_expected_pseudo_gradients_without_consuming_work(self):
        optimizer = torch.optim.SGD(self.parameters.values(), lr=0.1)
        for tokens in (8, 12, 16):
            optimizer.zero_grad()
            sum(tensor.sum() for tensor in self.parameters.values()).backward()
            optimizer.step()
            for state in self.states:
                state.record_step(tokens)
        for i, name in enumerate(("a", "b")):
            snapshot = self.states[i].snapshot(self.parameters)
            torch.testing.assert_close(snapshot.pseudo_gradient()[name], torch.full((2,), 0.3))
            self.assertEqual((snapshot.local_steps, snapshot.tokens, snapshot.base_revision), (3, 36, 0))
            self.assertFalse(snapshot.current[name].requires_grad)
            self.assertEqual(snapshot.current[name].device.type, "cpu")
            self.assertEqual(self.states[i].metadata().local_steps, 3)
            repeated = self.states[i].snapshot(self.parameters)
            torch.testing.assert_close(snapshot.pseudo_gradient()[name], repeated.pseudo_gradient()[name])

    def test_snapshots_and_baselines_do_not_alias_live_parameters_or_each_other(self):
        first = self.states[0].snapshot(self.parameters)
        first.baseline["a"].fill_(100)
        first.current["a"].fill_(200)
        with torch.no_grad():
            self.parameters["a"].add_(0.5)
        second = self.states[0].snapshot(self.parameters)
        torch.testing.assert_close(second.baseline["a"], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(second.current["a"], torch.tensor([1.5, 2.5]))

    def test_queue_does_not_mutate_live_weights_and_replacement_resets_only_one_fragment(self):
        for state in self.states:
            state.record_step(16)
        incoming = {"a": torch.tensor([10.0, 20.0])}
        self.assertTrue(self.queue(self.states[0], incoming, 1))
        incoming["a"].fill_(100)  # Received buffers may be reused by networking.
        torch.testing.assert_close(self.parameters["a"], torch.tensor([1.0, 2.0]))
        metadata = self.states[0].metadata()
        self.assertEqual((metadata.last_server_revision, metadata.last_applied_revision), (1, 0))
        self.assertEqual(metadata.tokens, 16)
        self.assertTrue(self.states[0].apply_pending(self.parameters))
        torch.testing.assert_close(self.parameters["a"], torch.tensor([10.0, 20.0]))
        torch.testing.assert_close(self.parameters["b"], torch.tensor([3.0, 4.0]))
        self.assertEqual(self.states[0].metadata().local_steps, 0)
        self.assertEqual(self.states[0].metadata().tokens, 0)
        self.assertEqual(self.states[1].metadata().local_steps, 1)
        self.assertEqual(self.states[1].metadata().tokens, 16)
        snapshot = self.states[0].snapshot(self.parameters)
        torch.testing.assert_close(snapshot.pseudo_gradient()["a"], torch.zeros(2))
        self.assertEqual(snapshot.base_revision, 1)
        self.assertFalse(self.states[0].apply_pending(self.parameters))

    def test_out_of_order_and_duplicate_updates_never_roll_weights_back(self):
        state = self.states[0]
        self.assertTrue(self.queue(state, {"a": torch.ones(2)}, 3))
        self.assertFalse(self.queue(state, {"a": torch.zeros(2)}, 2))
        self.assertFalse(self.queue(state, {"a": torch.zeros(2)}, 3))
        state.apply_pending(self.parameters)
        self.assertFalse(self.queue(state, {"a": torch.zeros(2)}, 1))
        self.assertTrue(self.queue(state, {"a": torch.full((2,), 4.0)}, 4))
        state.apply_pending(self.parameters)
        torch.testing.assert_close(self.parameters["a"], torch.full((2,), 4.0))
        self.assertEqual(state.metadata().last_applied_revision, 4)

    def test_invalid_update_keeps_pending_revision_and_model_unchanged(self):
        state = self.states[0]
        self.queue(state, {"a": torch.full((2,), 9.0)}, 1)
        for update in ({"b": torch.ones(2)}, {"a": torch.ones(3)}, {"a": torch.ones(2), "b": torch.ones(2)}):
            with self.assertRaises(ValueError):
                self.queue(state, update, 2)
        with self.assertRaises(ValueError):
            state.queue_update({"a": torch.zeros(2)}, 2, layout_signature="different layout")
        self.assertEqual(state.metadata().last_server_revision, 1)
        torch.testing.assert_close(self.parameters["a"], torch.tensor([1.0, 2.0]))
        state.apply_pending(self.parameters)
        torch.testing.assert_close(self.parameters["a"], torch.full((2,), 9.0))

    def test_bf16_baseline_matches_weights_actually_applied(self):
        local = {"a": torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))}
        manager = FragmentManager(local.items(), 1)
        state = FragmentState(manager, 0, local)
        update = {"a": torch.tensor([1.001, 1.003])}
        state.queue_update(update, 1, layout_signature=manager.layout_signature)
        state.apply_pending(local)
        self.assertFalse(torch.equal(local["a"].float(), update["a"]))
        torch.testing.assert_close(state.snapshot(local).pseudo_gradient()["a"], torch.zeros(2), rtol=0, atol=0)

    def test_full_fragment_validation_precedes_any_replacement(self):
        manager = FragmentManager(self.parameters.items(), 1)
        state = FragmentState(manager, 0, self.parameters)
        state.queue_update({"a": torch.zeros(2), "b": torch.zeros(2)}, 1, layout_signature=manager.layout_signature)
        invalid_targets = {"a": self.parameters["a"], "b": torch.zeros(3)}
        with self.assertRaises(ValueError):
            state.apply_pending(invalid_targets)
        torch.testing.assert_close(self.parameters["a"], torch.tensor([1.0, 2.0]))
        self.assertEqual(state.metadata().last_applied_revision, 0)
        self.assertTrue(state.apply_pending(self.parameters))

    def test_nonmaterialized_parameters_and_bad_work_counts_are_rejected(self):
        meta = {"a": torch.empty(2, device="meta")}
        with self.assertRaises(ValueError):
            FragmentState(self.manager, 0, meta)
        for tokens in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                self.states[0].record_step(tokens)
        for revision in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                self.queue(self.states[0], {"a": torch.zeros(2)}, revision)


if __name__ == "__main__":
    unittest.main()
