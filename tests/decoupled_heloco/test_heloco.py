"""HeLoCo math against the unchanged implementation, plus dispatch lifecycle."""

import ast
import copy
from pathlib import Path
from typing import Any, Dict, Optional
import unittest
from unittest.mock import patch

import torch

from panoengine.decentralized.decoupled_heloco.config import HeLoCoConfig
from panoengine.decentralized.decoupled_heloco.correction import block_correct
from panoengine.decentralized.decoupled_heloco.fragment_manager import FragmentManager
from panoengine.decentralized.decoupled_heloco.learner import DecoupledLearner
from panoengine.decentralized.decoupled_heloco.optimizer import FragmentHeLoCo
from panoengine.decentralized.decoupled_heloco.syncer import DecoupledSyncer


def _reference_definitions():
    # Importing heloco.py would also import its server/torchft stack. Execute
    # only its two pure PyTorch definitions in tests, directly from that file.
    path = Path(__file__).resolve().parents[2] / "panoengine/decentralized/heloco.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {"block_correct", "HeLoCoOptimizer"}:
            node = copy.deepcopy(node)
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []  # Profiling does not affect the equations.
            selected.append(node)
    if len(selected) != 2:
        raise AssertionError("expected both unchanged HeLoCo reference definitions")
    namespace = {"torch": torch, "optim": torch.optim, "Any": Any, "Dict": Dict, "Optional": Optional}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["block_correct"], namespace["HeLoCoOptimizer"]


_reference_correct, _ReferenceOptimizer = _reference_definitions()


class CorrectionTests(unittest.TestCase):
    def test_aligned_anti_aligned_and_weak_blocks_match_hand_calculations(self):
        delta = {"aligned": torch.tensor([1.0, 0.0]), "anti": torch.tensor([-1.0, 0.0]), "weak": torch.tensor([0.0, 1.0])}
        moments = {name: torch.tensor([1.0, 0.0]) for name in delta}
        result = block_correct(delta, moments)
        torch.testing.assert_close(result["aligned"], delta["aligned"])
        torch.testing.assert_close(result["anti"], torch.tensor([-0.875, 0.0]))
        expected_weak = torch.tensor([0.25, 0.75])
        expected_weak /= expected_weak.norm()
        torch.testing.assert_close(result["weak"], expected_weak)
        torch.testing.assert_close(result["weak"].norm(), delta["weak"].norm())

    def test_exact_reference_equivalence_across_branches_dtypes_and_controls(self):
        for dtype in (torch.float32, torch.bfloat16):
            for controls in ({}, {"rho": 0.3, "c_ok": 0.4, "k_s": 1.0, "k_d": 0.5, "kappa": 0.5, "beta_max": 0.2}):
                with self.subTest(dtype=dtype, controls=controls):
                    delta = {
                        "aligned": torch.tensor([2.0, 1.0], dtype=dtype),
                        "anti": torch.tensor([-2.0, -1.0], dtype=dtype),
                        "weak": torch.tensor([0.0, 1.0], dtype=dtype),
                        "no_m": torch.tensor([1.0, 2.0], dtype=dtype),
                        "zero_m": torch.tensor([1.0, -1.0], dtype=dtype),
                        "zero_d": torch.zeros(2, dtype=dtype),
                    }
                    moments = {name: torch.tensor([1.0, 0.0]) for name in delta}
                    moments["aligned"] = torch.tensor([2.0, 1.0])
                    moments["no_m"] = None
                    moments["zero_m"] = torch.zeros(2)
                    before = {name: tensor.clone() for name, tensor in delta.items()}
                    expected = _reference_correct(delta, moments, **controls)
                    actual = block_correct(delta, moments, **controls)
                    for name in delta:
                        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
                        torch.testing.assert_close(delta[name], before[name], rtol=0, atol=0)
                        self.assertEqual(actual[name].dtype, dtype)

    def test_blocks_use_their_own_tensor_momentum(self):
        delta = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([1.0, 0.0])}
        result = block_correct(delta, {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([-1.0, 0.0])})
        torch.testing.assert_close(result["a"], delta["a"])
        torch.testing.assert_close(result["b"], torch.tensor([0.875, 0.0]))


class FragmentHeLoCoTests(unittest.TestCase):
    def make(self, *, fragments=1, lr=1.0, momentum=0.9, config=None):
        self.initial = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
        self.manager = FragmentManager(self.initial.items(), fragments)
        self.optimizer = FragmentHeLoCo(self.manager, self.initial, lr=lr, momentum=momentum, config=config)

    def test_multistep_weights_and_moments_match_existing_outer_optimizer(self):
        self.make(lr=0.7)
        parameters = {name: torch.nn.Parameter(value.clone()) for name, value in self.initial.items()}
        reference = _ReferenceOptimizer(parameters.values(), lr=0.7, momentum=0.9)
        gradients = (
            {"a": torch.tensor([0.1, 0.0]), "b": torch.tensor([0.0, 0.2])},
            {"a": torch.tensor([-0.1, 0.0]), "b": torch.tensor([0.2, 0.0])},
            {"a": torch.tensor([0.0, 0.1]), "b": torch.tensor([0.0, 0.0])},
        )
        for gradient in gradients:
            moments = {name: reference.state[param].get("m") for name, param in parameters.items()}
            corrected = _reference_correct(gradient, moments)
            for name, parameter in parameters.items():
                parameter.grad = corrected[name].clone()
            reference.step()
            outgoing = self.optimizer.step(0, gradient)
            for name, value in self.optimizer.model_snapshot().items():
                torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
                moment = reference.state[parameters[name]]["m"]
                torch.testing.assert_close(self.optimizer.momentum_snapshot(0)[name], moment, rtol=0, atol=0)
                torch.testing.assert_close(outgoing[name], parameters[name] - 0.7 * 0.9 * moment)

    def test_known_scalar_global_weights_and_lookahead_remain_distinct(self):
        initial = {"a": torch.ones(1)}
        optimizer = FragmentHeLoCo(FragmentManager(initial.items(), 1), initial, lr=1.0)
        for expected_weight, expected_dispatch, expected_moment in ((0.891, 0.882, 0.01), (0.7739, 0.7568, 0.019)):
            outgoing = optimizer.step(0, {"a": torch.tensor([0.1])})
            torch.testing.assert_close(optimizer.model_snapshot()["a"], torch.tensor([expected_weight]))
            torch.testing.assert_close(outgoing["a"], torch.tensor([expected_dispatch]))
            torch.testing.assert_close(optimizer.momentum_snapshot(0)["a"], torch.tensor([expected_moment]))
            torch.testing.assert_close(optimizer.dispatch_snapshot(0)["a"], outgoing["a"])

    def test_updating_one_fragment_leaves_other_weights_and_momentum_untouched(self):
        self.make(fragments=2)
        self.optimizer.step(0, {"a": torch.tensor([0.1, 0.2])})
        torch.testing.assert_close(self.optimizer.snapshot(1)["b"], self.initial["b"], rtol=0, atol=0)
        self.assertIsNone(self.optimizer.momentum_snapshot(1)["b"])
        first_moment = self.optimizer.momentum_snapshot(0)["a"]
        first_weight = self.optimizer.snapshot(0)["a"]
        self.optimizer.step(1, {"b": torch.tensor([-0.1, 0.1])})
        torch.testing.assert_close(self.optimizer.momentum_snapshot(0)["a"], first_moment, rtol=0, atol=0)
        torch.testing.assert_close(self.optimizer.snapshot(0)["a"], first_weight, rtol=0, atol=0)

    def test_failed_second_tensor_does_not_partially_commit_weights_or_momentum(self):
        self.make(lr=1e38)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.optimizer.step(0, {"a": torch.tensor([0.1, 0.1]), "b": torch.full((2,), 1e38)})
        for name, value in self.optimizer.model_snapshot().items():
            torch.testing.assert_close(value, self.initial[name], rtol=0, atol=0)
            self.assertIsNone(self.optimizer.momentum_snapshot(0)[name])

    def test_ablation_disables_correction_and_lookahead_but_retains_scaled_momentum(self):
        self.make(config=HeLoCoConfig(rho=0.5, correction_enabled=False, lookahead=False))
        with patch("panoengine.decentralized.decoupled_heloco.optimizer.block_correct") as correct:
            outgoing = self.optimizer.step(0, {name: torch.full((2,), 0.2) for name in self.initial})
        correct.assert_not_called()
        for name, value in self.initial.items():
            torch.testing.assert_close(self.optimizer.momentum_snapshot(0)[name], torch.full((2,), 0.01))
            torch.testing.assert_close(outgoing[name], value - 0.109)
            torch.testing.assert_close(outgoing[name], self.optimizer.snapshot(0)[name])

    def test_returned_weights_and_momentum_snapshots_cannot_mutate_optimizer(self):
        self.make()
        outgoing = self.optimizer.step(0, {name: torch.full((2,), 0.1) for name in self.initial})
        weights = self.optimizer.model_snapshot()
        moment = self.optimizer.momentum_snapshot(0)["a"]
        outgoing["a"].zero_()
        moment.zero_()
        for name, value in self.optimizer.model_snapshot().items():
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
        torch.testing.assert_close(self.optimizer.momentum_snapshot(0)["a"], torch.full((2,), 0.01))


class _ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


class HeLoCoSyncerTests(unittest.TestCase):
    def make(self, *, count=1):
        global_model = _ScalarModel()
        self.models = [copy.deepcopy(global_model) for _ in range(count)]
        self.learners = [DecoupledLearner(model, 1, learner_id=i) for i, model in enumerate(self.models)]
        self.syncer = DecoupledSyncer(global_model, self.learners, 1, min_quorum=count, overlap_steps=1, outer_lr=1.0)

    def train(self, i, delta, tokens=8):
        optimizer = torch.optim.SGD(self.models[i].parameters(), lr=1.0)
        with self.learners[i].training_step(tokens):
            optimizer.zero_grad()
            (delta * self.models[i].weight.sum()).backward()
            optimizer.step()

    def capture(self):
        self.syncer.begin_sync()
        for learner in self.learners:
            learner.boundary()

    def apply(self):
        for learner in self.learners:
            learner.boundary()

    def test_learner_baseline_uses_dispatched_lookahead_on_the_next_capture(self):
        self.make()
        self.train(0, 0.1)
        self.capture()
        self.syncer.poll()
        self.apply()
        torch.testing.assert_close(self.models[0].weight, torch.tensor([0.882]))
        torch.testing.assert_close(self.syncer.optimizer.model_snapshot()["weight"], torch.tensor([0.891]))
        self.train(0, 0.1)
        self.capture()
        with patch("panoengine.decentralized.decoupled_heloco.optimizer.block_correct", wraps=block_correct) as correct:
            self.syncer.poll()
        # A baseline of global theta would falsely report .109 here.
        torch.testing.assert_close(correct.call_args.args[0]["weight"], torch.tensor([0.1]))
        self.apply()
        torch.testing.assert_close(self.models[0].weight, torch.tensor([0.7568]))
        torch.testing.assert_close(self.syncer.optimizer.model_snapshot()["weight"], torch.tensor([0.7739]))

    def test_weighted_quorum_is_merged_once_before_tensor_correction(self):
        self.make(count=2)
        self.train(0, 0.1, tokens=8)
        self.train(1, 0.3, tokens=24)
        self.capture()
        with patch("panoengine.decentralized.decoupled_heloco.optimizer.block_correct", wraps=block_correct) as correct:
            result = self.syncer.poll()
        self.assertEqual(correct.call_count, 1)
        torch.testing.assert_close(correct.call_args.args[0]["weight"], torch.tensor([0.25]))
        self.assertEqual(result.weights, (0.25, 0.75))
        torch.testing.assert_close(self.syncer.optimizer.model_snapshot()["weight"], torch.tensor([0.7275]))
        self.apply()
        for model in self.models:
            torch.testing.assert_close(model.weight, torch.tensor([0.705]))

    def test_retrying_failed_lookahead_delivery_does_not_advance_momentum_twice(self):
        self.make()
        self.train(0, 0.1)
        self.capture()
        with patch.object(self.learners[0], "queue_update", side_effect=RuntimeError("temporary delivery failure")):
            result = self.syncer.poll()
        self.assertEqual(result.pending_broadcast, (0,))
        moment = self.syncer.optimizer.momentum_snapshot(0)["weight"]
        self.assertEqual(self.syncer.retry_broadcast(), ())
        torch.testing.assert_close(self.syncer.optimizer.momentum_snapshot(0)["weight"], moment, rtol=0, atol=0)
        self.assertEqual(self.syncer.fragment_revisions, (1,))
        self.apply()
        torch.testing.assert_close(self.models[0].weight, torch.tensor([0.882]))


if __name__ == "__main__":
    unittest.main()
