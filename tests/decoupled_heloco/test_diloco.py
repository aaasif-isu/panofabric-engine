"""Nesterov reference math, routing, matched configs and real CPU/TCP runs."""

import copy
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

import prepare_decoupled_comparison as preparation
import run_decoupled_heloco as launcher
from panoengine.decentralized.decoupled_heloco.config import ConfigError, DecoupledConfig, ExperimentConfig, load_config
from panoengine.decentralized.decoupled_heloco.evaluation import load_global_parameters
from panoengine.decentralized.decoupled_heloco.fragment_manager import FragmentManager
from panoengine.decentralized.decoupled_heloco.gpu_training import STOPPING_POLICY, _run_training
from panoengine.decentralized.decoupled_heloco.learner import DecoupledLearner
from panoengine.decentralized.decoupled_heloco.optimizer import FragmentDiLoCo
from panoengine.decentralized.decoupled_heloco.process_smoke import run_process_smoke
from panoengine.decentralized.decoupled_heloco.smoke import _TinyTokenModel
from panoengine.decentralized.decoupled_heloco.syncer import DecoupledSyncer

ROOT = Path(__file__).resolve().parents[2]


class DiLoCoOptimizerTests(unittest.TestCase):
    def make(self, *, fragments=1, momentum=0.9, lr=0.7):
        initial = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([-1.0, 3.0])}
        manager = FragmentManager(initial.items(), fragments)
        return initial, FragmentDiLoCo(manager, initial, lr=lr, momentum=momentum)

    def test_matches_torch_nesterov_with_interleaved_fragments_and_changing_gradients(self):
        for momentum in (0.0, 0.4, 0.9):
            for fragments in (1, 2):
                with self.subTest(momentum=momentum, fragments=fragments):
                    initial, actual = self.make(fragments=fragments, momentum=momentum)
                    parameters = {name: torch.nn.Parameter(value.clone()) for name, value in initial.items()}
                    reference = torch.optim.SGD(parameters.values(), lr=0.7, momentum=momentum,
                                                nesterov=bool(momentum), dampening=0, foreach=False)
                    for step in range(9):
                        fragment = step % fragments
                        names = actual.manager.fragment(fragment).parameter_names
                        gradient = {name: torch.tensor([0.03 * (step + 1), (-1.0) ** step * 0.2]) for name in names}
                        reference.zero_grad(set_to_none=True)
                        for name in names:
                            parameters[name].grad = gradient[name].clone()
                        reference.step()
                        outgoing = actual.step(fragment, gradient)
                        for name, value in actual.model_snapshot().items():
                            torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
                        for name in names:
                            torch.testing.assert_close(outgoing[name], parameters[name], rtol=0, atol=0)
                            buffer = actual.momentum_snapshot(fragment)[name]
                            expected = reference.state[parameters[name]].get("momentum_buffer")
                            if expected is None:
                                self.assertIsNone(buffer)
                            else:
                                torch.testing.assert_close(buffer, expected, rtol=0, atol=0)

    def test_known_scalar_sum_momentum_and_plain_global_dispatch(self):
        initial = {"a": torch.ones(1)}
        actual = FragmentDiLoCo(FragmentManager(initial.items(), 1), initial, lr=1.0)
        for weight, moment in ((0.81, 0.1), (0.539, 0.19)):
            outgoing = actual.step(0, {"a": torch.tensor([0.1])})
            torch.testing.assert_close(outgoing["a"], torch.tensor([weight]))
            torch.testing.assert_close(actual.momentum_snapshot(0)["a"], torch.tensor([moment]))
            torch.testing.assert_close(actual.dispatch_snapshot(0)["a"], actual.snapshot(0)["a"], rtol=0, atol=0)

    def test_other_fragments_and_input_snapshots_do_not_alias_or_change(self):
        initial, actual = self.make(fragments=2)
        gradient = {"a": torch.tensor([0.1, 0.2])}
        gradient_before = gradient["a"].clone()
        outgoing = actual.step(0, gradient)
        outgoing["a"].zero_()
        initial["a"].zero_()
        actual.snapshot(0)["a"].zero_()
        actual.momentum_snapshot(0)["a"].zero_()
        torch.testing.assert_close(gradient["a"], gradient_before, rtol=0, atol=0)
        torch.testing.assert_close(actual.snapshot(0)["a"], torch.tensor([0.867, 1.734]))
        torch.testing.assert_close(actual.snapshot(1)["b"], initial["b"], rtol=0, atol=0)
        self.assertIsNone(actual.momentum_snapshot(1)["b"])
        torch.testing.assert_close(actual.momentum_snapshot(0)["a"], gradient_before, rtol=0, atol=0)

    def test_bad_second_tensor_never_partially_commits_weights_or_momentum(self):
        _, actual = self.make(lr=1e38)
        actual.step(0, {"a": torch.full((2,), 1e-38), "b": torch.full((2,), 1e-38)})
        before_weights, before_moments = actual.model_snapshot(), actual.momentum_snapshot(0)
        for invalid in (float("nan"), 1e38):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "nonfinite"):
                actual.step(0, {"a": torch.full((2,), 0.1), "b": torch.full((2,), invalid)})
            for name, value in actual.model_snapshot().items():
                torch.testing.assert_close(value, before_weights[name], rtol=0, atol=0)
                torch.testing.assert_close(actual.momentum_snapshot(0)[name], before_moments[name], rtol=0, atol=0)

    def test_invalid_controls_fail_at_construction(self):
        for kwargs in ({"momentum": True}, {"momentum": 1}, {"momentum": -0.1}, {"momentum": float("nan")}, {"lr": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make(**kwargs)

    def test_token_weighted_quorum_then_one_nesterov_step_broadcasts_to_noncontributor(self):
        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(1.0)
        local = [copy.deepcopy(model) for _ in range(3)]
        learners = [DecoupledLearner(m, 1, learner_id=i) for i, m in enumerate(local)]
        syncer = DecoupledSyncer(model, learners, 1, min_quorum=2, overlap_steps=1,
                                outer_method="diloco", outer_lr=1.0, weighting="tokens")
        for index, tokens, delta in ((0, 10, 0.1), (1, 30, 0.3)):
            with learners[index].training_step(tokens=tokens), torch.no_grad():
                for parameter in local[index].parameters():
                    parameter.sub_(delta)
        syncer.begin_sync()
        for learner in learners:
            learner.boundary()
        result = syncer.poll()
        self.assertEqual(result.learner_ids, (0, 1))
        self.assertEqual(result.weights, (0.25, 0.75))
        self.assertEqual(syncer.fragment_revisions, (1,))
        for learner, local_model in zip(learners, local):
            learner.boundary()
            self.assertEqual(learner.metadata().fragments[0].last_applied_revision, 1)
            for parameter in local_model.parameters():
                torch.testing.assert_close(parameter, torch.full_like(parameter, 0.525))
        self.assertIsNone(syncer.poll())
        self.assertEqual(syncer.fragment_revisions, (1,))


class DiLoCoWorkflowTests(unittest.TestCase):
    def test_selected_method_routes_every_action_without_legacy_training(self):
        with patch.object(launcher.subprocess, "call") as legacy, redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.main(["--method", "decoupled_diloco", "--check-config"]), 0)
            self.assertEqual(launcher.main(["--method", "decoupled_diloco", "--dry-run"]), 0)
            with patch("panoengine.decentralized.decoupled_heloco.gpu_training.run_gpu_training", return_value=0) as train:
                self.assertEqual(launcher.main(["--method", "decoupled_diloco"]), 0)
                self.assertEqual(train.call_args.args[0].fragment_outer_method, "diloco")
                self.assertEqual(launcher.main(["--method", "decoupled_diloco", "--check-training"]), 0)
                self.assertTrue(train.call_args.kwargs["check_only"])
            with patch("panoengine.decentralized.decoupled_heloco.smoke.run_smoke", return_value=0) as smoke:
                self.assertEqual(launcher.main(["--method", "decoupled_diloco", "--smoke-test"]), 0)
                self.assertEqual(smoke.call_args.kwargs["outer_method"], "diloco")
            with patch("panoengine.decentralized.decoupled_heloco.process_smoke.run_process_smoke", return_value=0) as smoke:
                self.assertEqual(launcher.main(["--method", "decoupled_diloco", "--process-smoke-test"]), 0)
                self.assertEqual(smoke.call_args.args[0].fragment_outer_method, "diloco")
            legacy.assert_not_called()

    def test_real_process_smoke_uses_diloco_and_shuts_down_cleanly(self):
        config = ExperimentConfig("decoupled_diloco", run={"seed": 42, "outer_lr": 0.7},
                                  decoupled=DecoupledConfig(num_fragments=2, min_quorum=2, overlap_steps=2,
                                                          sync_interval=0.1, grace_window_factor=0.2))
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            self.assertEqual(run_process_smoke(config, learners=2, cycles=2, timeout=30.0), 0, output.getvalue())
        self.assertIn("Outer optimizer: diloco", output.getvalue())
        self.assertIn("PROCESS SMOKE TEST PASSED", output.getvalue())

    def test_real_fixed_budget_coordinator_exports_tagged_diloco_checkpoint_and_tail(self):
        config = ExperimentConfig("decoupled_diloco", decoupled=DecoupledConfig(
            num_fragments=2, min_quorum=2, overlap_steps=2, sync_interval=0.1, grace_window_factor=0.2))
        options = SimpleNamespace(islands=2, steps=12, outer_lr=0.7, outer_momentum=0.9,
                                  ps_timeout=30.0, island_slowness_factors=[1, 2])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model = _TinyTokenModel(2)
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                status = _run_training(config, options, model, ["0", "1"], folder, 30.0, ROOT,
                                       worker_command=[sys.executable, str(Path(__file__).with_name("cpu_training_worker.py"))])
            self.assertEqual(status, 0, output.getvalue())
            summary = json.loads((folder / "summary.json").read_text())
            self.assertEqual(summary["method"], "decoupled_diloco")
            self.assertEqual(summary["outer_optimizer"]["name"], "nesterov_sgd")
            self.assertFalse(summary["outer_optimizer"]["lookahead"])
            self.assertFalse(summary["outer_optimizer"]["correction_enabled"])
            self.assertEqual(summary["stopping_policy"], STOPPING_POLICY)
            self.assertEqual([l["total_local_steps"] for l in summary["learners"]], [12, 12])
            self.assertEqual([l["total_tokens"] for l in summary["learners"]], [192, 192])
            payload = torch.load(folder / "global_model.pt", weights_only=True)
            self.assertEqual(payload["method"], "decoupled_diloco")
            load_global_parameters(model, payload, num_fragments=2,
                                   expected_revisions=summary["fragment_revisions"], expected_method="decoupled_diloco")
            before = {name: value.clone() for name, value in model.named_parameters()}
            with self.assertRaisesRegex(ValueError, "method differs"):
                load_global_parameters(model, payload, num_fragments=2, expected_method="decoupled_heloco")
            for name, value in model.named_parameters():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)


class ComparisonPreparationTests(unittest.TestCase):
    def fixture(self, directory):
        folder = Path(directory) / "reference"
        folder.mkdir()
        data = yaml.safe_load((ROOT / "run_script" / "decoupled_heloco.yaml").read_text())
        data["run"].update(steps=500, island_slowness_factors=[1, 2, 3, 1])
        config_file = folder / "experiment.yaml"
        config_file.write_text(yaml.safe_dump(data))
        config = load_config(config_file)
        effective = folder / "legacy.yaml"
        effective.write_text(yaml.safe_dump(config.legacy_options()))
        options = launcher._parse_legacy_options(launcher._load_legacy_launcher(), effective)
        (folder / "run-options.json").write_text(json.dumps(vars(options)))
        (folder / "summary.json").write_text(json.dumps({"status": "passed", "steps_per_learner": 500}))
        (folder / "global_model.pt").write_bytes(b"fixture: preparation reads metadata only")
        return folder, config, options

    def test_reuses_recorded_settings_and_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, original, options = self.fixture(directory)
            before = {p.name: p.read_bytes() for p in folder.iterdir()}
            destination = Path(directory) / "diloco.yaml"
            target = preparation.prepare_comparison(folder, destination)
            self.assertEqual(target.method, "decoupled_diloco")
            self.assertEqual(target.decoupled, original.decoupled)
            self.assertEqual(target.run["steps"], 500)
            self.assertEqual(target.run["island_slowness_factors"], [1, 2, 3, 1])
            self.assertEqual(target.run["log_dir"], "outputs/decoupled_diloco")
            for field in preparation.MATCHED_RUN_FIELDS:
                self.assertEqual(target.run[field], getattr(options, field))
            loaded = load_config(destination)
            self.assertEqual(loaded, target)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(launcher.main(["--config-file", str(destination), "--check-config"]), 0)
            content = destination.read_bytes()
            with self.assertRaises(FileExistsError):
                preparation.prepare_comparison(folder, destination)
            self.assertEqual(destination.read_bytes(), content)
            self.assertEqual({p.name: p.read_bytes() for p in folder.iterdir()}, before)

    def test_resolved_token_budget_is_copied_as_fixed_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, _, options = self.fixture(directory)
            data = yaml.safe_load((folder / "experiment.yaml").read_text())
            data["run"].update(steps=30, tokens_per_parameter=1.0)
            (folder / "experiment.yaml").write_text(yaml.safe_dump(data))
            options.tokens_per_parameter = 1.0
            (folder / "run-options.json").write_text(json.dumps(vars(options)))
            result = preparation.prepare_comparison(folder, Path(directory) / "diloco.yaml")
            self.assertEqual(result.run["steps"], 500)
            self.assertIsNone(result.run["tokens_per_parameter"])

    def test_missing_checkpoint_failed_run_or_disagreeing_recipe_prevents_creation(self):
        for condition in ("checkpoint", "failed", "seed", "budget"):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as directory:
                folder, _, options = self.fixture(directory)
                if condition == "checkpoint":
                    (folder / "global_model.pt").unlink()
                elif condition == "failed":
                    (folder / "summary.json").write_text('{"status": "failed"}')
                elif condition == "seed":
                    options.seed += 1
                    (folder / "run-options.json").write_text(json.dumps(vars(options)))
                else:
                    (folder / "summary.json").write_text('{"status": "passed", "steps_per_learner": 30}')
                destination = Path(directory) / "diloco.yaml"
                with self.assertRaises(ConfigError):
                    preparation.prepare_comparison(folder, destination)
                self.assertFalse(destination.exists())

    def test_cli_creates_config_and_missing_metadata_returns_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, _, _ = self.fixture(directory)
            destination = Path(directory) / "diloco.yaml"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(preparation.main(["--reference-run", str(folder), "--output-config", str(destination)]), 0)
                self.assertEqual(preparation.main(["--reference-run", str(folder / "missing")]), 2)
            self.assertEqual(load_config(destination).method, "decoupled_diloco")


if __name__ == "__main__":
    unittest.main()
