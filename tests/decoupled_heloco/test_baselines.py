"""Original optimizers/client/server over real HTTP with CPU model fixtures."""

from contextlib import redirect_stdout, redirect_stderr
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

import prepare_decoupled_comparison as preparation
import run_decoupled_heloco as old_launcher
import run_matched_baselines as launcher
from panoengine.decentralized.decoupled_heloco.baseline_options import (
    BASELINE_FORMAT, BASELINE_SCOPE, BASELINE_STOPPING, baseline_metadata, validate_baseline_options,
)
from panoengine.decentralized.decoupled_heloco.baseline_training import _run_baseline, build_baseline_server
from panoengine.decentralized.decoupled_heloco.config import ConfigError, ExperimentConfig, LEGACY_METHODS, load_config
from panoengine.decentralized.decoupled_heloco.evaluation import load_global_parameters, parameter_fingerprint
from panoengine.decentralized.decoupled_heloco.global_validation import load_run
from panoengine.decentralized.decoupled_heloco.smoke import _TinyTokenModel

ROOT = Path(__file__).resolve().parents[2]


def options_for(config):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "legacy.yaml"
        path.write_text(yaml.safe_dump(config.legacy_options()))
        return old_launcher._parse_legacy_options(old_launcher._load_legacy_launcher(), path)


def configuration(method, **changes):
    source = load_config(ROOT / "decoupled_heloco.yaml")
    run = {**source.run, "islands": 2, "gpus": [0, 1], "steps": 12, "sync_steps": 3,
           "ps_timeout": 30.0, "island_slowness_factors": [1, 2], **changes}
    return ExperimentConfig(method, run=run, decoupled=source.decoupled)


try:
    from panoengine.decentralized.async_diloco import AsyncDiLoCo, AsyncDiLoCoServer, DelayedNesterovOptimizer
    from panoengine.decentralized.heloco import HeLoCoOptimizer, HeLoCoServer
    from panoengine.decentralized.mla import MLAOptimizer, MLAServer
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False


class BaselineConfigurationTests(unittest.TestCase):
    def test_all_methods_validate_without_torch_and_never_use_original_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            for method in LEGACY_METHODS:
                path = Path(directory) / f"{method}.yaml"
                cfg = configuration(method)
                path.write_text(yaml.safe_dump({"method": method, "run": cfg.run}))
                # Assert the validation route never imports either runtime,
                # even when the test environment has them available.
                code = "import sys; import run_matched_baselines as launcher; status=launcher.main(sys.argv[1:]); assert 'torch' not in sys.modules and 'torchft' not in sys.modules; raise SystemExit(status)"
                result = subprocess.run([sys.executable, "-c", code, "--config-file", str(path), "--check-config"],
                                        cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                with patch("panoengine.decentralized.decoupled_heloco.baseline_training.run_baseline_training", return_value=0) as training, patch.object(old_launcher.subprocess, "call") as legacy, redirect_stdout(io.StringIO()):
                    self.assertEqual(launcher.main(["--config-file", str(path), "--check-training"]), 0)
                    self.assertTrue(training.call_args.kwargs["check_only"])
                    self.assertEqual(training.call_args.args[1].outer_lr, 0.7)
                    legacy.assert_not_called()

    def test_rejects_partial_windows_sync_fragment_hybrid_and_recipe_overrides(self):
        for changes in ({"steps": 11}, {"coordination_method": "sync"}, {"num_fragments": 3},
                        {"correction_workers": "none"}, {"async_interval": 2}, {"diloco_lr": 0.3}):
            with self.subTest(changes=changes), self.assertRaises(ConfigError):
                config = configuration("heloco", **changes)
                validate_baseline_options(config, options_for(config))
        config = configuration("decoupled_heloco")
        with self.assertRaisesRegex(ConfigError, "requires heloco"):
            validate_baseline_options(config, options_for(config))

    def test_original_launcher_defaults_preserved_but_lr_is_explicit(self):
        for method in LEGACY_METHODS:
            config = configuration(method, islands=4, gpus=[0, 1, 2, 3], island_slowness_factors=[1, 2, 3, 1])
            metadata = baseline_metadata(config, options_for(config))
            self.assertEqual(metadata["lr"], 0.7)
            self.assertEqual(metadata["rho"], 0.5 if method == "heloco" else None)
            self.assertEqual(metadata["nesterov_period"], 4 if method == "diloco" else None)
            self.assertEqual(metadata["lookahead"], method == "heloco")

    def test_prepare_all_three_from_recorded_500_step_budget_without_touching_source(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "reference"
            folder.mkdir()
            config = load_config(ROOT / "decoupled_heloco.yaml")
            config = ExperimentConfig(config.method, run={**config.run, "steps": 500, "island_slowness_factors": [1, 2, 3, 1]}, decoupled=config.decoupled)
            saved = options_for(config)
            (folder / "experiment.yaml").write_text(yaml.safe_dump({"method": config.method, "run": config.run}))
            (folder / "run-options.json").write_text(json.dumps(vars(saved)))
            (folder / "summary.json").write_text(json.dumps({"status": "passed", "steps_per_learner": 500}))
            (folder / "global_model.pt").write_bytes(b"preparation reads only metadata")
            before = {p.name: p.read_bytes() for p in folder.iterdir()}
            for method in LEGACY_METHODS:
                destination = Path(directory) / f"{method}.yaml"
                target = preparation.prepare_comparison(folder, destination, method=method)
                self.assertEqual(target.run["sync_steps"], saved.sync_steps)
                self.assertEqual(target.run["steps"], 500)
                self.assertEqual(target.run["island_slowness_factors"], [1, 2, 3, 1])
                self.assertEqual(target.run["log_dir"], f"outputs/{method}")
                self.assertEqual(load_config(destination), target)
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(launcher.main(["--config-file", str(destination), "--check-config"]), 0)
            self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})


@unittest.skipUnless(HTTP_AVAILABLE, "requires the project's TorchFT HTTP helpers")
class OriginalHTTPTests(unittest.TestCase):
    def test_first_real_http_update_and_dispatch_match_each_original_algorithm(self):
        for method, server_type, optimizer_type in (("heloco", HeLoCoServer, HeLoCoOptimizer),
                                                   ("mla", MLAServer, MLAOptimizer),
                                                   ("diloco", AsyncDiLoCoServer, DelayedNesterovOptimizer)):
            with self.subTest(method=method), patch.dict(os.environ, {"NO_PROXY": "localhost,127.0.0.1,::1,[::1]", "no_proxy": "localhost,127.0.0.1,::1,[::1]", "PF_WIRE_BF16": "1", "PF_ISLAND_ID": "0"}):
                config = configuration(method)
                options = options_for(config)
                global_model, local = torch.nn.Linear(2, 1, bias=False), torch.nn.Linear(2, 1, bias=False)
                with torch.no_grad():
                    global_model.weight.fill_(1)
                server = build_baseline_server(global_model, config, options)
                try:
                    self.assertIs(type(server), server_type)
                    self.assertIs(type(server._outer_optimizer), optimizer_type)
                    inner = torch.optim.AdamW(local.parameters(), lr=0.01)
                    with AsyncDiLoCo(server.address(), local, inner, sync_every=3, wire_bf16=False) as client:
                        for handle in client._hooks:
                            handle.remove()
                        client._hooks.clear()
                        self.assertFalse(client._wire_bf16)
                        initial = local.weight.detach().clone()
                        # Real AdamW moments must survive every replacement.
                        local.weight.grad = torch.ones_like(local.weight)
                        inner.step()
                        moments_before = parameter_fingerprint(inner.state[local.weight])
                        with torch.no_grad():
                            local.weight.sub_(torch.tensor([[0.1, -0.2]]))
                        delta = initial - local.weight.detach()
                        gradient = delta * baseline_metadata(config, options)["rho"] if method == "heloco" else delta
                        if method == "diloco":
                            expected_global = initial - 0.7 * gradient / options.islands
                        else:
                            expected_global = initial - 0.7 * (gradient + 0.9 * 0.1 * gradient)
                        client._local_step = 3
                        client._sync()
                        self.assertEqual(server.status()["applied_pushes"], 1)
                        self.assertEqual(client._baseline_revision, 1)
                        torch.testing.assert_close(global_model.weight, expected_global)
                        dispatch = expected_global - 0.7 * 0.9 * 0.1 * gradient if method == "heloco" else expected_global
                        torch.testing.assert_close(local.weight, dispatch)
                        self.assertEqual(parameter_fingerprint(inner.state[local.weight]), moments_before)
                        client._pull_global()
                        self.assertEqual(server.status()["applied_pushes"], 1)
                        self.assertEqual(parameter_fingerprint(inner.state[local.weight]), moments_before)
                finally:
                    server.shutdown()

    def test_separate_cpu_learners_commit_every_window_and_export_global_weights_for_all_methods(self):
        for method in LEGACY_METHODS:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                config = configuration(method)
                options = options_for(config)
                model = _TinyTokenModel(2)
                initial_hash = parameter_fingerprint(dict(model.named_parameters()))
                output = io.StringIO()
                with redirect_stdout(output), redirect_stderr(output):
                    status = _run_baseline(config, options, model, ["0", "1"], folder, 30.0, ROOT,
                                           worker_command=[sys.executable, str(Path(__file__).with_name("cpu_baseline_worker.py"))])
                self.assertEqual(status, 0, output.getvalue() + "\n" + "\n".join(p.read_text() for p in folder.glob("learner_*/trainer.log")))
                summary = json.loads((folder / "summary.json").read_text())
                self.assertEqual(summary["scope"], BASELINE_SCOPE)
                self.assertEqual(summary["stopping_policy"], BASELINE_STOPPING)
                self.assertEqual(summary["initial_parameters_sha256"], initial_hash)
                self.assertEqual(summary["fragment_revisions"], [8])
                self.assertEqual(summary["sync_updates"], 8)
                self.assertEqual([r["total_local_steps"] for r in summary["learners"]], [12, 12])
                self.assertEqual([r["total_tokens"] for r in summary["learners"]], [192, 192])
                payload = torch.load(folder / "global_model.pt", weights_only=True)
                self.assertEqual(payload["format"], BASELINE_FORMAT)
                self.assertEqual(parameter_fingerprint(payload["parameters"]), summary["global_parameters_sha256"])
                self.assertEqual(summary["final_dispatch_sha256"] == summary["global_parameters_sha256"], method != "heloco")
                for learner_id in range(2):
                    with (folder / f"learner_{learner_id}/windows.csv").open() as stream:
                        windows = list(csv.DictReader(stream))
                    self.assertEqual(len(windows), 4)
                    self.assertEqual([int(row["window_tokens"]) for row in windows], [48] * 4)
                    ready = json.loads((folder / f"learner_{learner_id}/ready.json").read_text())
                    self.assertEqual(ready["initial_parameters_sha256"], initial_hash)
                    self.assertEqual(ready["revision"], 0)
                loaded = load_run(folder)
                self.assertEqual(loaded["config"].method, method)
                evaluation_model = _TinyTokenModel(2)
                load_global_parameters(evaluation_model, payload, num_fragments=1, expected_method=method, expected_revisions=[8])
                self.assertEqual(parameter_fingerprint(dict(evaluation_model.named_parameters())), summary["global_parameters_sha256"])
                with self.assertRaisesRegex(ValueError, "method differs"):
                    load_global_parameters(evaluation_model, payload, num_fragments=1, expected_method="decoupled_heloco")

    def test_startup_failure_stops_siblings_and_never_exports_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config = configuration("diloco")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = _run_baseline(config, options_for(config), _TinyTokenModel(2), ["0", "1"], folder, 10.0, ROOT,
                                       worker_command=[sys.executable, "-c", "raise SystemExit(7)"])
            self.assertEqual(status, 2)
            self.assertEqual(json.loads((folder / "summary.json").read_text())["status"], "failed")
            self.assertFalse((folder / "global_model.pt").exists())


if __name__ == "__main__":
    unittest.main()
