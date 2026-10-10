"""Routing checks requiring no GPUs, torchft, or torchtitan."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

import run_decoupled_heloco as launcher


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config_path = Path(self.directory.name) / "experiment.yaml"
        self.config = yaml.safe_load((launcher.REPO_ROOT / "decoupled_heloco.yaml").read_text())

    def write_config(self):
        self.config_path.write_text(yaml.safe_dump(self.config))

    def invoke(self, *arguments):
        self.write_config()
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            status = launcher.main(["--config-file", str(self.config_path), *arguments])
        return status, output.getvalue()

    def test_decoupled_validation_and_preview_never_launch(self):
        with patch.object(launcher.subprocess, "call") as launch:
            for action in ("--check-config", "--dry-run"):
                with self.subTest(action=action):
                    status, _ = self.invoke(action)
                    self.assertEqual(status, 0)
            launch.assert_not_called()

    def test_decoupled_execution_routes_to_gpu_coordinator(self):
        with patch("panoengine.decentralized.decoupled_heloco.gpu_training.run_gpu_training", return_value=0) as train:
            with patch.object(launcher.subprocess, "call") as launch:
                status, _ = self.invoke()
        self.assertEqual(status, 0)
        self.assertEqual(train.call_args.kwargs, {"timeout": 1800.0, "check_only": False})
        launch.assert_not_called()

    def test_gpu_preflight_routes_without_running_legacy_launcher(self):
        with patch("panoengine.decentralized.decoupled_heloco.gpu_training.run_gpu_training", return_value=0) as train:
            status, _ = self.invoke("--check-training", "--training-timeout", "120")
        self.assertEqual(status, 0)
        self.assertEqual(train.call_args.kwargs, {"timeout": 120.0, "check_only": True})
        status, _ = self.invoke("--check-training", "--method", "heloco")
        self.assertEqual(status, 2)

    def test_smoke_flag_routes_to_local_demo_without_launching_baselines(self):
        with patch("panoengine.decentralized.decoupled_heloco.smoke.run_smoke", return_value=0) as smoke:
            with patch.object(launcher.subprocess, "call") as launch:
                status, _ = self.invoke("--smoke-test", "--smoke-ticks", "12")
        self.assertEqual(status, 0)
        self.assertEqual(smoke.call_args.kwargs, {"ticks": 12, "outer_method": "heloco"})
        launch.assert_not_called()

    def test_smoke_rejects_legacy_method_and_invalid_tick_count(self):
        with patch.object(launcher.subprocess, "call") as launch:
            for arguments in (("--smoke-test", "--method", "heloco"), ("--smoke-test", "--smoke-ticks", "0")):
                status, _ = self.invoke(*arguments)
                self.assertEqual(status, 2)
        launch.assert_not_called()

    def test_process_smoke_routes_to_local_process_demo(self):
        with patch("panoengine.decentralized.decoupled_heloco.process_smoke.run_process_smoke", return_value=0) as smoke:
            with patch.object(launcher.subprocess, "call") as launch:
                status, _ = self.invoke("--process-smoke-test", "--process-learners", "3", "--process-cycles", "1", "--process-timeout", "20")
        self.assertEqual(status, 0)
        self.assertEqual(smoke.call_args.kwargs, {"learners": 3, "cycles": 1, "timeout": 20.0})
        launch.assert_not_called()

    def test_process_smoke_rejects_wrong_method_and_impossible_quorum(self):
        with patch.object(launcher.subprocess, "call") as launch:
            for arguments in (("--process-smoke-test", "--method", "heloco"), ("--process-smoke-test", "--process-learners", "1"), ("--process-smoke-test", "--process-cycles", "0")):
                status, _ = self.invoke(*arguments)
                self.assertEqual(status, 2)
        launch.assert_not_called()

    def test_each_baseline_delegates_exactly_once_and_preserves_settings(self):
        self.config["run"]["steps"] = 17
        original_heloco_yaml = (launcher.REPO_ROOT / "heloco.yaml").read_bytes()
        for method in ("heloco", "diloco", "mla"):
            with self.subTest(method=method):
                config_paths = []

                def fake_launch(command, cwd):
                    self.assertEqual(command[:2], [launcher.sys.executable, str(launcher.REPO_ROOT / "run_heloco.py")])
                    self.assertEqual(cwd, launcher.REPO_ROOT)
                    path = Path(command[3])
                    config_paths.append(path)
                    effective = yaml.safe_load(path.read_text())
                    self.assertEqual(effective["methods"], [method])
                    self.assertEqual(effective["outer_method"], method)
                    self.assertEqual(effective["steps"], 17)
                    self.assertEqual(effective["num_fragments"], 1)
                    self.assertNotIn("decoupled", effective)
                    return 7

                with patch.object(launcher.subprocess, "call", side_effect=fake_launch) as launch:
                    status, _ = self.invoke("--method", method)
                self.assertEqual(status, 7)
                launch.assert_called_once()
                self.assertFalse(config_paths[0].exists())
        self.assertEqual((launcher.REPO_ROOT / "heloco.yaml").read_bytes(), original_heloco_yaml)

    def test_bad_configurations_fail_before_launch(self):
        cases = [
            ("method typo", lambda c: c.update(method="decoupeld_heloco")),
            ("unknown run key", lambda c: c["run"].update(islans=4)),
            ("competing method selector", lambda c: c["run"].update(methods=["diloco"])),
            ("quorum too large", lambda c: c["decoupled"].update(min_quorum=5)),
            ("fractional fragments", lambda c: c["decoupled"].update(num_fragments=1.5)),
            ("zero overlap", lambda c: c["decoupled"].update(overlap_steps=0)),
            ("nan poll interval", lambda c: c["decoupled"].update(sync_interval=float("nan"))),
            ("legacy fragments in new mode", lambda c: c["run"].update(num_fragments=4)),
            ("sync coordination in new mode", lambda c: c["run"].update(coordination_method="sync")),
            ("incorrect boolean", lambda c: c["run"].update(should_quantize="false")),
            ("repeated GPU", lambda c: c["run"].update(gpus=[0, 0, 2, 3])),
            ("null extra flags", lambda c: c["run"].update(extra=None)),
            ("legacy arrival scaling in new mode", lambda c: c["run"].update(rho=0.25)),
            ("per-worker correction in new mode", lambda c: c["run"].update(correction_workers="none")),
            ("whole-gradient correction in new mode", lambda c: c["run"].update(correction_scope="whole_gradient")),
            ("unknown correction option", lambda c: c["decoupled"]["heloco"].update(rho_typo=1.0)),
            ("negative correction scale", lambda c: c["decoupled"]["heloco"].update(rho=-0.1)),
            ("nan correction scale", lambda c: c["decoupled"]["heloco"].update(rho=float("nan"))),
            ("zero correction epsilon", lambda c: c["decoupled"]["heloco"].update(eps=0.0)),
            ("invalid cosine threshold", lambda c: c["decoupled"]["heloco"].update(c_ok=1.1)),
            ("invalid beta limit", lambda c: c["decoupled"]["heloco"].update(beta_max=1.1)),
            ("string correction boolean", lambda c: c["decoupled"]["heloco"].update(correction_enabled="false")),
            ("string lookahead boolean", lambda c: c["decoupled"]["heloco"].update(lookahead="true")),
        ]
        original = yaml.safe_dump(self.config)
        for label, mutate in cases:
            with self.subTest(case=label):
                self.config = yaml.safe_load(original)
                mutate(self.config)
                with patch.object(launcher.subprocess, "call") as launch:
                    status, _ = self.invoke("--check-config")
                self.assertEqual(status, 2)
                launch.assert_not_called()

    def test_legacy_constraints_and_decoupled_controls_are_independent(self):
        # P=4 does not divide legacy H=10, but it is valid for new-mode checks.
        status, _ = self.invoke("--check-config")
        self.assertEqual(status, 0)
        for method in ("heloco", "diloco", "mla"):
            status, _ = self.invoke("--method", method, "--check-config")
            self.assertEqual(status, 0)
        self.config["run"].update(sync_steps=10, num_fragments=4)
        status, _ = self.invoke("--method", "heloco", "--check-config")
        self.assertEqual(status, 2)

    def test_merged_correction_scale_and_ablation_flags_validate(self):
        self.config["decoupled"]["heloco"].update(rho=0.5, correction_enabled=False, lookahead=False)
        status, _ = self.invoke("--check-config")
        self.assertEqual(status, 0)

    def test_sgd_smoke_option_and_legacy_arrival_scale_remain_available(self):
        with patch("panoengine.decentralized.decoupled_heloco.smoke.run_smoke", return_value=0) as smoke:
            status, _ = self.invoke("--smoke-test", "--smoke-outer", "sgd")
        self.assertEqual(status, 0)
        self.assertEqual(smoke.call_args.kwargs["outer_method"], "sgd")
        self.config["run"].update(rho=0.25, correction_workers="none")
        status, _ = self.invoke("--method", "heloco", "--check-config")
        self.assertEqual(status, 0)


if __name__ == "__main__":
    unittest.main()
