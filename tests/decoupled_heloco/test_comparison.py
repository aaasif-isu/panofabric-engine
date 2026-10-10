"""One-YAML orchestration with real CPU trainers/transports and frozen metrics."""

from contextlib import redirect_stdout, redirect_stderr
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

import run_method_comparison as comparison
from panoengine.decentralized.decoupled_heloco.baseline_training import _run_baseline, run_baseline_training
from panoengine.decentralized.decoupled_heloco.config import ConfigError, DECOUPLED_METHODS, load_config
from panoengine.decentralized.decoupled_heloco.evaluation import evaluate_model
from panoengine.decentralized.decoupled_heloco.global_validation import run_validation
from panoengine.decentralized.decoupled_heloco.gpu_training import _run_training, run_gpu_training
from panoengine.decentralized.decoupled_heloco.smoke import _TinyTokenModel
from test_baselines import HTTP_AVAILABLE

ROOT = Path(__file__).resolve().parents[2]


class ComparisonConfigTests(unittest.TestCase):
    def test_default_five_methods_validate_and_preview_without_torch_or_launch(self):
        code = "import sys; import run_method_comparison as c; status=c.main(['--check-config']); assert 'torch' not in sys.modules and 'torchft' not in sys.modules; raise SystemExit(status)"
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        loaded = comparison.load_comparison(ROOT / "run_script" / "method_comparison.yaml")
        self.assertEqual(loaded["methods"], ["decoupled_heloco", "decoupled_diloco", "heloco", "diloco", "mla"])
        self.assertEqual([r.steps for r in loaded["options"]], [40] * 5)
        self.assertEqual([r.outer_lr for r in loaded["options"]], [0.7] * 5)
        with patch.object(comparison.subprocess, "Popen") as launch, redirect_stdout(io.StringIO()):
            self.assertEqual(comparison.main(["--dry-run"]), 0)
            launch.assert_not_called()

    def test_bad_method_lists_and_evaluation_options_fail_before_any_output(self):
        for changes in ({"method_run": []}, {"method_run": "heloco"}, {"method_run": ["heloco", "heloco"]},
                        {"method_run": ["decoupled heloco"]}, {"evaluation": {"batches": True}},
                        {"evaluation": {"timeout": -1}}, {"evaluation": {"enabled": "yes"}},
                        {"evaluation": {"validation_cache": []}}, {"extra_section": True}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                data = yaml.safe_load((ROOT / "run_script" / "method_comparison.yaml").read_text())
                data.update(changes)
                path = Path(directory) / "invalid.yaml"
                path.write_text(yaml.safe_dump(data))
                with self.assertRaises(ConfigError):
                    comparison.load_comparison(path)

    def test_later_baseline_invalid_budget_is_rejected_before_training_any_method(self):
        with tempfile.TemporaryDirectory() as directory:
            data = yaml.safe_load((ROOT / "run_script" / "method_comparison.yaml").read_text())
            data["run"]["steps"] = 11  # Decoupled budgets valid; baseline H=10 invalid.
            path = Path(directory) / "invalid.yaml"
            path.write_text(yaml.safe_dump(data))
            output = Path(directory) / "outputs"
            with patch.object(comparison.subprocess, "Popen") as launch, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(comparison.main(["--config-file", str(path), "--output-dir", str(output)]), 2)
                launch.assert_not_called()
            self.assertFalse(output.exists())

    def test_explicit_run_directory_routes_and_refuses_to_overwrite_for_both_coordinators(self):
        loaded = comparison.load_comparison(ROOT / "run_script" / "method_comparison.yaml")
        for index, function, module, preflight, coordinator in (
            (0, run_gpu_training, "gpu_training", "preflight", "_run_training"),
            (2, run_baseline_training, "baseline_training", "preflight_baseline", "_run_baseline"),
        ):
            with self.subTest(module=module), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "method"
                package = "panoengine.decentralized.decoupled_heloco." + module
                with patch(package + "." + preflight, return_value=(_TinyTokenModel(2), ["0", "1"])), patch(package + "." + coordinator, return_value=0) as train:
                    self.assertEqual(function(loaded["configs"][index], loaded["options"][index], ROOT, output_dir=output), 0)
                    self.assertEqual(train.call_args.args[4], output)
                    self.assertTrue(output.is_dir())
                    with self.assertRaises(FileExistsError):
                        function(loaded["configs"][index], loaded["options"][index], ROOT, output_dir=output)
                    self.assertEqual(train.call_count, 1)


class ComparisonSupervisionTests(unittest.TestCase):
    def test_output_is_streamed_and_child_failures_and_timeouts_surface(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            log = Path(directory) / "child.log"
            comparison.run_command([sys.executable, "-c", "print('child output')"], log, ROOT, 5.0)
            self.assertIn("child output", log.read_text())
            with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                comparison.run_command([sys.executable, "-c", "raise SystemExit(7)"], log, ROOT, 5.0)
            with self.assertRaisesRegex(TimeoutError, "deadline"):
                comparison.run_command([sys.executable, "-c", "import time; time.sleep(30)"], log, ROOT, 0.2)

    def test_failed_method_preserves_manifest_and_never_runs_later_methods_or_evaluation(self):
        loaded = comparison.load_comparison(ROOT / "run_script" / "method_comparison.yaml")
        calls = []

        def failure(command, *_):
            calls.append(command)
            raise RuntimeError("deliberate child failure")

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            output = Path(directory) / "comparison"
            self.assertEqual(comparison.run_comparison(loaded, ROOT, output_dir=output, command_runner=failure), 2)
            self.assertEqual(len(calls), 1)
            manifest = json.loads((output / "comparison.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["current_method"], "decoupled_heloco")
            self.assertIn("deliberate child failure", manifest["error"])
            self.assertFalse((output / "comparison.csv").exists())

    def test_missing_requested_validation_cache_fails_before_launch_or_output(self):
        loaded = comparison.load_comparison(ROOT / "run_script" / "method_comparison.yaml")
        with tempfile.TemporaryDirectory() as directory:
            loaded["evaluation"]["validation_cache"] = str(Path(directory) / "absent.pt")
            output = Path(directory) / "comparison"
            with self.assertRaisesRegex(ConfigError, "cache does not exist"):
                comparison.run_comparison(loaded, ROOT, output_dir=output)
            self.assertFalse(output.exists())


@unittest.skipUnless(HTTP_AVAILABLE, "requires the project's TorchFT HTTP helpers")
class FullCPUComparisonTests(unittest.TestCase):
    def test_one_yaml_runs_all_five_real_cpu_coordinators_then_one_shared_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "tokenizer"
            assets.mkdir()
            (assets / "tokenizer.json").write_text('{"fixture":"cpu-only"}')
            data = yaml.safe_load((ROOT / "run_script" / "method_comparison.yaml").read_text())
            data["run"].update(islands=2, gpus=[0, 1], steps=12, sync_steps=2,
                               seq_len=8, batch=2, hf_assets=str(assets), ps_timeout=30.0,
                               island_slowness_factors=[1, 2])
            data["decoupled"].update(num_fragments=2, min_quorum=2, overlap_steps=2,
                                     sync_interval=0.1, grace_window_factor=0.2)
            data["evaluation"]["batches"] = 2
            path = root / "comparison.yaml"
            path.write_text(yaml.safe_dump(data))
            loaded = comparison.load_comparison(path)
            calls = []
            batches_loaded = []

            class CpuModel(_TinyTokenModel):
                def to(self, *args, **kwargs):
                    return self  # Explicit CPU fixture, never a CUDA test.

                def forward(self, inputs, **_kwargs):
                    return super().forward(inputs)

            def initial_model(*_args):
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(42)
                    return CpuModel(2)

            tokenizer = SimpleNamespace(get_vocab=lambda: {str(i): i for i in range(8)}, bos_id=0, eos_id=7,
                                        encode=lambda *args, **kwargs: [0, 1, 7])
            recipe = SimpleNamespace(tokenizer=SimpleNamespace(build=lambda **kwargs: tokenizer), hf_assets_path=str(assets),
                                     model_spec=SimpleNamespace(model=SimpleNamespace(vocab_size=8)))
            tokens = torch.arange(8).repeat(2, 1)
            batches = [({"input": tokens, "positions": torch.arange(8).repeat(2, 1)}, (tokens + 1) % 8)] * 2
            loader_cfg = SimpleNamespace(build=lambda **kwargs: batches_loaded.append(kwargs) or batches)
            modules = {name: ModuleType(name) for name in ("torchtitan", "torchtitan.distributed", "torchtitan.distributed.utils",
                                                           "torchtitan.hf_datasets", "torchtitan.hf_datasets.text_datasets")}
            modules["torchtitan.distributed.utils"].set_spmd_backend = lambda _: None
            modules["torchtitan.hf_datasets.text_datasets"].HuggingFaceTextDataLoader = SimpleNamespace(Config=lambda **kwargs: loader_cfg)

            def cpu_metrics(model, frozen, **kwargs):
                return evaluate_model(model, frozen, device="cpu", vocab_size=8)

            def runner(command, log, repo_root, timeout):
                calls.append(Path(command[1]).name)
                log.write_text("CPU fixture invokes actual trainers/transports and metrics\n")
                if calls[-1] == "evaluate_decoupled_heloco.py":
                    dirs = command[command.index("--run-dirs") + 1:command.index("--batches")]
                    output = Path(command[command.index("--output-dir") + 1])
                    with patch.dict(sys.modules, modules), patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1), patch("torch.cuda.set_device"), patch("panoengine.decentralized.decoupled_heloco.global_validation.build_recipe", return_value=recipe), patch("panoengine.decentralized.decoupled_heloco.global_validation.build_initial_model", side_effect=initial_model), patch("panoengine.decentralized.decoupled_heloco.global_validation.evaluate_model", side_effect=cpu_metrics):
                        self.assertEqual(run_validation(dirs, ROOT, batches=2, output_dir=output), 0)
                    return
                config = load_config(Path(command[command.index("--config-file") + 1]))
                index = loaded["methods"].index(config.method)
                options = loaded["options"][index]
                folder = Path(command[command.index("--output-dir") + 1])
                folder.mkdir()
                model = initial_model()
                if config.method in DECOUPLED_METHODS:
                    worker = Path(__file__).with_name("cpu_training_worker.py")
                    status = _run_training(config, options, model, ["0", "1"], folder, 30.0, ROOT,
                                           worker_command=[sys.executable, str(worker)])
                else:
                    worker = Path(__file__).with_name("cpu_baseline_worker.py")
                    status = _run_baseline(config, options, model, ["0", "1"], folder, 30.0, ROOT,
                                           worker_command=[sys.executable, str(worker)])
                if status:
                    self.fail("\n".join(p.read_text() for p in folder.glob("learner_*/trainer.log")))

            output = root / "results"
            text = io.StringIO()
            with redirect_stdout(text), redirect_stderr(text):
                status = comparison.run_comparison(loaded, ROOT, output_dir=output, command_runner=runner)
            self.assertEqual(status, 0, text.getvalue())
            self.assertEqual(calls, ["run_decoupled_heloco.py"] * 2 + ["run_matched_baselines.py"] * 3 + ["evaluate_decoupled_heloco.py"])
            self.assertEqual(len(batches_loaded), 1)
            manifest = json.loads((output / "comparison.json").read_text())
            self.assertEqual(manifest["status"], "passed")
            self.assertEqual([r["method"] for r in manifest["results"]], ["initial", *loaded["methods"]])
            self.assertEqual([r["processed_tokens"] for r in manifest["results"][1:]], [384] * 5)
            self.assertTrue(all(r["initial_fingerprint_verified"] for r in manifest["evaluation"]["models"][1:]))
            with (output / "comparison.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertEqual({int(r["valid_tokens"]) for r in rows}, {32})
            self.assertTrue((output / "evaluation/validation_batches.pt").is_file())


if __name__ == "__main__":
    unittest.main()
