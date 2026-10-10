"""Real CPU metrics and export checks; no claim of CUDA/TorchTitan execution."""

from contextlib import redirect_stdout, redirect_stderr
import csv
from dataclasses import asdict
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

import evaluate_decoupled_heloco as launcher
import run_decoupled_heloco as training_launcher
from panoengine.decentralized.decoupled_heloco.baseline_options import BASELINE_FORMAT, BASELINE_SCOPE, BASELINE_STOPPING
from panoengine.decentralized.decoupled_heloco.config import ConfigError, load_config
from panoengine.decentralized.decoupled_heloco.evaluation import (
    batch_fingerprint, evaluate_model, freeze_batches, load_global_parameters,
    load_validation_cache, parameter_fingerprint, validation_cache,
)
from panoengine.decentralized.decoupled_heloco.fragment_manager import FragmentManager
from panoengine.decentralized.decoupled_heloco.global_validation import (
    check_compatible_runs, load_run, run_validation, tokenizer_fingerprint,
)
from panoengine.decentralized.decoupled_heloco.gpu_recipe import build_initial_model

ROOT = Path(__file__).resolve().parents[2]


def batch(tokens, labels):
    inputs = torch.tensor([tokens], dtype=torch.long)
    return {"input": inputs, "positions": torch.arange(len(tokens)).unsqueeze(0)}, torch.tensor([labels], dtype=torch.long)


class TokenTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Parameter(torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]))
        self.bias = torch.nn.Parameter(torch.zeros(3))
        self.dropout = torch.nn.Dropout(0.5)
        self.seen = []

    def get_attention_masks(self, *, positions):
        return positions.eq(0)

    def forward(self, tokens, positions=None, attention_masks=None):
        self.seen.append((self.training, torch.is_grad_enabled(), positions.clone(), attention_masks.clone()))
        return self.dropout(self.table[tokens] + self.bias)


def checkpoint(model, count=2):
    manager = FragmentManager.from_model(model, count)
    return {"format": "decoupled_heloco_global_v1", "parameters": {name: p.detach().clone() for name, p in model.named_parameters()},
            "fragment_revisions": [1] * count, "layout_signature": manager.layout_signature, "resumable": False}


class EvaluationTests(unittest.TestCase):
    def test_token_weighted_loss_ignores_padding_without_shifting_targets(self):
        model = TokenTable()
        model.dropout.eval()  # Restore this mixed mode after evaluation.
        before = parameter_fingerprint(dict(model.named_parameters()))
        batches = [batch([0, 1, 2], [0, 1, 2]), batch([0, 1, 2], [1, -100, -100])]
        result = evaluate_model(model, batches, device="cpu", vocab_size=3)
        good = math.log(math.exp(4) + 2) - 4
        bad = good + 4
        self.assertAlmostEqual(result["loss"], (3 * good + bad) / 4, places=6)
        self.assertAlmostEqual(result["perplexity"], math.exp(result["loss"]), places=12)
        self.assertEqual(result["next_token_accuracy"], 0.75)
        self.assertEqual(result["valid_tokens"], 4)
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)
        self.assertTrue(all(not training and not grad for training, grad, _, _ in model.seen))
        self.assertTrue(torch.equal(model.seen[0][2], batches[0][0]["positions"]))
        self.assertTrue(torch.equal(model.seen[0][3], torch.tensor([[True, False, False]])))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(parameter_fingerprint(dict(model.named_parameters())), before)

    def test_invalid_labels_and_nonfinite_output_fail_and_restore_modes(self):
        model = TokenTable()
        for labels in ([0, 3], [-100, -100], [0, -1]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                evaluate_model(model, [batch([0, 1], labels)], device="cpu", vocab_size=3)
            self.assertTrue(model.training)
        with torch.no_grad():
            model.bias[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite logits"):
            evaluate_model(model, [batch([0, 1], [0, 1])], device="cpu", vocab_size=3)
        self.assertTrue(model.training)

    def test_perplexity_overflow_is_explicit_and_json_safe(self):
        model = TokenTable()
        with torch.no_grad():
            model.table[0] = torch.tensor([0.0, -1000.0, -1000.0])
        result = evaluate_model(model, [batch([0], [1])], device="cpu", vocab_size=3)
        self.assertEqual(result["loss"], 1000.0)
        self.assertIsNone(result["perplexity"])
        self.assertTrue(result["perplexity_overflow"])
        json.dumps(result, allow_nan=False)

    def test_frozen_batches_copy_reused_storage_and_consume_exact_count(self):
        original = batch([0, 1], [1, 2])
        visits = []

        def source():
            for index in range(3):
                visits.append(index)
                original[0]["input"].fill_(index)
                yield original

        frozen = freeze_batches(source(), 2, seq_len=2, batch_size=1, vocab_size=3)
        self.assertEqual(visits, [0, 1])
        self.assertTrue(torch.equal(frozen[0][0]["input"], torch.zeros((1, 2), dtype=torch.long)))
        original[0]["input"].fill_(2)
        self.assertTrue(torch.equal(frozen[1][0]["input"], torch.ones((1, 2), dtype=torch.long)))
        with self.assertRaisesRegex(ValueError, "data ended"):
            freeze_batches([], 1, seq_len=2, batch_size=1, vocab_size=3)

    def test_cache_checks_split_tokenizer_shapes_and_actual_tensor_bytes(self):
        batches = [batch([0, 1], [1, 2])]
        metadata = {"dataset": "c4_validation", "split": "validation", "seq_len": 2,
                    "batch_size": 1, "batch_count": 1, "tokenizer_sha256": "tokenizer-a"}
        cached = validation_cache(batches, metadata)
        self.assertIs(load_validation_cache(cached, metadata, vocab_size=3), batches)
        for changed in ({"split": "train"}, {"tokenizer_sha256": "b"}, {"seq_len": 1}, {"batch_count": 2}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "cache"):
                load_validation_cache(cached, {**metadata, **changed}, vocab_size=3)
        original_hash = batch_fingerprint(batches)
        batches[0][0]["positions"][0, 1] = 0
        self.assertNotEqual(batch_fingerprint(batches), original_hash)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            load_validation_cache(cached, metadata, vocab_size=3)

    def test_global_export_validates_every_parameter_before_replacing_any(self):
        for failure in ("nan", "dtype", "shape", "missing", "layout", "revision"):
            model = TokenTable()
            payload = checkpoint(model)
            payload["parameters"]["table"].fill_(8)
            if failure == "nan":
                payload["parameters"]["bias"][0] = float("nan")
            elif failure == "dtype":
                payload["parameters"]["bias"] = payload["parameters"]["bias"].double()
            elif failure == "shape":
                payload["parameters"]["bias"] = torch.zeros(2)
            elif failure == "missing":
                del payload["parameters"]["bias"]
            elif failure == "layout":
                payload["layout_signature"] = "wrong"
            else:
                payload["fragment_revisions"] = [2, 1]
            before = parameter_fingerprint(dict(model.named_parameters()))
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                load_global_parameters(model, payload, num_fragments=2, expected_revisions=[1, 1])
            self.assertEqual(parameter_fingerprint(dict(model.named_parameters())), before)

    def test_loaded_global_values_do_not_alias_checkpoint_storage(self):
        model = TokenTable()
        payload = checkpoint(model)
        payload["parameters"]["bias"].fill_(2)
        load_global_parameters(model, payload, num_fragments=2)
        payload["parameters"]["bias"].zero_()
        self.assertTrue(torch.equal(model.bias, torch.full((3,), 2.0)))
        with self.assertRaisesRegex(ValueError, "global export"):
            load_global_parameters(model, {"model": model.state_dict()}, num_fragments=2)

    def test_initialization_reproduces_cpu_seed_and_restores_rng_and_dtype(self):
        class InitialModel(torch.nn.Linear):
            def init_weights(self, *, buffer_device):
                torch.nn.init.normal_(self.weight)
                torch.nn.init.normal_(self.bias)

        calls = []
        cfg = SimpleNamespace(model_spec=SimpleNamespace(model=SimpleNamespace(
            update_from_config=lambda **kwargs: calls.append(kwargs), build=lambda: InitialModel(2, 3))))
        before = torch.random.get_rng_state().clone()
        dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            a, b = build_initial_model(cfg, 42), build_initial_model(cfg, 42)
            self.assertEqual(torch.get_default_dtype(), torch.float64)
            self.assertEqual(a.weight.dtype, torch.float32)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), before))
            self.assertEqual(parameter_fingerprint(dict(a.named_parameters())), parameter_fingerprint(dict(b.named_parameters())))
            self.assertEqual(calls, [{"config": cfg}, {"config": cfg}])
        finally:
            torch.set_default_dtype(dtype)

    def test_shared_run_comparison_rejects_different_seed_or_shapes(self):
        a = {"folder": Path("a"), "options": SimpleNamespace(module="m", config="c", seed=42, seq_len=512, batch=1)}
        for key, value in (("seed", 43), ("seq_len", 256), ("batch", 2)):
            b = {"folder": Path("b"), "options": SimpleNamespace(**{**vars(a["options"]), key: value})}
            with self.subTest(key=key), self.assertRaisesRegex(ConfigError, key):
                check_compatible_runs([a, b])

    def test_tokenizer_fingerprint_tracks_configuration_and_special_ids(self):
        tokenizer = SimpleNamespace(get_vocab=lambda: {"a": 0}, bos_id=0, eos_id=0)
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory)
            (assets / "tokenizer.json").write_text("version A")
            first = tokenizer_fingerprint(tokenizer, assets)
            (assets / "tokenizer.json").write_text("version B")
            second = tokenizer_fingerprint(tokenizer, assets)
            self.assertNotEqual(first, second)
            tokenizer.eos_id = 1
            self.assertNotEqual(second, tokenizer_fingerprint(tokenizer, assets))


class ValidationCoordinatorTests(unittest.TestCase):
    def test_completed_runs_share_one_frozen_stream_and_write_metrics_and_tail(self):
        # Exercise coordinator contracts with real CPU forward/metric logic;
        # substitute only unavailable CUDA allocation and TorchTitan builders.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "fixture-tokenizer"
            assets.mkdir()
            (assets / "tokenizer.json").write_text('{"fixture": "shared tokenizer"}')
            # Populate a separate repository's default tokenizer directory.
            # The fixture must pass even when real assets are installed there.
            repo = root / "repo"
            unrelated = repo / "assets/tokenizer/debug"
            unrelated.mkdir(parents=True)
            (unrelated / "tokenizer.json").write_text('{"fixture": "unrelated tokenizer"}')
            folders = [root / name for name in ("a", "b", "c", "d", "e")]
            methods = ["decoupled_heloco", "decoupled_diloco", "heloco", "diloco", "mla"]
            config_data = yaml.safe_load((ROOT / "decoupled_heloco.yaml").read_text())
            config_data["decoupled"]["num_fragments"] = 2
            config_data["run"].update(seq_len=2, batch=1, hf_assets=str(assets))
            options = {**config_data["run"], "gpus": "0,1,2,3"}
            for index, folder in enumerate(folders):
                folder.mkdir()
                run_config = json.loads(json.dumps(config_data))
                run_config["method"] = methods[index]
                (folder / "experiment.yaml").write_text(yaml.safe_dump(run_config))
                parsed_config = load_config(folder / "experiment.yaml")
                effective = folder / "legacy.yaml"
                effective.write_text(yaml.safe_dump(parsed_config.legacy_options()))
                resolved = training_launcher._parse_legacy_options(training_launcher._load_legacy_launcher(), effective)
                (folder / "run-options.json").write_text(json.dumps(vars(resolved)))
                metadata = {"status": "passed", "fragment_revisions": [1, 1], "initial_parameters_sha256": parameter_fingerprint(dict(TokenTable().named_parameters())),
                            "learners": [{"learner_id": 2, "fragments": [
                                {"fragment_id": 0, "local_steps": 184, "tokens": 94208},
                                {"fragment_id": 1, "local_steps": 238, "tokens": 121856}]}]}
                if index >= 2:
                    metadata.update(method=methods[index], scope=BASELINE_SCOPE, stopping_policy=BASELINE_STOPPING,
                                    fragment_revisions=[1], learners=[{"learner_id": 2, "fragments": [
                                        {"fragment_id": 0, "local_steps": 0, "tokens": 0}]}])
                (folder / "summary.json").write_text(json.dumps(metadata))
                trained = TokenTable()
                with torch.no_grad():
                    trained.table.zero_()
                payload = checkpoint(trained, count=1 if index >= 2 else 2)
                if index == 1:
                    payload["method"] = "decoupled_diloco"
                if index >= 2:
                    payload.update(format=BASELINE_FORMAT, method=methods[index])
                torch.save(payload, folder / "global_model.pt")
            tokenizer = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1, "c": 2}, bos_id=0, eos_id=2,
                                        encode=lambda *args, **kwargs: [0, 1, 2])
            recipe = SimpleNamespace(tokenizer=SimpleNamespace(build=lambda **kwargs: tokenizer), hf_assets_path=str(assets),
                                     model_spec=SimpleNamespace(model=SimpleNamespace(vocab_size=3)))
            calls = []
            loader_cfg = SimpleNamespace(build=lambda **kwargs: calls.append(kwargs) or [batch([0, 1], [0, 1]), batch([1, 2], [1, 2])])
            modules = {}
            for name in ("torchtitan", "torchtitan.distributed", "torchtitan.distributed.utils", "torchtitan.hf_datasets", "torchtitan.hf_datasets.text_datasets"):
                modules[name] = ModuleType(name)
            modules["torchtitan.distributed.utils"].set_spmd_backend = lambda _: None
            modules["torchtitan.hf_datasets.text_datasets"].HuggingFaceTextDataLoader = SimpleNamespace(Config=lambda **kwargs: loader_cfg)

            class CpuModel(TokenTable):
                def to(self, *args, **kwargs):
                    return self  # A CPU fixture, never a CUDA capability test.

            def cpu_metrics(model, batches, **kwargs):
                return evaluate_model(model, batches, device="cpu", vocab_size=kwargs["vocab_size"], progress=kwargs["progress"])

            with patch.dict(sys.modules, modules), patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1), patch("torch.cuda.set_device"), patch("panoengine.decentralized.decoupled_heloco.global_validation.build_recipe", return_value=recipe), patch("panoengine.decentralized.decoupled_heloco.global_validation.build_initial_model", side_effect=lambda *args: CpuModel()), patch("panoengine.decentralized.decoupled_heloco.global_validation.evaluate_model", side_effect=cpu_metrics), redirect_stdout(io.StringIO()):
                self.assertEqual(run_validation(folders, repo, batches=2, output_dir=root / "evaluation-a"), 0)
                self.assertEqual(run_validation(folders, repo, batches=2, output_dir=root / "evaluation-b", validation_cache_path=root / "evaluation-a/validation_batches.pt"), 0)
                # Preserve the real guard: equal mocked vocabularies with
                # different tokenizer file contents must still be rejected.
                different = json.loads(json.dumps(config_data))
                different["run"]["hf_assets"] = str(unrelated)
                (folders[1] / "experiment.yaml").write_text(yaml.safe_dump(different))
                (folders[1] / "run-options.json").write_text(json.dumps({**options, "hf_assets": str(unrelated)}))
                with self.assertRaisesRegex(ConfigError, "runs use different tokenizer assets"):
                    run_validation(folders, repo, batches=2, output_dir=root / "reject-mismatched-tokenizers")
            self.assertEqual(len(calls), 1)
            result = json.loads((root / "evaluation-a/evaluation.json").read_text())
            reused = json.loads((root / "evaluation-b/evaluation.json").read_text())
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["validation"]["batches_sha256"], reused["validation"]["batches_sha256"])
            self.assertEqual([r["label"] for r in result["models"]], ["initial", "a", "b", "c", "d", "e"])
            self.assertEqual([r["method"] for r in result["models"][1:]], methods)
            self.assertTrue(all(r["valid_tokens"] == 4 for r in result["models"]))
            self.assertGreater(result["models"][1]["loss"], result["models"][0]["loss"])
            self.assertEqual(result["models"][1]["unmerged_work"][0]["fragments"][1]["local_steps"], 238)
            self.assertTrue(result["models"][1]["initial_fingerprint_verified"])
            with (root / "evaluation-a/evaluation.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 6)
            self.assertTrue(all(r["unmerged_work"][0]["fragments"][0]["tokens"] == 0 for r in result["models"][3:]))

    def test_cli_routes_to_evaluator_and_rejects_invalid_count_before_work(self):
        with patch("panoengine.decentralized.decoupled_heloco.global_validation.run_validation", return_value=0) as evaluate:
            self.assertEqual(launcher.main(["--run-dirs", "a", "b", "--batches", "20"]), 0)
            self.assertEqual(evaluate.call_args.args[0], [Path("a"), Path("b")])
            self.assertEqual(evaluate.call_args.kwargs["device"], "cuda:0")
            with redirect_stderr(io.StringIO()):
                self.assertEqual(launcher.main(["--run-dirs", "a", "--batches", "0"]), 2)
            self.assertEqual(evaluate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
