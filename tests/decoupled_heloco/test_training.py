"""Real token accounting and process integration; CUDA/TorchTitan not emulated."""

from contextlib import redirect_stderr, redirect_stdout
import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from panoengine.decentralized.decoupled_heloco.config import ConfigError, DecoupledConfig, ExperimentConfig
from panoengine.decentralized.decoupled_heloco.fragment_manager import FragmentManager
from panoengine.decentralized.decoupled_heloco.gpu_recipe import validate_training_options
from panoengine.decentralized.decoupled_heloco.gpu_training import _run_training, select_devices, validate_frame_sizes
from panoengine.decentralized.decoupled_heloco.gpu_worker import initialize_fragments
from panoengine.decentralized.decoupled_heloco.learner import DecoupledLearner
from panoengine.decentralized.decoupled_heloco.process_smoke import _wait_message
from panoengine.decentralized.decoupled_heloco.smoke import _TinyTokenModel
from panoengine.decentralized.decoupled_heloco.training_adapter import IslandDataLoaderConfig, train_one_step
from panoengine.decentralized.decoupled_heloco.transport import FramedTransport, RemoteLearner


ROOT = Path(__file__).resolve().parents[2]


class TrainingAdapterTests(unittest.TestCase):
    def test_accumulated_valid_tokens_and_one_real_optimizer_step(self):
        model = torch.nn.Linear(2, 2, bias=False)
        learner = DecoupledLearner(model, 1, learner_id=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        before = model.weight.detach().clone()

        class Trainer:
            def train_step(self, iterator):
                optimizer.zero_grad()
                for _ in range(2):
                    inputs, labels = next(iterator)
                    loss = torch.nn.functional.cross_entropy(model(inputs["input"]), labels, ignore_index=-100)
                    loss.backward()
                optimizer.step()

        batches = iter([({"input": torch.ones(3, 2)}, torch.tensor([0, 1, -100])), ({"input": torch.ones(2, 2)}, torch.tensor([1, 1]))])
        self.assertEqual(train_one_step(Trainer(), batches, learner), 4)
        self.assertEqual(learner.metadata().total_local_steps, 1)
        self.assertEqual(learner.metadata().total_tokens, 4)
        self.assertFalse(torch.equal(before, model.weight))
        self.assertEqual(int(optimizer.state[model.weight]["step"]), 1)

    def test_failed_step_does_not_count_consumed_tokens_and_clears_boundary(self):
        learner = DecoupledLearner(torch.nn.Linear(2, 2), 1, learner_id=0)

        class Trainer:
            def train_step(self, iterator):
                next(iterator)
                raise RuntimeError("forward failed")

        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            train_one_step(Trainer(), iter([({}, torch.tensor([0, 1]))]), learner)
        self.assertEqual(learner.metadata().total_tokens, 0)
        self.assertEqual(learner.metadata().total_local_steps, 0)
        learner.boundary()

    def test_zero_valid_tokens_fail_before_optimizer_update(self):
        learner = DecoupledLearner(torch.nn.Linear(2, 2), 1, learner_id=0)
        trainer = SimpleNamespace(train_step=lambda iterator: next(iterator))
        with self.assertRaisesRegex(ValueError, "no valid target"):
            train_one_step(trainer, iter([({}, torch.tensor([-100, -100]))]), learner)
        self.assertEqual(learner.metadata().total_local_steps, 0)

    def test_data_shard_identity_uses_island_instead_of_local_rank_zero(self):
        calls = []
        source = SimpleNamespace(build=lambda **kwargs: calls.append(kwargs) or "loader", to_dict=lambda: {"dataset": "c4"})
        config = IslandDataLoaderConfig(source, 4, 2)
        self.assertEqual(config.build(dp_world_size=1, dp_rank=0, seq_len=512), "loader")
        self.assertEqual(calls, [{"dp_world_size": 4, "dp_rank": 2, "seq_len": 512}])
        self.assertEqual(config.to_dict()["learner_id"], 2)

    def test_gpu_mapping_preserves_slurm_allocation_and_uuid_masks(self):
        options = SimpleNamespace(islands=2, gpus="1,0")
        self.assertEqual(select_devices(options, "3,5", 2), ["5", "3"])
        self.assertEqual(select_devices(options, "GPU-one,GPU-two", 2), ["GPU-two", "GPU-one"])
        self.assertEqual(select_devices(options, None, 2), ["1", "0"])
        with self.assertRaises(ConfigError):
            select_devices(options, "3", 2)
        with self.assertRaises(ConfigError):
            select_devices(SimpleNamespace(islands=2, gpus="0,2"), None, 2)

    def test_training_scope_rejects_unsupported_modes_before_launch(self):
        config = ExperimentConfig("decoupled_heloco")
        options = dict(gpus_per_island=1, module="models.llama3_small", config="llama3_15m", data_distribution="iid", languages=[], dataset="c4", should_quantize=False, correction_heatmap=False, extra=[], host="127.0.0.1", islands=2, ps_timeout=30.0, island_slowness_factors=[1, 1])
        validate_training_options(config, SimpleNamespace(**options))
        for changes in ({"gpus_per_island": 2}, {"config": "llama3_1b"}, {"data_distribution": "non_iid"}, {"should_quantize": True}, {"host": "0.0.0.0"}, {"extra": ["--training.steps=99"]}, {"island_slowness_factors": [0.5, 1]}):
            with self.subTest(changes=changes), self.assertRaises(ConfigError):
                validate_training_options(config, SimpleNamespace(**{**options, **changes}))

    def test_frame_check_accounts_for_both_snapshot_copies(self):
        model = _TinyTokenModel(2)
        self.assertEqual(validate_frame_sizes(model, 2).num_fragments, 2)
        with patch("panoengine.decentralized.decoupled_heloco.gpu_training.MAX_FRAME_BYTES", 1024):
            with self.assertRaisesRegex(ConfigError, "snapshot exceeds"):
                validate_frame_sizes(model, 2)

    def test_fragment_initialization_transfers_exact_global_weights_before_ready(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        client = socket.create_connection(listener.getsockname())
        connection, _ = listener.accept()
        listener.close()
        sender, receiver = FramedTransport(connection), FramedTransport(client)
        self.addCleanup(sender.close)
        self.addCleanup(receiver.close)
        global_model, local_model = _TinyTokenModel(2), _TinyTokenModel(2)
        manager = FragmentManager.from_model(global_model, 2)
        failures = []

        def initialize():
            try:
                initialize_fragments(receiver, local_model, 2, time.monotonic() + 5)
            except Exception as exc:
                failures.append(exc)

        thread = threading.Thread(target=initialize)
        thread.start()
        try:
            for fragment in manager.fragments:
                sender.send("initialize_fragment", {"fragment_id": fragment.fragment_id, "layout_signature": manager.layout_signature, "parameters": manager.select(fragment.fragment_id, dict(global_model.named_parameters()))})
                self.assertEqual(_wait_message(sender, time.monotonic() + 5)["kind"], "initialize_ack")
        finally:
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(failures)
        for name, parameter in global_model.named_parameters():
            self.assertTrue(torch.equal(parameter, dict(local_model.named_parameters())[name]))

    def test_training_done_is_distinct_from_process_shutdown(self):
        learner = DecoupledLearner(_TinyTokenModel(2), 2, learner_id=0)
        peer = RemoteLearner(None, learner.metadata())
        peer.handle({"kind": "training_done", "body": {"metadata": asdict(learner.metadata())}})
        self.assertTrue(peer.training_done)
        self.assertFalse(peer.stopped)
        self.assertFalse(peer.paused)


class TrainingProcessTests(unittest.TestCase):
    def config(self):
        return ExperimentConfig("decoupled_heloco", decoupled=DecoupledConfig(num_fragments=2, min_quorum=2, overlap_steps=2, sync_interval=0.1, grace_window_factor=0.2))

    def options(self):
        return SimpleNamespace(islands=2, steps=12, outer_lr=0.7, outer_momentum=0.9, ps_timeout=30.0, island_slowness_factors=[1, 2], gpus_per_island=1, async_interval=1, max_wait_time=0.0, seq_len=512, batch=1, sync_steps=2, log_dir="logs", num_fragments=1, tokens_per_parameter=None, dataset="c4", rho=None, correction_workers="all", correction_scope="tensorwise", correction_heatmap=False, other_islands_method="heloco_uncorrected", diloco_lr=None, diloco_momentum=None, diloco_nesterov_period=None, coordination_method="async", data_distribution="iid", should_quantize=False, extra=[], module="models.llama3_small", config="llama3_15m", gpus=None, host="localhost")

    def test_baseline_time_budget(self):
        import copy
        from panoengine.decentralized.decoupled_heloco.baseline_training import _run_baseline
        config = self.config()
        config = ExperimentConfig(
            "heloco",
            run=config.run,
            decoupled=DecoupledConfig(stopping="time", time_budget=1.0, num_fragments=1, sync_interval=1.0),
            monitoring=config.monitoring,
        )
        options = self.options()
        options.steps = 100
        options.sync_steps = 2
        options.island_slowness_factors = [1, 2]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            start = time.monotonic()
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                status = _run_baseline(config, options, _TinyTokenModel(2), ["0", "1"], folder, 20.0, ROOT, worker_command=[sys.executable, str(Path(__file__).with_name("cpu_baseline_worker.py"))])
            duration = time.monotonic() - start
            
            self.assertEqual(status, 0, output.getvalue())
            
            summary = json.loads((folder / "summary.json").read_text())
            self.assertEqual(summary["status"], "passed")
            # workers should complete different numbers of steps and pushes
            learners = summary["learners"]
            self.assertEqual(len(learners), 2)
            self.assertTrue(any(l["total_local_steps"] > 0 for l in learners))
            
            # Learner 0 is faster, so should have more steps
            l0 = next(l for l in learners if l["learner_id"] == 0)
            l1 = next(l for l in learners if l["learner_id"] == 1)
            self.assertGreater(l0["total_local_steps"], l1["total_local_steps"])
            
            # total pushes matches summary expected revision
            self.assertEqual(summary["sync_updates"], l0["pushes"] + l1["pushes"])

    def test_fixed_budgets_final_drain_logs_and_global_export_over_real_tcp(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            output = io.StringIO()
            torch.manual_seed(42)
            model = _TinyTokenModel(2)
            initial = {name: p.detach().clone() for name, p in model.named_parameters()}
            with redirect_stdout(output), redirect_stderr(output):
                status = _run_training(self.config(), self.options(), model, ["0", "1"], folder, 30.0, ROOT, worker_command=[sys.executable, str(Path(__file__).with_name("cpu_training_worker.py"))])
            self.assertEqual(status, 0, output.getvalue())
            summary = json.loads((folder / "summary.json").read_text())
            self.assertEqual(summary["status"], "passed")
            self.assertTrue(all(r >= 1 for r in summary["fragment_revisions"]))
            self.assertEqual([l["total_local_steps"] for l in summary["learners"]], [12, 12])
            self.assertEqual([l["total_tokens"] for l in summary["learners"]], [192, 192])
            for index in range(2):
                with (folder / f"learner_{index}" / "steps.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 12)
                self.assertEqual([int(row["tokens"]) for row in rows], [16] * 12)
                self.assertTrue(all(math_isfinite(row["loss"]) for row in rows))
            with (folder / "syncs.csv").open() as stream:
                syncs = list(csv.DictReader(stream))
            self.assertGreaterEqual(len(syncs), 2)
            for row in syncs:
                tokens, weights = json.loads(row["tokens"]), json.loads(row["weights"])
                for tokens_i, weight in zip(tokens, weights):
                    self.assertAlmostEqual(weight, tokens_i / sum(tokens))
            checkpoint = torch.load(folder / "global_model.pt", weights_only=True)
            self.assertFalse(checkpoint["resumable"])
            self.assertEqual(tuple(summary["fragment_revisions"]), checkpoint["fragment_revisions"])
            self.assertTrue(any(not torch.equal(initial[name], value) for name, value in checkpoint["parameters"].items()))

    def test_startup_failure_returns_diagnostics_and_stops_siblings(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = _run_training(self.config(), self.options(), _TinyTokenModel(2), ["0", "1"], folder, 10.0, ROOT, worker_command=[sys.executable, "-c", "raise SystemExit(7)"])
            self.assertEqual(status, 2)
            summary = json.loads((folder / "summary.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertIn("startup", summary["error"])
            self.assertFalse((folder / "global_model.pt").exists())

    def test_slow_first_batch_does_not_inflate_grace_or_sleep(self):
        import copy
        config = self.config()
        new_decoupled = copy.copy(config.decoupled)
        object.__setattr__(new_decoupled, "adaptive_grace", True)
        config = ExperimentConfig(
            config.method,
            run=config.run,
            decoupled=new_decoupled,
            monitoring=config.monitoring,
        )
        options = self.options()
        options.steps = 5
        # Learner 1 has factor=2. If dataset took 2s, step 1 sleep would be 2*(2-1)=2s.
        options.island_slowness_factors = [1, 2]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            start = time.monotonic()
            with redirect_stdout(sys.stdout), redirect_stderr(sys.stderr):
                status = _run_training(config, options, _TinyTokenModel(2), ["0", "1"], folder, 20.0, ROOT, worker_command=[sys.executable, str(Path(__file__).with_name("cpu_training_worker.py")), "--slow-first-batch", "2.0"])
            duration = time.monotonic() - start
            
            self.assertEqual(status, 0)
            
            summary = json.loads((folder / "summary.json").read_text())
            self.assertLess(summary["peak_inflight_captures"], 3)
            self.assertLess(duration, 10.0)


def math_isfinite(value):
    import math
    return math.isfinite(float(value))


if __name__ == "__main__":
    unittest.main()
