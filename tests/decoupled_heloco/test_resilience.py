"""Healthy quorums progress through failed deliveries, disconnects and crashes."""

import copy
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from panoengine.decentralized.decoupled_heloco.config import DecoupledConfig, ExperimentConfig
from panoengine.decentralized.decoupled_heloco.gpu_training import _run_training, _pump_until, _send_available
from panoengine.decentralized.decoupled_heloco.learner import DecoupledLearner
from panoengine.decentralized.decoupled_heloco.smoke import _TinyTokenModel
from panoengine.decentralized.decoupled_heloco.syncer import DecoupledSyncer
from panoengine.decentralized.decoupled_heloco.transport import RemoteLearner

ROOT = Path(__file__).resolve().parents[2]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.ones(2))
        self.b = torch.nn.Parameter(torch.ones(2))


class ResilienceTests(unittest.TestCase):
    def make(self, method):
        model = Model()
        self.models = [copy.deepcopy(model) for _ in range(3)]
        self.learners = [DecoupledLearner(m, 2, learner_id=i) for i, m in enumerate(self.models)]
        self.syncer = DecoupledSyncer(model, self.learners, 2, min_quorum=2, overlap_steps=1, outer_method=method)

    def cycle(self):
        for i in (0, 1):
            with self.learners[i].training_step(tokens=8), torch.no_grad():
                for p in self.models[i].parameters():
                    p.sub_(.1)
        self.assertIsNotNone(self.syncer.begin_sync())
        for learner in self.learners[:2]:
            learner.boundary()
        result = self.syncer.poll()
        self.assertEqual(result.learner_ids, (0, 1))
        for learner in self.learners[:2]:
            learner.boundary()
        return result

    def test_failed_delivery_does_not_block_other_fragments_or_later_cycles(self):
        for method in ("diloco", "heloco"):
            with self.subTest(method=method):
                self.make(method)
                with patch.object(self.learners[2], "queue_update", side_effect=ConnectionError("offline")):
                    for step in range(6):
                        result = self.cycle()
                        self.assertEqual(result.sync_step, step)
                        self.assertEqual(result.pending_broadcast, (2,))
                    self.assertEqual(len(self.syncer._broadcasts), 2)
                    self.assertEqual(self.syncer.fragment_revisions, (3, 3))
                before = self.syncer.optimizer.model_snapshot()
                moments = [self.syncer.optimizer.momentum_snapshot(f) for f in (0, 1)]
                self.assertEqual(self.syncer.retry_broadcast(), ())
                self.learners[2].boundary()
                for fragment in self.learners[2].metadata().fragments:
                    self.assertEqual(fragment.last_applied_revision, 3)
                    self.assertEqual(fragment.local_steps, 0)
                for name, p in self.models[2].named_parameters():
                    fragment = next(f.fragment_id for f in self.syncer.manager.fragments if name in f.parameter_names)
                    torch.testing.assert_close(p, self.syncer.optimizer.dispatch_snapshot(fragment)[name])
                    torch.testing.assert_close(self.syncer.optimizer.model_snapshot()[name], before[name], rtol=0, atol=0)
                for f in (0, 1):
                    for name, value in moments[f].items():
                        torch.testing.assert_close(self.syncer.optimizer.momentum_snapshot(f)[name], value, rtol=0, atol=0)

    def test_replacement_endpoint_catches_up_after_successful_outboxes_are_gone(self):
        self.make("heloco")
        for _ in range(4):
            self.cycle()
            self.learners[2].boundary()
        self.assertEqual(self.syncer._broadcasts, {})
        replacement_model = Model()
        replacement = DecoupledLearner(replacement_model, 2, learner_id=2)
        self.syncer.reconnect_learner(replacement)
        replacement.boundary()
        for f in replacement.metadata().fragments:
            self.assertEqual(f.last_applied_revision, 2)
        for f in (0, 1):
            for name, value in self.syncer.optimizer.dispatch_snapshot(f).items():
                torch.testing.assert_close(dict(replacement_model.named_parameters())[name], value)

    def test_a_pull_connection_failure_excludes_only_that_contributor(self):
        self.make("diloco")
        for i in range(3):
            with self.learners[i].training_step(tokens=8), torch.no_grad():
                self.models[i].a.sub_(.1)
        with patch.object(self.learners[2], "request_snapshot", side_effect=ConnectionError("disconnected")):
            self.assertEqual(self.syncer.begin_sync().learner_ids, (0, 1))
            for learner in self.learners[:2]:
                learner.boundary()
            self.assertEqual(self.syncer.poll().learner_ids, (0, 1))

    def test_remote_disconnect_fails_pending_captures_and_reconnect_resets_transport_state(self):
        learner = DecoupledLearner(Model(), 2, learner_id=0)
        transport = SimpleNamespace(send_reliable=lambda *a, **kw: None, receive=lambda: (_ for _ in ()).throw(ConnectionError("EOF")), close=lambda: None)
        remote = RemoteLearner(transport, learner.metadata())
        future = remote.request_snapshot("0", 0, 0)
        remote.pump()
        self.assertFalse(remote.available)
        with self.assertRaises(ConnectionError):
            future.result()
        remote.release_snapshot("0")
        replacement_transport = SimpleNamespace(send_reliable=lambda *a, **kw: None, receive=lambda: None)
        remote.reconnect(replacement_transport, learner.metadata())
        self.assertTrue(remote.available)
        self.assertIsNone(remote.failure)
        self.assertFalse(remote.request_snapshot("1", 0, 0).done())

    def test_below_quorum_fails_explicitly(self):
        peer = RemoteLearner(SimpleNamespace(receive=lambda: None), DecoupledLearner(Model(), 2, learner_id=0).metadata())
        process = SimpleNamespace(poll=lambda: 17)
        import time
        with self.assertRaisesRegex(RuntimeError, "below min_quorum"):
            _pump_until([peer], [process], time.monotonic() + 1, lambda: True, min_quorum=1)

    def test_a_disconnect_during_control_send_does_not_abort_other_peers(self):
        sent = []
        peers = [RemoteLearner(SimpleNamespace(send_reliable=lambda *args, **kw: sent.append(args)), DecoupledLearner(Model(), 2, learner_id=i).metadata()) for i in range(2)]
        peers[0].transport.send_reliable = lambda *args, **kw: (_ for _ in ()).throw(ConnectionError("EOF during pause"))
        _send_available(peers, "pause")
        self.assertFalse(peers[0].available)
        self.assertTrue(peers[1].available)
        self.assertEqual(sent, [("pause", {})])

    def test_hard_learner_crash_preserves_surviving_budgets_over_real_tcp(self):
        for method in ("decoupled_diloco", "decoupled_heloco"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                config = ExperimentConfig(method, decoupled=DecoupledConfig(num_fragments=2, min_quorum=2, overlap_steps=2, sync_interval=.05, grace_window_factor=.2))
                options = SimpleNamespace(islands=3, steps=18, outer_lr=.7, outer_momentum=.9, ps_timeout=20.0, island_slowness_factors=[1, 1, 1])
                output = io.StringIO()
                folder = Path(directory)
                with redirect_stdout(output), redirect_stderr(output):
                    status = _run_training(config, options, _TinyTokenModel(2), ["0", "1", "2"], folder, 30., ROOT,
                        worker_command=[sys.executable, str(Path(__file__).with_name("cpu_training_worker.py")), "--fail-learner", "2"])
                logs = "\n".join(p.read_text() for p in folder.glob("learner_*/trainer.log"))
                self.assertEqual(status, 0, output.getvalue() + logs)
                summary = json.loads((folder / "summary.json").read_text())
                self.assertTrue(summary["degraded"])
                self.assertEqual(summary["completed_learner_ids"], [0, 1])
                self.assertIn("2", summary["failed_learners"])
                self.assertEqual([m["total_local_steps"] for m in summary["learners"][:2]], [18, 18])
                self.assertTrue(all(r >= 1 for r in summary["fragment_revisions"]))
                self.assertTrue((folder / "global_model.pt").is_file())


if __name__ == "__main__":
    unittest.main()
