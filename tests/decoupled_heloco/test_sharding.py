"""Real IPC replicas: numerical equivalence, empty shards and fail-stop clocks."""
import copy
import csv
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import os
import signal
import time
import tempfile
import unittest
from unittest.mock import patch

import torch

from panoengine.decentralized.decoupled_heloco.config import DecoupledConfig, ConfigError, ExperimentConfig, HeLoCoConfig
from panoengine.decentralized.decoupled_heloco.fragment_manager import FragmentManager
from panoengine.decentralized.decoupled_heloco.optimizer import FragmentDiLoCo, FragmentHeLoCo
from panoengine.decentralized.decoupled_heloco.merging import merge_gradients
from panoengine.decentralized.decoupled_heloco.sharding import ShardedOptimizer
from panoengine.decentralized.decoupled_heloco.learner import DecoupledLearner
from panoengine.decentralized.decoupled_heloco.syncer import DecoupledSyncer
from panoengine.decentralized.decoupled_heloco.gpu_training import _run_training
from panoengine.decentralized.decoupled_heloco.smoke import _TinyTokenModel

ROOT = Path(__file__).resolve().parents[2]


class ShardingTests(unittest.TestCase):
    def test_configuration(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ConfigError):
                DecoupledConfig.from_mapping({'syncer_shards': value})
        for value in (0, -1, True, float('inf')):
            with self.assertRaises(ConfigError):
                DecoupledConfig.from_mapping({'syncer_timeout': value})
        self.assertEqual(DecoupledConfig.from_mapping({'syncer_shards': 3}).syncer_shards, 3)

    def test_replicated_state_matches_serial_all_merge_and_correction_modes(self):
        torch.manual_seed(13)
        model = _TinyTokenModel(2)
        manager = FragmentManager.from_model(model, 2)
        for method, order in [('diloco', None), ('heloco', 'merge_then_correct'), ('heloco', 'correct_then_merge')]:
            options = dict(lr=.3, momentum=.9)
            if order:
                options['config'] = replace(HeLoCoConfig(), correction_order=order)
            cls = FragmentDiLoCo if method == 'diloco' else FragmentHeLoCo
            reference = cls(manager, dict(model.named_parameters()), **options)
            backend = ShardedOptimizer(reference.model_snapshot(), manager, method, options, 3, 30)
            try:
                self.assertEqual(len(set(backend.pids)), 3)
                for step, mode in enumerate(('weighted_average', 'rda', 'paper_rda', 'rda')):
                    fragment = step % 2
                    snapshots = {}
                    # Rank 2 stays alive with no learner contribution.
                    for learner_id in (0, 1):
                        baseline = reference.snapshot(fragment)
                        current = {n: t - torch.randn_like(t) * .05 for n, t in baseline.items()}
                        snapshots[learner_id] = SimpleNamespace(baseline=baseline, current=current)
                    weights = (.25, .75)
                    merged = {}
                    before = order == 'correct_then_merge'
                    for name in manager.fragment(fragment).parameter_names:
                        gradients = [snapshots[i].baseline[name]-snapshots[i].current[name] for i in (0, 1)]
                        if before:
                            gradients = [reference.correct({name:g})[name] for g in gradients]
                        actual_mode = mode
                        if mode == 'paper_rda':
                            actual_mode = 'weighted_average' if {'embedding', 'tok_embeddings', 'embed_tokens'}.intersection(name.split('.')) else 'rda'
                        merged[name] = merge_gradients(gradients, weights, actual_mode)
                    expected = reference.step(fragment, merged, already_corrected=before) if order else reference.step(fragment, merged)
                    actual, norm = backend.merge_step(fragment, snapshots, weights, mode)
                    for name in expected:
                        torch.testing.assert_close(actual[name], expected[name], atol=2e-6, rtol=2e-5)
                    for rank in range(3):
                        replica = backend._read('model_snapshot', rank=rank)
                        for name, value in reference.model_snapshot().items():
                            torch.testing.assert_close(replica[name], value, atol=2e-6, rtol=2e-5)
                        momentum = backend._read('momentum_snapshot', fragment, rank=rank)
                        for name, value in reference.momentum_snapshot(fragment).items():
                            torch.testing.assert_close(momentum[name], value, atol=2e-6, rtol=2e-5)
                self.assertTrue(all(row['collective_tensor_bytes'] > 0 for row in backend.stats))
            finally:
                backend.close()
            self.assertTrue(all(not p.is_alive() for p in backend.processes))

    def test_replica_failure_cannot_advance_revision_or_clock(self):
        model = torch.nn.Linear(2, 1)
        locals = [copy.deepcopy(model) for _ in range(2)]
        learners = [DecoupledLearner(m, 2, learner_id=i) for i, m in enumerate(locals)]
        syncer = DecoupledSyncer(model, learners, 2, min_quorum=2, overlap_steps=1,
                                 outer_method='diloco', syncer_shards=2, syncer_timeout=10)
        try:
            for learner, local in zip(learners, locals):
                with learner.training_step(8), torch.no_grad():
                    for p in local.parameters():
                        p.sub_(.1)
            syncer.begin_sync()
            for learner in learners:
                learner.boundary()
            syncer.optimizer.processes[1].terminate()
            syncer.optimizer.processes[1].join(2)
            with self.assertRaises((RuntimeError, OSError, TimeoutError)):
                syncer.poll()
            self.assertEqual(syncer.fragment_revisions, (0, 0))
            self.assertEqual(syncer.global_step, 0)
            self.assertTrue(all(not p.is_alive() for p in syncer.optimizer.processes))
        finally:
            syncer.close()

    def test_stopped_replica_send_has_bounded_timeout_and_cleanup(self):
        model = torch.nn.Linear(256, 256)
        manager = FragmentManager.from_model(model, 2)
        reference = FragmentDiLoCo(manager, dict(model.named_parameters()), lr=.3)
        backend = ShardedOptimizer(reference.model_snapshot(), manager, 'diloco', {'lr':.3}, 2, 20)
        backend.timeout = .5
        baseline = reference.snapshot(0)
        snapshot = SimpleNamespace(baseline=baseline, current={n:t-.1 for n,t in baseline.items()})
        try:
            os.kill(backend.pids[1], signal.SIGSTOP)
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                backend.merge_step(0, {1:snapshot}, (1.,), 'weighted_average')
            self.assertLess(time.monotonic()-started, 5.)
            self.assertTrue(all(not p.is_alive() for p in backend.processes))
        finally:
            backend.close(force=True)

    def test_resource_report_includes_replicas_and_rejects_missing_aggregate(self):
        from run_method_comparison import generate_memory_plot
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'central_memory.csv').write_text('rss_bytes\n1048576\n')
            (root/'central_processing.csv').write_text('phase,wall_s,thread_cpu_s\nmerge_correction_outer_update,3,2\n')
            (root/'monitoring.json').write_text(json.dumps({'snapshot_overhead_s':0}))
            (root/'syncer_memory.csv').write_text('role,rss_bytes\ncoordinator,1048576\naggregate,7340032\n')
            (root/'syncer_replica_processing.csv').write_text('rank,process_cpu_s\n0,2\n1,2\n')
            runs = {'decoupled_diloco':{'status':'passed', 'folder':str(root), 'summary':{'syncer_shards':2, 'elapsed_s':5, 'learners':[{'total_tokens':12}]}}}
            with patch.dict(sys.modules, {'matplotlib.pyplot':None}):
                generate_memory_plot(root, runs)
            with (root/'memory_footprint.csv').open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row['peak_central_rss_mib']), 1.)
            self.assertEqual(float(row['peak_syncer_total_rss_mib']), 7.)
            self.assertEqual(float(row['total_processing_cpu_s']), 6.)
            (root/'syncer_memory.csv').unlink()
            (root/'memory_footprint.csv').unlink()
            with redirect_stdout(io.StringIO()):
                generate_memory_plot(root, runs)
            self.assertFalse((root/'memory_footprint.csv').exists())

    def test_tcp_pipeline_clock_and_missing_learner_shard(self):
        for method in ('decoupled_diloco', 'decoupled_heloco'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                config = ExperimentConfig(method, decoupled=DecoupledConfig(
                    num_fragments=2, min_quorum=2, overlap_steps=20, max_inflight_captures=2,
                    syncer_shards=3, syncer_timeout=30,
                    sync_interval=.03, grace_window_factor=1., adaptive_grace=False,
                    scheduler='paper_offsets', sync_period=5, fragment_offsets=[1, 3],
                    stopping='syncer_steps', syncer_steps=8), monitoring={'enabled': True})
                options = SimpleNamespace(islands=3, steps=1, outer_lr=.7, outer_momentum=.9,
                    ps_timeout=40., island_slowness_factors=[1, 2, 1])
                folder = Path(directory); output = io.StringIO()
                with redirect_stdout(output), redirect_stderr(output):
                    status = _run_training(config, options, _TinyTokenModel(2), ['0', '1', '2'], folder, 60., ROOT,
                        worker_command=[sys.executable, str(Path(__file__).with_name('cpu_training_worker.py')),
                                        '--capture-delay-seconds', '.35', '--fail-learner', '2', '--fail-after', '2'])
                logs = '\n'.join(p.read_text() for p in folder.glob('learner_*/trainer.log'))
                self.assertEqual(status, 0, output.getvalue()+logs)
                summary = json.loads((folder/'summary.json').read_text())
                self.assertEqual(summary['syncer_step'], 8)
                self.assertEqual(summary['syncer_shards'], 3)
                self.assertEqual(len(summary['syncer_replica_pids']), 3)
                self.assertTrue(summary['degraded'])
                self.assertEqual(summary['peak_inflight_captures'], 2)
                with (folder/'syncs.csv').open() as stream:
                    self.assertEqual([int(r['global_step']) for r in csv.DictReader(stream)], [1, 3, 6, 8])
                with (folder/'syncer_memory.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                aggregate = next(r for r in rows if r['role'] == 'aggregate')
                same = [r for r in rows if r['elapsed_s'] == aggregate['elapsed_s'] and r['role'] != 'aggregate']
                self.assertEqual(len(same), 4)
                self.assertEqual(int(aggregate['rss_bytes']), sum(int(r['rss_bytes']) for r in same))
                with (folder/'syncer_replica_processing.csv').open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 12)


if __name__ == '__main__':
    unittest.main()
