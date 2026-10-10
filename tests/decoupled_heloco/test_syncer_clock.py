"""Shared clock semantics and real-process clock-budget shutdown."""
import copy
import csv
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import torch

from panoengine.decentralized.decoupled_heloco.config import DecoupledConfig, ExperimentConfig, ConfigError
from panoengine.decentralized.decoupled_heloco.learner import DecoupledLearner, SyncerClockComplete
from panoengine.decentralized.decoupled_heloco.gpu_training import _run_training
from panoengine.decentralized.decoupled_heloco.smoke import _TinyTokenModel
from panoengine.decentralized.decoupled_heloco.syncer import DecoupledSyncer
from panoengine.decentralized.decoupled_heloco.transport import metadata_from_wire

ROOT = Path(__file__).resolve().parents[2]


class SyncerClockTests(unittest.TestCase):
    def test_clock_lr_policy_is_constant_or_preserves_explicit_local_horizon(self):
        from panoengine.decentralized.decoupled_heloco.gpu_recipe import configure_clock_learning_rate
        cfg = SimpleNamespace(lr_scheduler=SimpleNamespace(warmup_steps=20, min_lr_factor=0, decay_type='linear'))
        clock = ExperimentConfig('decoupled_diloco', decoupled=DecoupledConfig(
            scheduler='paper_offsets', stopping='syncer_steps', syncer_steps=10))
        configure_clock_learning_rate(cfg, clock)
        self.assertEqual((cfg.lr_scheduler.warmup_steps, cfg.lr_scheduler.min_lr_factor), (0, 1))
        local = SimpleNamespace(lr_scheduler=SimpleNamespace(warmup_steps=20, min_lr_factor=0, decay_type='linear'))
        configure_clock_learning_rate(local, ExperimentConfig('decoupled_diloco'))
        self.assertEqual((local.lr_scheduler.warmup_steps, local.lr_scheduler.min_lr_factor), (20, 0))
        explicit = ExperimentConfig('decoupled_diloco', decoupled=DecoupledConfig(
            scheduler='paper_offsets', stopping='syncer_steps', syncer_steps=10, clock_lr_schedule='local_horizon'))
        configure_clock_learning_rate(local, explicit)
        self.assertEqual((local.lr_scheduler.warmup_steps, local.lr_scheduler.min_lr_factor), (20, 0))

    def test_clock_comparison_rejects_mixed_budgets_before_launch(self):
        import yaml
        import run_method_comparison as comparison
        with tempfile.TemporaryDirectory() as directory:
            data = yaml.safe_load((ROOT/'run_script/method_comparison.yaml').read_text())
            data['decoupled'].update(stopping='syncer_steps', syncer_steps=7)
            path = Path(directory)/'comparison.yaml'
            path.write_text(yaml.safe_dump(data))
            with self.assertRaisesRegex(ConfigError, 'decoupled-only'):
                comparison.load_comparison(path)
            data['method_run'] = ['decoupled_diloco', 'decoupled_heloco']
            path.write_text(yaml.safe_dump(data))
            self.assertEqual(comparison.load_comparison(path)['methods'], data['method_run'])

    def test_empty_sparse_budget_keeps_initial_snapshot_and_exports_zero_revisions(self):
        from panoengine.decentralized.decoupled_heloco.evaluation import load_global_parameters
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig('decoupled_diloco', decoupled=DecoupledConfig(
                num_fragments=2, min_quorum=2, scheduler='paper_offsets', sync_period=5,
                fragment_offsets=[2, 3], stopping='syncer_steps', syncer_steps=1),
                monitoring={'enabled': True})
            options = SimpleNamespace(islands=2, steps=1, outer_lr=.7, outer_momentum=.9,
                                      ps_timeout=20., island_slowness_factors=[1, 1])
            folder = Path(directory)
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                status = _run_training(config, options, _TinyTokenModel(2), ['0', '1'], folder, 30., ROOT,
                    worker_command=[sys.executable, str(Path(__file__).with_name('cpu_training_worker.py'))])
            self.assertEqual(status, 0, output.getvalue())
            summary = json.loads((folder/'summary.json').read_text())
            self.assertEqual(summary['fragment_revisions'], [0, 0])
            self.assertEqual(summary['sync_updates'], 0)
            self.assertEqual(summary['syncer_step'], 1)
            self.assertTrue((folder/'trajectory/00000000.pt').exists())
            self.assertTrue((folder/'trajectory/00000000-final.pt').exists())
            payload = torch.load(folder/'global_model.pt', weights_only=True)
            model = _TinyTokenModel(2)
            load_global_parameters(model, payload, num_fragments=2, expected_syncer_step=1)
            with self.assertRaises(ValueError):
                load_global_parameters(model, payload, num_fragments=2)
            with self.assertRaises(ValueError):
                load_global_parameters(model, payload, num_fragments=2, expected_syncer_step=2)

    def test_clock_budget_survives_a_crashed_learner_over_tcp(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig('decoupled_heloco', decoupled=DecoupledConfig(
                num_fragments=2, min_quorum=2, overlap_steps=2, sync_interval=.05,
                grace_window_factor=0, scheduler='paper_offsets', sync_period=5,
                fragment_offsets=[1, 3], stopping='syncer_steps', syncer_steps=12))
            options = SimpleNamespace(islands=3, steps=1, outer_lr=.7, outer_momentum=.9,
                                      ps_timeout=20., island_slowness_factors=[1, 1, 1])
            folder = Path(directory)
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                status = _run_training(config, options, _TinyTokenModel(2), ['0', '1', '2'], folder, 30., ROOT,
                    worker_command=[sys.executable, str(Path(__file__).with_name('cpu_training_worker.py')), '--fail-learner', '2'])
            self.assertEqual(status, 0, output.getvalue())
            summary = json.loads((folder/'summary.json').read_text())
            self.assertTrue(summary['degraded'])
            self.assertEqual(summary['completed_learner_ids'], [0, 1])
            self.assertEqual(summary['syncer_step'], 12)
            self.assertEqual([row['syncer_step'] for row in summary['learners'][:2]], [12, 12])

    def test_budget_validation(self):
        for settings in ({'stopping': 'unknown'}, {'stopping': 'syncer_steps'},
                         {'stopping': 'syncer_steps', 'scheduler': 'paper_offsets', 'syncer_steps': True},
                         {'stopping': 'syncer_steps', 'scheduler': 'paper_offsets', 'syncer_steps': 0},
                         {'stopping': 'syncer_steps', 'syncer_steps': 10}, {'syncer_steps': 10}):
            with self.subTest(settings=settings), self.assertRaises(ConfigError):
                DecoupledConfig.from_mapping(settings)
        config = DecoupledConfig.from_mapping({'stopping': 'syncer_steps', 'scheduler': 'paper_offsets', 'syncer_steps': 10})
        self.assertEqual(config.syncer_steps, 10)
        self.assertEqual(DecoupledConfig.from_mapping({}).stopping, 'local_steps')

    def test_clock_is_applied_with_weights_at_safe_boundary_and_never_regresses(self):
        model = torch.nn.Linear(1, 1, bias=False)
        learner = DecoupledLearner(model, 1, learner_id=0, stop_at_syncer_step=7)
        initial = model.weight.detach().clone()
        learner.begin_step()
        learner.queue_update(0, {'weight': initial + 1}, 1,
                             layout_signature=learner.manager.layout_signature, global_step=7)
        self.assertEqual(learner.metadata().syncer_step, 0)
        torch.testing.assert_close(model.weight, initial)
        learner.end_step(8)
        self.assertEqual(learner.metadata().syncer_step, 7)
        torch.testing.assert_close(model.weight, initial + 1)
        learner.queue_syncer_clock(2)
        with self.assertRaises(SyncerClockComplete):
            learner.begin_step()
        self.assertEqual(learner.metadata().total_local_steps, 1)
        self.assertEqual(learner.metadata().syncer_step, 7)
        with self.assertRaises(ValueError):
            learner.queue_update(0, {'weight': initial + 1}, 1,
                                 layout_signature=learner.manager.layout_signature, global_step=8)
        for bad in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                learner.queue_syncer_clock(bad)

    def test_latest_revision_catchup_restores_clock_without_repeating_outer_update(self):
        model = torch.nn.Linear(1, 1, bias=False)
        local = copy.deepcopy(model)
        learner = DecoupledLearner(local, 1, learner_id=0)
        syncer = DecoupledSyncer(model, [learner], 1, min_quorum=1, overlap_steps=1,
                                scheduler='paper_offsets', sync_period=4, fragment_offsets=[2], outer_method='sgd')
        with learner.training_step(8), torch.no_grad():
            local.weight.sub_(.1)
        syncer.begin_sync(); learner.boundary(); syncer.poll(); learner.boundary()
        self.assertEqual(syncer.global_step, 2)
        self.assertEqual(learner.metadata().syncer_step, 2)
        replacement = DecoupledLearner(copy.deepcopy(model), 1, learner_id=0)
        syncer.reconnect_learner(replacement); replacement.boundary()
        self.assertEqual(replacement.metadata().syncer_step, 2)
        self.assertEqual(syncer.fragment_revisions, (1,))

    def test_clock_budget_over_tcp_includes_sparse_terminal_slot_and_ignores_local_cap(self):
        for method, target in [('decoupled_diloco', 7), ('decoupled_heloco', 8)]:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                d = DecoupledConfig(num_fragments=2, min_quorum=2, overlap_steps=2,
                    sync_interval=.05, grace_window_factor=0, scheduler='paper_offsets',
                    sync_period=5, fragment_offsets=[1, 3], stopping='syncer_steps', syncer_steps=target)
                config = ExperimentConfig(method, decoupled=d, monitoring={'enabled': True, 'global_every_updates': 1})
                options = SimpleNamespace(islands=2, steps=1, outer_lr=.7, outer_momentum=.9,
                                          ps_timeout=20., island_slowness_factors=[1, 2])
                folder = Path(directory)
                output = io.StringIO()
                with redirect_stdout(output), redirect_stderr(output):
                    status = _run_training(config, options, _TinyTokenModel(2), ['0', '1'], folder, 30., ROOT,
                        worker_command=[sys.executable, str(Path(__file__).with_name('cpu_training_worker.py'))])
                self.assertEqual(status, 0, output.getvalue() + '\n' + '\n'.join(p.read_text() for p in folder.glob('learner_*/trainer.log')))
                summary = json.loads((folder/'summary.json').read_text())
                self.assertEqual(summary['syncer_step'], target)
                self.assertIsNone(summary['steps_per_learner'])
                self.assertTrue(all(row['syncer_step'] == target for row in summary['learners']))
                self.assertTrue(all(row['total_local_steps'] > options.steps for row in summary['learners']))
                with (folder/'syncs.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                expected = [t for t in range(1, target + 1) if t % 5 in (1, 3)]
                self.assertEqual([int(row['global_step']) for row in rows], expected)
                self.assertEqual(summary['sync_updates'], len(expected))
                import run_method_comparison as comparison
                self.assertEqual(comparison.completed_run(folder, config, options)['syncer_step'], target)
                final = torch.load(next((folder/'trajectory').glob('*-final.pt')), weights_only=True)
                self.assertEqual(final['processed_tokens'], sum(row['total_tokens'] for row in summary['learners']))
                for learner in summary['learners']:
                    child = folder/f"learner_{learner['learner_id']}"
                    final = torch.load(next((child/'trajectory').glob('*-final.pt')), weights_only=True)
                    self.assertEqual(final['processed_tokens'], learner['total_tokens'])

