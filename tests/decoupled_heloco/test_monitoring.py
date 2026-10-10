import csv
from pathlib import Path
import tempfile
import time
import unittest
import torch
from panoengine.decentralized.decoupled_heloco.monitoring import Trajectory, CentralMonitor, attach_baseline
from panoengine.decentralized.decoupled_heloco.config import load_config, ConfigError

class MonitoringTests(unittest.TestCase):
    def test_snapshots_are_immutable_and_include_zero_interval_and_final(self):
        with tempfile.TemporaryDirectory() as directory:
            t=Trajectory(directory,{'enabled':True,'learner_every_steps':3},kind='learner')
            parameter=torch.ones(2)
            t.save(0,{'w':parameter},force=True)
            parameter.add_(2)
            t.save(1,{'w':parameter})
            t.save(3,{'w':parameter},tokens=24)
            t.save(4,{'w':parameter},tokens=32,force=True)
            files=sorted((Path(directory)/'trajectory').glob('*.pt'))
            self.assertEqual(len(files),3)
            zero=torch.load(files[0],weights_only=True)
            self.assertEqual(zero['elapsed_s'],0)
            self.assertTrue(torch.equal(zero['parameters']['w'],torch.ones(2)))
            self.assertEqual(torch.load(files[-1],weights_only=True)['processed_tokens'],32)

    def test_forced_final_keeps_last_update_and_completed_tokens(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            t = Trajectory(directory, {'enabled': True, 'global_every_updates': 1}, kind='global', origin=10)
            weights = {'w': torch.ones(2)}
            t.save(0, weights, force=True)
            with patch('panoengine.decentralized.decoupled_heloco.monitoring.time.monotonic', return_value=12):
                t.save(1, weights, tokens=64)
            with patch('panoengine.decentralized.decoupled_heloco.monitoring.time.monotonic', return_value=15):
                t.save_from(1, lambda: weights, tokens=80, force=True)
            regular = torch.load(Path(directory)/'trajectory/00000001.pt', weights_only=True)
            final = torch.load(Path(directory)/'trajectory/00000001-final.pt', weights_only=True)
            self.assertEqual((regular['processed_tokens'], regular['elapsed_s']), (64, 2))
            self.assertEqual((final['processed_tokens'], final['elapsed_s']), (80, 5))
            self.assertEqual(final['snapshot_event'], 'final')
            torch.testing.assert_close(regular['parameters']['w'], final['parameters']['w'])
            self.assertEqual(len(list((Path(directory)/'trajectory').glob('*.pt'))), 3)

    def test_final_without_outer_updates_preserves_initial_point(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            t = Trajectory(directory, {'enabled': True}, kind='global', origin=10)
            t.save(0, {'w': torch.ones(1)}, force=True)
            with patch('panoengine.decentralized.decoupled_heloco.monitoring.time.monotonic', return_value=15):
                t.save(0, {'w': torch.ones(1)}, tokens=80, force=True)
            zero = torch.load(Path(directory)/'trajectory/00000000.pt', weights_only=True)
            final = torch.load(Path(directory)/'trajectory/00000000-final.pt', weights_only=True)
            self.assertEqual(zero['processed_tokens'], 0)
            self.assertEqual(zero['elapsed_s'], 0)
            self.assertEqual(final['elapsed_s'], 5)

    def test_endpoint_validation_checks_completed_tokens_and_order(self):
        from panoengine.decentralized.decoupled_heloco.global_validation import validate_final_trajectory
        run = {'folder': Path('/example'), 'summary': {'learners': [
            {'learner_id': 0, 'total_tokens': 40}, {'learner_id': 1, 'total_tokens': 40}]}}
        initial = {'snapshot_event': 'initial', 'processed_tokens': 0, 'elapsed_s': 0}
        final = {'snapshot_event': 'final', 'processed_tokens': 80, 'elapsed_s': 5}
        self.assertEqual(validate_final_trajectory(run, 'global', 'global', [initial, final]), 'verified')
        with self.assertRaises(ConfigError):
            validate_final_trajectory(run, 'global', 'global', [initial, dict(final, processed_tokens=64)])
        with self.assertRaises(ConfigError):
            validate_final_trajectory(run, 'global', 'global', [initial])
        with self.assertRaises(ConfigError):
            validate_final_trajectory(run, 'global', 'global', [dict(initial, elapsed_s=6), final])
        self.assertEqual(validate_final_trajectory(run, 'learner', 'learner_0',
            [initial, dict(final, processed_tokens=40)]), 'verified')
        self.assertEqual(validate_final_trajectory(run, 'global', 'global',
            [{'processed_tokens': 64, 'elapsed_s': 4}]), 'legacy_unverified')
        run['summary']['failed_learners'] = {'1': 'crashed'}
        self.assertEqual(validate_final_trajectory(run, 'learner', 'learner_1', [initial]),
                         'interrupted_learner')

    def test_baseline_hooks_record_real_updates_and_memory_without_changing_result(self):
        class Server:
            _revision=0
            def _commit_step_locked(self):
                model.weight.data.add_(1);self._revision+=1
            def _apply_one(self):self._commit_step_locked()
            def _build_snapshot_locked(self):return model.weight.detach().clone()
        model=torch.nn.Linear(1,1,bias=False)
        with tempfile.TemporaryDirectory() as directory:
            monitor=CentralMonitor(directory,{'enabled':True,'global_every_updates':1,'memory_sample_seconds':.01})
            server=Server();attach_baseline(server,model,monitor)
            initial=model.weight.detach().clone()
            server._apply_one();server._build_snapshot_locked();monitor.close()
            self.assertTrue(torch.equal(model.weight,initial+1))
            with (Path(directory)/'central_memory.csv').open() as stream:
                samples=list(csv.DictReader(stream))
            self.assertGreater(int(samples[0]['rss_bytes']),0)
            with (Path(directory)/'central_processing.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual([r['phase'] for r in rows],['outer_apply','dispatch_snapshot'])
            self.assertTrue(all(float(r['wall_s'])>=0 for r in rows))
            self.assertTrue((Path(directory)/'trajectory/00000001.pt').is_file())

    def test_disabled_monitoring_creates_no_measurement_files(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor=CentralMonitor(directory,{})
            monitor.trajectory.save(0,{'w':torch.ones(1)},force=True)
            monitor.close()
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_measured_global_snapshots_follow_actual_syncer_updates(self):
        import copy
        from panoengine.decentralized.decoupled_heloco.learner import DecoupledLearner
        from panoengine.decentralized.decoupled_heloco.syncer import DecoupledSyncer
        global_model=torch.nn.Linear(1,1,bias=False)
        local=copy.deepcopy(global_model)
        learner=DecoupledLearner(local,1,learner_id=0)
        syncer=DecoupledSyncer(global_model,[learner],1,min_quorum=1,overlap_steps=1,outer_method='sgd',outer_lr=.5)
        with tempfile.TemporaryDirectory() as directory:
            monitor=CentralMonitor(directory,{'enabled':True,'global_every_updates':1})
            syncer.monitor=monitor
            monitor.trajectory.save(0,syncer.optimizer.model_snapshot(),force=True)
            with learner.training_step(8):local.weight.data.sub_(.2)
            syncer.begin_sync();learner.boundary();syncer.poll();monitor.close()
            zero=torch.load(Path(directory)/'trajectory/00000000.pt',weights_only=True)
            final=torch.load(Path(directory)/'trajectory/00000001.pt',weights_only=True)
            torch.testing.assert_close(final['parameters']['weight'],zero['parameters']['weight']-.1)
            self.assertEqual(final['processed_tokens'],8)
            self.assertTrue((Path(directory)/'central_processing.csv').exists())

    def test_held_out_trajectory_uses_changed_global_weights_and_verifies_zero(self):
        from test_evaluation import TokenTable, batch
        from panoengine.decentralized.decoupled_heloco.evaluation import evaluate_model, evaluate_trajectory_snapshot, parameter_fingerprint
        model=TokenTable()
        frozen=[batch([0,1],[0,1])]
        initial={n:p.detach().clone() for n,p in model.named_parameters()}
        sha=parameter_fingerprint(initial)
        measured=evaluate_model(model,frozen,device='cpu',vocab_size=3)
        zero=evaluate_trajectory_snapshot(model,{'step':0,'parameters':initial},frozen,
            device='cpu',vocab_size=3,initial_sha=sha,initial_metrics=measured)
        self.assertEqual(zero['loss'],measured['loss'])
        changed={n:torch.zeros_like(p) for n,p in initial.items()}
        final=evaluate_trajectory_snapshot(model,{'step':1,'parameters':changed},frozen,
            device='cpu',vocab_size=3,initial_sha=sha,initial_metrics=measured)
        self.assertGreater(final['loss'],zero['loss'])
        with self.assertRaises(ValueError):
            evaluate_trajectory_snapshot(model,{'step':0,'parameters':changed},frozen,
                device='cpu',vocab_size=3,initial_sha=sha,initial_metrics=measured)
