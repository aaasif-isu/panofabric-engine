"""Protocol completion versus delayed exit, failure, and bounded cleanup."""
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

from panoengine.decentralized.decoupled_heloco.config import ExperimentConfig, DecoupledConfig
from panoengine.decentralized.decoupled_heloco.gpu_training import _run_training
from panoengine.decentralized.decoupled_heloco.shutdown import wait_for_learner_shutdown
from panoengine.decentralized.decoupled_heloco.smoke import _TinyTokenModel

ROOT = Path(__file__).resolve().parents[2]


class ShutdownTests(unittest.TestCase):
    def test_failure_is_reported_without_waiting_for_another_teardown(self):
        slow = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
        failed = subprocess.Popen([sys.executable, '-c', 'raise SystemExit(7)'])
        try:
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, 'learner 1.*exit_code=7.*learner_1/trainer.log'):
                wait_for_learner_shutdown({0:slow, 1:failed}, started+5, Path('/tmp/test-run'))
            self.assertLess(time.monotonic()-started, 3)
        finally:
            for process in (slow, failed):
                if process.poll() is None:
                    process.kill()
                process.wait()

    def test_hung_teardown_uses_shared_deadline_and_identifies_process(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
        try:
            started = time.monotonic()
            with self.assertRaisesRegex(TimeoutError, 'learner shutdown exceeded --training-timeout.*learner 3 pid=.*learner_3/trainer.log'):
                wait_for_learner_shutdown({3:process}, started+.3, Path('/tmp/test-run'))
            self.assertLess(time.monotonic()-started, 2)
        finally:
            process.kill()
            process.wait()

    def test_empty_survivor_set_and_already_completed_process(self):
        wait_for_learner_shutdown({}, time.monotonic()-1, '/tmp/test-run')
        process = subprocess.Popen([sys.executable, '-c', 'pass'])
        process.wait()
        wait_for_learner_shutdown({0:process}, time.monotonic()-1, '/tmp/test-run')

    def test_delayed_exit_after_stopped_ack_exports_success_for_both_methods(self):
        for method in ('decoupled_heloco', 'decoupled_diloco'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                config = ExperimentConfig(method, decoupled=DecoupledConfig(num_fragments=2,
                    min_quorum=2, overlap_steps=2, max_inflight_captures=2, syncer_shards=2,
                    scheduler='paper_offsets', sync_period=2, sync_interval=.1, grace_window_factor=.2),
                    monitoring={'enabled':True})
                options = SimpleNamespace(islands=2, steps=12, outer_lr=.7, outer_momentum=.9,
                    ps_timeout=30., island_slowness_factors=[1,2])
                folder = Path(directory); output = io.StringIO()
                with redirect_stdout(output), redirect_stderr(output):
                    status = _run_training(config, options, _TinyTokenModel(2), ['0','1'], folder, 30., ROOT,
                        worker_command=[sys.executable, str(Path(__file__).with_name('cpu_training_worker.py')),
                                        '--exit-delay-seconds', '6'])
                self.assertEqual(status, 0, output.getvalue())
                summary = json.loads((folder/'summary.json').read_text())
                self.assertEqual(summary['status'], 'passed')
                self.assertEqual([l['total_local_steps'] for l in summary['learners']], [12,12])
                self.assertTrue((folder/'global_model.pt').exists())
                self.assertTrue((folder/'trajectory'/'00000000.pt').exists())
                self.assertEqual(len(list((folder/'trajectory').glob('*-final.pt'))), 1)


if __name__ == '__main__':
    unittest.main()
