"""Concurrent captures with ordered commits and bounded retained snapshots."""
import copy
import csv
import io
import json
from concurrent.futures import Future
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import torch

from panoengine.decentralized.decoupled_heloco.config import DecoupledConfig, ExperimentConfig, ConfigError
from panoengine.decentralized.decoupled_heloco.learner import DecoupledLearner, SnapshotQueueFull
from panoengine.decentralized.decoupled_heloco.syncer import DecoupledSyncer
from panoengine.decentralized.decoupled_heloco.timing import TimedSyncController
from panoengine.decentralized.decoupled_heloco.gpu_training import _run_training
from panoengine.decentralized.decoupled_heloco.smoke import _TinyTokenModel

ROOT = Path(__file__).resolve().parents[2]


class HeldLearner(DecoupledLearner):
    """Real boundary snapshots; hold delivery of fragment 0 to the syncer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.held = True
        self.pairs = {}
        self.hide_first_fragment = False
        self.hide_fragment = None

    def metadata(self):
        meta = super().metadata()
        hidden = 0 if self.hide_first_fragment else self.hide_fragment
        if hidden is not None:
            fragments = list(meta.fragments)
            fragments[hidden] = replace(fragments[hidden], local_steps=0)
            meta = replace(meta, fragments=tuple(fragments))
        return meta

    def request_snapshot(self, request_id, fragment_id, expected_revision, **kwargs):
        inner = super().request_snapshot(request_id, fragment_id, expected_revision, **kwargs)
        if fragment_id != 0:
            return inner
        if request_id in self.pairs:
            return self.pairs[request_id][1]
        outer = Future()
        self.pairs[request_id] = (inner, outer)
        def forward(future):
            if not self.held and not outer.done():
                try:
                    outer.set_result(future.result())
                except BaseException as exc:
                    outer.set_exception(exc)
        inner.add_done_callback(forward)
        return outer

    def release_snapshot(self, request_id):
        pair = self.pairs.pop(request_id, None)
        if pair is not None and not pair[1].done():
            pair[1].cancel()
        return super().release_snapshot(request_id)

    def unblock(self):
        self.held = False
        for inner, outer in self.pairs.values():
            if inner.done() and not outer.done():
                outer.set_result(inner.result())


class PipelineTests(unittest.TestCase):
    def make(self, count=2, method='sgd'):
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1)
        self.initial = copy.deepcopy(model)
        self.models = [copy.deepcopy(model) for _ in range(count)]
        self.learners = [HeldLearner(m, 2, learner_id=i, max_snapshot_requests=2) for i, m in enumerate(self.models)]
        for learner, local in zip(self.learners, self.models):
            with learner.training_step(8), torch.no_grad():
                for p in local.parameters():
                    p.sub_(.2)
        self.syncer = DecoupledSyncer(model, self.learners, 2, min_quorum=2, overlap_steps=2,
            outer_method=method, outer_lr=.5, scheduler='paper_offsets', sync_period=2,
            fragment_offsets=[1, 0], max_inflight_captures=2)
        self.now = 0.
        self.controller = TimedSyncController(self.syncer, sync_interval=1., grace_window_factor=0., clock=lambda: self.now)
        self.controller._observe = lambda now: None
        self.controller.step_time_ema = 1.

    def tick(self, now):
        self.now = now
        return self.controller.tick()

    def boundaries(self):
        for learner in self.learners:
            learner.boundary()

    def test_ready_younger_capture_waits_for_head_without_advancing_clock(self):
        self.make()
        self.tick(1); self.boundaries()
        self.tick(2); self.boundaries()
        self.assertEqual(self.syncer.peak_inflight_captures, 2)
        self.assertEqual(len(self.learners[0]._requests), 2)
        self.assertTrue(self.syncer.capture_quorum_ready(1))
        self.assertFalse(self.syncer.capture_quorum_ready(0))
        self.assertIsNone(self.tick(2.1))
        self.assertEqual(self.syncer.global_step, 0)
        self.assertEqual(self.syncer.fragment_revisions, (0, 0))
        for learner in self.learners:
            learner.unblock()
        first = self.tick(2.2)
        second = self.tick(2.3)
        self.assertEqual((first.global_step, second.global_step), (1, 2))
        self.assertEqual((first.fragment_id, second.fragment_id), (0, 1))
        self.assertEqual(self.syncer.fragment_revisions, (1, 1))
        self.boundaries()
        self.assertEqual([l.metadata().syncer_step for l in self.learners], [2, 2])
        for value in self.syncer.optimizer.model_snapshot().values():
            torch.testing.assert_close(value, torch.full_like(value, .9))
        self.assertTrue(all(not learner._requests for learner in self.learners))

    def test_timeout_retries_same_slot_and_keeps_younger_snapshot(self):
        self.make(method='heloco')
        self.tick(1); self.boundaries(); self.tick(2); self.boundaries(); self.tick(2.1)
        younger = self.syncer.capture_attempts[1]
        saved = younger.futures[0].result()
        self.tick(3.1)
        self.assertEqual(self.controller.timeouts, 1)
        self.assertEqual(self.syncer.global_step, 0)
        self.assertFalse(self.syncer.capture_attempts[0].futures)
        self.assertIs(self.syncer.capture_attempts[1], younger)
        self.assertIs(younger.futures[0].result(), saved)
        self.tick(4.2); self.boundaries()
        for learner in self.learners:
            learner.unblock()
        first = self.tick(4.3); second = self.tick(4.4)
        self.assertEqual((first.global_step, second.global_step), (1, 2))
        self.assertEqual(self.syncer.fragment_revisions, (1, 1))
        from panoengine.decentralized.decoupled_heloco.optimizer import FragmentHeLoCo
        reference = FragmentHeLoCo(self.syncer.manager, dict(self.initial.named_parameters()), lr=.5, momentum=.9)
        for fragment in (0, 1):
            gradients = {name: torch.full_like(dict(self.initial.named_parameters())[name], .2)
                         for name in self.syncer.manager.fragment(fragment).parameter_names}
            reference.step(fragment, gradients)
            for name, value in reference.momentum_snapshot(fragment).items():
                torch.testing.assert_close(self.syncer.optimizer.momentum_snapshot(fragment)[name], value)
        for name, value in reference.model_snapshot().items():
            torch.testing.assert_close(self.syncer.optimizer.model_snapshot()[name], value)

    def test_late_contributor_gets_fresh_request_id_after_younger_capture(self):
        self.make(count=3)
        self.learners[2].hide_first_fragment = True
        self.tick(1); self.boundaries(); self.tick(2); self.boundaries()
        head, younger = self.syncer.capture_attempts
        self.assertNotIn(2, head.futures)
        self.assertIn(2, younger.futures)
        self.learners[2].hide_first_fragment = False
        self.assertEqual(self.syncer.extend_sync(0), (2,))
        self.assertGreater(int(head.request_ids[2]), int(younger.request_ids[2]))
        self.boundaries()
        for learner in self.learners:
            learner.unblock()
        first = self.tick(2.1)
        self.assertEqual(first.learner_ids, (0, 1, 2))
        self.assertIsNotNone(self.tick(2.2))

    def test_distinct_fragments_and_capacity_are_enforced_and_cancel_releases_all(self):
        self.make()
        first = self.syncer.begin_sync(); second = self.syncer.begin_sync()
        self.assertEqual((first.fragment_id, second.fragment_id), (0, 1))
        self.assertIsNone(self.syncer.begin_sync())
        with self.assertRaises(SnapshotQueueFull):
            self.learners[0].request_snapshot('100', 0, 0)
        with self.assertRaises(ValueError):
            self.syncer.cancel_sync(1)
        self.assertTrue(self.syncer.cancel_sync())
        self.assertFalse(self.syncer.has_active_sync)
        self.assertTrue(all(not learner._requests for learner in self.learners))
        self.assertEqual(self.syncer.scheduler.global_step, 1)
        self.assertEqual(self.syncer.begin_sync().global_step, 1)

    def test_reconnect_discards_old_captures_from_every_reserved_slot(self):
        self.make()
        self.tick(1); self.boundaries(); self.tick(2); self.boundaries()
        old_future = self.syncer.capture_attempts[1].futures[1]
        replacement_model = copy.deepcopy(self.initial)
        replacement = HeldLearner(replacement_model, 2, learner_id=1, max_snapshot_requests=2)
        with replacement.training_step(8), torch.no_grad():
            for parameter in replacement_model.parameters():
                parameter.sub_(.4)
        self.syncer.reconnect_learner(replacement)
        self.assertTrue(old_future.done())  # Old completed data remains immutable, but is excluded.
        self.assertTrue(all(1 not in attempt.futures for attempt in self.syncer.capture_attempts))
        self.syncer.extend_sync(0); self.syncer.extend_sync(1)
        replacement.boundary(); replacement.unblock(); self.learners[0].unblock()
        self.assertIsNotNone(self.tick(2.1)); self.assertIsNotNone(self.tick(2.2))
        for value in self.syncer.optimizer.model_snapshot().values():
            torch.testing.assert_close(value, torch.full_like(value, .85))

    def test_pipeline_local_budget_drains_after_real_learner_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig('decoupled_heloco', decoupled=DecoupledConfig(
                num_fragments=2, min_quorum=2, overlap_steps=20, max_inflight_captures=2,
                sync_interval=.03, grace_window_factor=0., scheduler='paper_offsets',
                sync_period=2, fragment_offsets=[1, 0]))
            options = SimpleNamespace(islands=3, steps=18, outer_lr=.7, outer_momentum=.9,
                ps_timeout=20., island_slowness_factors=[1, 1, 1])
            folder = Path(directory); output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                status = _run_training(config, options, _TinyTokenModel(2), ['0', '1', '2'], folder, 30., ROOT,
                    worker_command=[sys.executable, str(Path(__file__).with_name('cpu_training_worker.py')),
                                    '--capture-delay-seconds', '.35', '--fail-learner', '2'])
            logs = '\n'.join(path.read_text() for path in folder.glob('learner_*/trainer.log'))
            self.assertEqual(status, 0, output.getvalue() + logs)
            summary = json.loads((folder/'summary.json').read_text())
            self.assertTrue(summary['degraded'])
            self.assertEqual(summary['completed_learner_ids'], [0, 1])
            self.assertEqual([row['total_local_steps'] for row in summary['learners'][:2]], [18, 18])
            self.assertEqual(summary['peak_inflight_captures'], 2)
            self.assertTrue(all(revision >= 1 for revision in summary['fragment_revisions']))

    def test_younger_quorum_is_frozen_when_its_own_grace_expires(self):
        self.make(count=3)
        self.learners[2].hide_fragment = 1
        self.tick(1); self.boundaries(); self.tick(2); self.boundaries(); self.tick(2.1)
        younger = self.syncer.capture_attempts[1]
        self.assertTrue(younger.sealed)
        self.assertEqual(younger.plan.learner_ids, (0, 1))
        self.learners[2].hide_fragment = None
        self.assertEqual(self.syncer.extend_sync(1), ())
        for learner in self.learners:
            learner.unblock()
        self.assertEqual(self.tick(2.2).learner_ids, (0, 1, 2))
        self.assertEqual(self.tick(2.3).learner_ids, (0, 1))

    def test_config_rejects_unbounded_or_non_integer_capture_limits(self):
        for value in (0, -1, True, 1.5, 3):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                DecoupledConfig.from_mapping({'num_fragments': 2, 'max_inflight_captures': value})
        self.assertEqual(DecoupledConfig.from_mapping({'num_fragments': 2, 'max_inflight_captures': 2}).max_inflight_captures, 2)

    def test_pipeline_drains_clock_budget_over_tcp_for_both_methods(self):
        for method in ('decoupled_diloco', 'decoupled_heloco'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                config = ExperimentConfig(method, decoupled=DecoupledConfig(
                    num_fragments=2, min_quorum=2, overlap_steps=20, max_inflight_captures=2,
                    sync_interval=.03, grace_window_factor=1., adaptive_grace=False,
                    scheduler='paper_offsets', sync_period=5, fragment_offsets=[1, 3],
                    stopping='syncer_steps', syncer_steps=8), monitoring={'enabled': True})
                options = SimpleNamespace(islands=2, steps=1, outer_lr=.7, outer_momentum=.9,
                    ps_timeout=20., island_slowness_factors=[1, 2])
                folder = Path(directory); output = io.StringIO()
                with redirect_stdout(output), redirect_stderr(output):
                    status = _run_training(config, options, _TinyTokenModel(2), ['0', '1'], folder, 30., ROOT,
                        worker_command=[sys.executable, str(Path(__file__).with_name('cpu_training_worker.py')),
                                        '--capture-delay-seconds', '.35'])
                logs = '\n'.join(p.read_text() for p in folder.glob('learner_*/trainer.log'))
                self.assertEqual(status, 0, output.getvalue() + logs)
                summary = json.loads((folder/'summary.json').read_text())
                self.assertEqual(summary['syncer_step'], 8)
                self.assertTrue(all(row['syncer_step'] == 8 for row in summary['learners']))
                self.assertEqual(summary['peak_inflight_captures'], 2)
                with (folder/'syncs.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([int(row['global_step']) for row in rows], [1, 3, 6, 8])
                with (folder/'capture_events.csv').open() as stream:
                    events = list(csv.DictReader(stream))
                self.assertEqual(sum(row['event'] == 'commit' for row in events), 4)
                self.assertTrue(all(int(row['global_step']) <= 8 for row in events))

