"""Actual spawned learners, TCP synchronization, and deadline cleanup."""

import ast
from contextlib import redirect_stderr, redirect_stdout
import io
import multiprocessing
import os
import re
import unittest
from unittest.mock import patch

from panoengine.decentralized.decoupled_heloco.config import DecoupledConfig, ExperimentConfig
from panoengine.decentralized.decoupled_heloco.process_smoke import run_process_smoke


def _exit_during_startup(*_):
    os._exit(7)


class ProcessSmokeTests(unittest.TestCase):
    def config(self):
        return ExperimentConfig(method="decoupled_heloco", run={"seed": 42, "outer_lr": 0.7}, decoupled=DecoupledConfig(num_fragments=2, min_quorum=2, overlap_steps=2, sync_interval=0.1, grace_window_factor=0.25))

    def assert_no_demo_children(self):
        self.assertFalse([p for p in multiprocessing.active_children() if p.name.startswith("decoupled-learner-")])

    def test_separate_processes_transfer_each_fragment_and_exit_cleanly(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = run_process_smoke(self.config(), learners=2, cycles=2, timeout=30.0)
        self.assertEqual(status, 0)
        text = output.getvalue()
        self.assertIn("PROCESS SMOKE TEST PASSED", text)
        self.assertIn("Final fragment revisions: [2, 2]", text)
        pids = ast.literal_eval(re.search(r"learner_pids=(\[[^\]]+\])", text)[1])
        self.assertEqual(len(set(pids + [os.getpid()])), 3)
        counts = re.search(r"syncer_sent=(\d+), syncer_received=(\d+)", text)
        self.assertGreater(int(counts[1]), 0)
        self.assertGreater(int(counts[2]), 0)
        self.assert_no_demo_children()

    def test_deadline_failure_cleans_up_spawned_children(self):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            self.assertEqual(run_process_smoke(self.config(), timeout=0.001), 2)
        self.assertIn("PROCESS SMOKE TEST FAILED", output.getvalue())
        self.assert_no_demo_children()

    def test_invalid_process_controls_fail_before_launch(self):
        for controls in ({"learners": 1}, {"cycles": 0}, {"timeout": float("nan")}, {"learners": True}):
            with self.subTest(controls=controls):
                with self.assertRaises(ValueError):
                    run_process_smoke(self.config(), **controls)
        self.assert_no_demo_children()

    def test_child_startup_failure_stops_the_run_and_cleans_up_siblings(self):
        output = io.StringIO()
        with patch("panoengine.decentralized.decoupled_heloco.process_smoke._learner_process", _exit_during_startup):
            with redirect_stdout(output), redirect_stderr(output):
                self.assertEqual(run_process_smoke(self.config(), timeout=15.0), 2)
        self.assertIn("a learner process exited during startup", output.getvalue())
        self.assert_no_demo_children()


if __name__ == "__main__":
    unittest.main()
