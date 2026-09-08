#!/usr/bin/env python3
"""Test the ACTUAL island-slowness-simulation mechanism.

There is no --fault_tolerance.slowness_factor torchtitan flag (an earlier
attempt tried that and crashed every island with "Unrecognized options").
The real mechanism:
  1. heloco.yaml's island_slowness_factors is parsed by run_heloco.py
  2. trainer_env() exports PF_ISLAND_SLOWNESS_FACTOR per island (env var,
     same pattern as ISLAND_LANGUAGE / PF_WIRE_BF16)
  3. panoengine.decentralized.async_diloco.AsyncDiLoCo reads that env var
     and sleeps proportionally in _step_post_hook after every inner step.

Run with the project's venv (needs torch): heloco/bin/python test_island_slowness.py
"""
import os
import sys
import time
from unittest import mock

import torch.nn as nn
import torch.optim as optim

from panoengine.decentralized.async_diloco import AsyncDiLoCo


def _make_model() -> nn.Module:
    return nn.Linear(4, 4)


def test_default_factor_is_one() -> None:
    model = _make_model()
    w = AsyncDiLoCo(
        "http://127.0.0.1:9/sync", model,
        optim.SGD(model.parameters(), lr=0.1), sync_every=1000,
    )
    assert w._slowness_factor == 1.0, w._slowness_factor
    print("PASS: default factor (no env) == 1.0")


def test_env_sets_factor() -> None:
    with mock.patch.dict(os.environ, {"PF_ISLAND_SLOWNESS_FACTOR": "5.0"}):
        model = _make_model()
        w = AsyncDiLoCo(
            "http://127.0.0.1:9/sync", model,
            optim.SGD(model.parameters(), lr=0.1), sync_every=1000,
        )
    assert w._slowness_factor == 5.0, w._slowness_factor
    print("PASS: env PF_ISLAND_SLOWNESS_FACTOR=5.0 -> factor == 5.0")


def test_invalid_env_falls_back_to_one() -> None:
    with mock.patch.dict(os.environ, {"PF_ISLAND_SLOWNESS_FACTOR": "garbage"}):
        model = _make_model()
        w = AsyncDiLoCo(
            "http://127.0.0.1:9/sync", model,
            optim.SGD(model.parameters(), lr=0.1), sync_every=1000,
        )
    assert w._slowness_factor == 1.0, w._slowness_factor
    print("PASS: invalid env value falls back to 1.0")


def test_negative_env_falls_back_to_one() -> None:
    with mock.patch.dict(os.environ, {"PF_ISLAND_SLOWNESS_FACTOR": "-3.0"}):
        model = _make_model()
        w = AsyncDiLoCo(
            "http://127.0.0.1:9/sync", model,
            optim.SGD(model.parameters(), lr=0.1), sync_every=1000,
        )
    assert w._slowness_factor == 1.0, w._slowness_factor
    print("PASS: negative env value falls back to 1.0")


def test_first_hook_call_never_crashes() -> None:
    """Regression test for a real crash observed in production:
    ValueError: sleep length must be non-negative, raised from
    _step_post_hook on the VERY FIRST optimizer.step() of a run with
    factor > 1.0. Root cause: time.monotonic() - getattr(self, "_x",
    time.monotonic()) evaluates the left operand before the getattr's
    default, so the default is a LATER (larger) timestamp than the left
    term, going negative on the very first (uninitialized) call. Fixed
    by initializing _last_step_end eagerly in __init__ instead of via a
    lazy getattr default. This test calls the hook immediately after
    construction, with no manual timestamp priming, exactly like the
    real first torch optimizer.step() in a running island."""
    with mock.patch.dict(os.environ, {"PF_ISLAND_SLOWNESS_FACTOR": "5.0"}):
        model = _make_model()
        w = AsyncDiLoCo(
            "http://127.0.0.1:9/sync", model,
            optim.SGD(model.parameters(), lr=0.1), sync_every=1000,
        )
    # No manual _last_step_end priming here -- this is the exact sequence
    # that crashed in production (first hook call right after __init__).
    w._step_post_hook(None, (), {})  # must not raise
    print("PASS: first _step_post_hook call right after __init__ does not crash")


def test_step_post_hook_sleeps_proportionally() -> None:
    # factor 1.0 -> negligible extra sleep
    model = _make_model()
    w = AsyncDiLoCo(
        "http://127.0.0.1:9/sync", model,
        optim.SGD(model.parameters(), lr=0.1), sync_every=1000,
    )
    w._last_step_end = time.monotonic() - 0.05
    start = time.monotonic()
    w._step_post_hook(None, (), {})
    elapsed = time.monotonic() - start
    assert elapsed < 0.02, elapsed
    print(f"PASS: factor=1.0 adds ~{elapsed:.3f}s extra sleep (expected ~0)")

    # factor 5.0 -> ~4x the simulated 50ms step = ~0.2s extra sleep
    with mock.patch.dict(os.environ, {"PF_ISLAND_SLOWNESS_FACTOR": "5.0"}):
        model2 = _make_model()
        w2 = AsyncDiLoCo(
            "http://127.0.0.1:9/sync", model2,
            optim.SGD(model2.parameters(), lr=0.1), sync_every=1000,
        )
    w2._last_step_end = time.monotonic() - 0.05
    start = time.monotonic()
    w2._step_post_hook(None, (), {})
    elapsed = time.monotonic() - start
    assert 0.15 < elapsed < 0.30, elapsed
    print(f"PASS: factor=5.0 adds ~{elapsed:.3f}s extra sleep (expected ~0.2s)")


def test_trainer_env_exports_per_island_factor() -> None:
    """run_heloco.py's trainer_env() must export PF_ISLAND_SLOWNESS_FACTOR,
    NOT any --fault_tolerance.* CLI flag (torchtitan has no such flag)."""
    sys.argv = ["run_heloco.py", "--dry-run"]
    from run_heloco import parse_args, trainer_cmd, trainer_env

    args = parse_args()
    args.island_slowness_factors = [1.0, 5.0]
    args.islands = 2

    env0 = trainer_env(args, 0, [0, 1, 2, 3], "http://x/sync", "http://x/hb")
    env1 = trainer_env(args, 1, [0, 1, 2, 3], "http://x/sync", "http://x/hb")
    assert env0.get("PF_ISLAND_SLOWNESS_FACTOR") == "1.0", env0.get("PF_ISLAND_SLOWNESS_FACTOR")
    assert env1.get("PF_ISLAND_SLOWNESS_FACTOR") == "5.0", env1.get("PF_ISLAND_SLOWNESS_FACTOR")

    cmd0 = trainer_cmd(args, 0)
    assert not any("slowness_factor" in c for c in cmd0), (
        "trainer_cmd must NOT emit a --fault_tolerance.slowness_factor flag; "
        "torchtitan has no such config field and this crashes every island."
    )
    print("PASS: trainer_env exports PF_ISLAND_SLOWNESS_FACTOR; "
          "trainer_cmd emits no bogus torchtitan flag")


if __name__ == "__main__":
    test_default_factor_is_one()
    test_env_sets_factor()
    test_invalid_env_falls_back_to_one()
    test_negative_env_falls_back_to_one()
    test_first_hook_call_never_crashes()
    test_step_post_hook_sleeps_proportionally()
    test_trainer_env_exports_per_island_factor()
    print("\nAll island-slowness tests passed.")
