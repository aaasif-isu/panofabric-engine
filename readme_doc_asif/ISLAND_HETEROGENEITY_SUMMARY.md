# Island Heterogeneity Simulation - Implementation Summary

## ✅ Feature complete, tested, and CORRECTED

Simulate heterogeneous (different-speed) islands via `heloco.yaml` only --
**no CLI flag**. Fixed an earlier broken attempt that tried to pass a
non-existent `--fault_tolerance.slowness_factor` torchrun argument (crashed
every island with "Unrecognized options").

---

## The Bug That Was Fixed

The first implementation added `f"--fault_tolerance.slowness_factor={x}"` to
each island's `torchrun -m torchtitan.train ...` command. That flag does not
exist: torchtitan's `FaultTolerance` config dataclass
(`torchtitan/experiments/torchft/config/job_config.py`) only defines
`sync_steps`, `should_quantize`, `fragment_sync_delay`,
`fragment_update_alpha`, `module_fqns_per_model_fragment`, `num_fragments`.
Passing an unknown field crashes the tyro-based CLI parser immediately:

```
Unrecognized options: --fault_tolerance.slowness_factor=1.0
```

---

## The Fix

Use an environment variable instead of a CLI flag -- the same pattern already
used by this codebase for `ISLAND_LANGUAGE` (non-IID data) and `PF_WIRE_BF16`
(wire compression), both of which travel through env because there's no
corresponding torchtitan config field for them either.

1. **`heloco.yaml`** -- `island_slowness_factors: [1, 5, 3, 14]` (or `null`
   for the default, all islands at speed 1.0).
2. **`run_heloco.py`** -- `trainer_env()` exports
   `PF_ISLAND_SLOWNESS_FACTOR=<factor>` into each island's subprocess env.
   `trainer_cmd()` no longer emits the bogus flag.
3. **`panoengine/decentralized/async_diloco.py`** -- `AsyncDiLoCo.__init__`
   reads `$PF_ISLAND_SLOWNESS_FACTOR` (default `1.0`, falls back to `1.0` on
   invalid/negative values). `_step_post_hook` (runs after every inner
   optimizer step) sleeps `step_seconds * (factor - 1.0)` extra, where
   `step_seconds` is the wall-clock time of the step that just completed.

---

## Files Changed

| File | Change |
|------|--------|
| `heloco.yaml` | `island_slowness_factors` key + corrected comment |
| `run_heloco.py` | Removed bogus `--fault_tolerance.slowness_factor` flag from `trainer_cmd()`; added `PF_ISLAND_SLOWNESS_FACTOR` env export in `trainer_env()`; kept a hidden (`argparse.SUPPRESS`) `--island-slowness-factors` arg purely so the yaml-config loader recognizes the key -- it is not a supported/documented CLI flag |
| `panoengine/decentralized/async_diloco.py` | `AsyncDiLoCo.__init__` reads the env var into `self._slowness_factor`; `_step_post_hook` applies the proportional sleep |

---

## Usage

```yaml
# heloco.yaml
islands: 4
island_slowness_factors: [1, 5, 3, 14]
```

```bash
python run_heloco.py
```

No CLI arguments needed.

---

## Verification Performed

1. **Syntax check** both modified files (`ast.parse`) -- OK.
2. **Unit test** (`AsyncDiLoCo.__init__`): confirmed `_slowness_factor`
   correctly reads `1.0` (default/no env), `5.0` (env set), and `1.0`
   (invalid env value falls back safely).
3. **Unit test** (`_step_post_hook` sleep behavior): with a simulated
   50ms step, factor `1.0` adds ~0s extra sleep; factor `5.0` adds ~0.2s
   extra sleep (4 x 50ms), i.e. ~5x the per-step wall time as intended.
4. **Regression check**: ran the existing `async_diloco_test.py` /
   `heloco_test.py` suites before and after the change (`git stash`
   comparison) -- the same 7-8 pre-existing bf16-precision-tolerance
   failures occur on both, confirming this change introduces no new
   failures.
5. **`--dry-run` check**: confirmed the generated `torchrun` command no
   longer contains `--fault_tolerance.slowness_factor` (the crash is gone),
   and `trainer_env()` sets `PF_ISLAND_SLOWNESS_FACTOR` correctly per island.

---

## Expected Behavior by Method

| Method | Robustness to stragglers | Why |
|--------|---------------------------|-----|
| **HeLoCo** | Best | No barrier; async push/pull, so a slow island never blocks others |
| **DiLoCo** | Worst | Sync barrier; waits for the slowest island every window |
| **MLA** | Medium | Momentum accumulation smooths out slow islands somewhat |

---

## FAQ

**Q: Is there a `--island-slowness-factors` CLI flag?**
A: No supported one. `heloco.yaml` is the only interface.

**Q: What's the default?**
A: `null` → `[1.0, 1.0, ...]` for all islands (no artificial delay).

**Q: Where exactly is the delay applied?**
A: `panoengine/decentralized/async_diloco.py`, in
`AsyncDiLoCo._step_post_hook`, via `time.sleep()`.
