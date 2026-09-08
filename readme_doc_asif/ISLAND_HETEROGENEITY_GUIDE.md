# Island Heterogeneity Simulation Guide

## Overview

Configure **island slowness factors in `heloco.yaml`** to simulate heterogeneous
(different-speed) islands. Test how HeLoCo, DiLoCo, and MLA handle stragglers
and slow workers. **There is no CLI flag for this -- it is heloco.yaml only.**

---

## Slowness Factor Concept

```
Slowness Factor = per-step delay multiplier

1.0 = Normal speed (baseline, no artificial delay)
5.0 = ~5x slower (adds sleep proportional to 4x the real step time)
```

### Example: 4 Islands with [1, 5, 3, 14]

```
Island 0: factor 1.0  -> Normal (baseline)
Island 1: factor 5.0  -> ~5x slower (mild straggler)
Island 2: factor 3.0  -> ~3x slower (moderate straggler)
Island 3: factor 14.0 -> ~14x slower (severe straggler)
```

---

## Usage (heloco.yaml ONLY)

```yaml
islands: 4
island_slowness_factors: [1, 5, 3, 14]
```

Then run:
```bash
python run_heloco.py
```

No command-line arguments. Everything is configured in `heloco.yaml`.

---

## How It's Actually Implemented

**Important:** there is no `--fault_tolerance.slowness_factor` CLI flag.
torchtitan's `fault_tolerance` config dataclass has no such field, and an
earlier attempt to pass it as a torchrun argument crashed every island with
"Unrecognized options: --fault_tolerance.slowness_factor=...". The real
mechanism is an environment variable, the same pattern already used for
`ISLAND_LANGUAGE` (non-IID data) and `PF_WIRE_BF16`:

1. `run_heloco.py`'s `trainer_env()` reads `island_slowness_factors` (parsed
   from `heloco.yaml`) and exports `PF_ISLAND_SLOWNESS_FACTOR=<factor>` into
   each island's subprocess environment -- not a torchrun/torchtitan argument.
2. `panoengine.decentralized.async_diloco.AsyncDiLoCo` (the HeLoCo/DiLoCo
   worker class we own) reads `$PF_ISLAND_SLOWNESS_FACTOR` in `__init__`
   (defaults to `1.0` if unset or invalid).
3. Its `_step_post_hook` -- called after every inner optimizer step --
   sleeps an extra `step_seconds * (factor - 1.0)`, where `step_seconds` is
   how long the real step just took. Factor `5.0` makes that island take
   ~5x as long per step as factor `1.0`, whatever the real step time is.

---

## Validation Rules

✅ **Valid**
- Exactly `islands` number of factors in the list
- All factors positive (> 0)
- Fractional allowed (0.5, 2.5, etc.)

❌ **Invalid**
- Wrong count → `SystemExit` with error
- Negative or zero → `SystemExit` with error

```yaml
# ❌ ERROR: Only 3 factors for 4 islands
islands: 4
island_slowness_factors: [1, 5, 3]

# ❌ ERROR: Negative factor
islands: 2
island_slowness_factors: [1, -5]

# ✅ OK
islands: 4
island_slowness_factors: [1, 5, 3, 14]
```

---

## Real-World Scenarios

```yaml
# Mild heterogeneity (2 fast, 2 slow)
islands: 4
island_slowness_factors: [1, 1, 5, 5]

# One severe straggler
islands: 4
island_slowness_factors: [1, 1, 1, 20]

# Gradual degradation
islands: 5
island_slowness_factors: [1, 2, 3, 4, 5]

# No heterogeneity (baseline)
island_slowness_factors: null   # or omit the key entirely
```

---

## Expected Behavior by Method

| Method | Robustness | Mechanism |
|--------|-----------|-----------|
| **HeLoCo** | Best | No barrier; async push/pull, so a slow island never blocks others |
| **DiLoCo** | Worst | Sync barrier; waits for the slowest island at each window |
| **MLA** | Medium | Momentum accumulation helps smooth out slow islands |

---

## FAQ

**Q: Is there a `--island-slowness-factors` CLI flag?**
A: No. This is heloco.yaml-only by design.

**Q: What's the default slowness factor?**
A: 1.0 for all islands (no heterogeneity, no artificial delay).

**Q: Where is the delay actually applied?**
A: `panoengine/decentralized/async_diloco.py`, `AsyncDiLoCo._step_post_hook`,
via `time.sleep()` proportional to `(factor - 1.0)` on every inner step.

