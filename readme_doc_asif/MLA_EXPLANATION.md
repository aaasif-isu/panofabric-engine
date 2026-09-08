# MLA — Momentum Look-Ahead Explained

## What is MLA?

**MLA = Momentum Look-Ahead** is a simple **outer optimizer** for decentralized/async training.

It performs a **momentum accumulation on the server side**, with no direction correction.

## The Algorithm

MLA implements a single server-side update rule:

```
m      ← γ·m + (1−γ)·Δ          (momentum accumulation)
θ      ← θ − lr·(γ·m_new + Δ)   (parameter update with lookahead term)
```

Where:
- `Δ` = pseudo-gradient from worker = `θ_start − θ_final`
- `m` = momentum buffer (initialized to zero)
- `γ` = momentum coefficient (typically 0.9)
- `lr` = outer learning rate (typically 0.7)
- `m_new` = updated momentum

## Key Points

✅ **Simple:** Only 2 hyperparameters (`lr`, `momentum`)
✅ **Server-side only:** Applied when aggregating pseudo-gradients
✅ **No direction correction:** Plain momentum accumulation
✅ **No worker look-ahead dispatch:** Worker initialization at `θ − lr·γ·m` is **HeLoCo**, not MLA

## How to Use

```bash
# Run MLA method
python run_heloco.py --methods mla --outer-lr 0.7 --outer-momentum 0.9

# MLA with async coordination
python run_heloco.py --methods mla --coordination-method async

# Compare all three methods
python run_heloco.py --methods heloco,diloco,mla
```

## Comparison: HeLoCo vs DiLoCo vs MLA

| Aspect | HeLoCo | DiLoCo | MLA |
|--------|--------|--------|-----|
| **Outer Optimizer** | HeLoCoOptimizer | DiLoCoOptimizer | MLAOptimizer |
| **Momentum** | ✓ (with lookahead) | ✓ (basic) | ✓ (basic) |
| **Block Correction** | ✓ (direction-aware) | ✗ | ✗ |
| **Worker Dispatch** | ✓ (look-ahead init) | ✗ | ✗ |
| **Complexity** | High | Medium | Low |
| **Use Case** | Heterogeneous envs | Baseline async | Simple baseline |

## Implementation Details

**File:** `panoengine/decentralized/mla.py`

```python
class MLAOptimizer(optim.Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.9):
        # Just lr and momentum — that's it!
        
    def step(self, closure=None):
        # For each parameter p with gradient Δ:
        for p in params:
            delta = p.grad
            m = momentum_buffer[p]  # accumulate here
            
            m ← γ·m + (1−γ)·Δ
            θ ← θ − lr·(γ·m_new + Δ)
```

## Important Note: Worker Look-Ahead

Some papers discuss "look-ahead" worker initialization where workers start from `θ − lr·γ·m` instead of `θ`. This is **NOT part of MLA**—it's part of the **HeLoCo** algorithm.

- **MLA alone:** Just server momentum, standard worker init at `θ`
- **HeLoCo:** Includes MLA + worker look-ahead dispatch + block correction

## References

- **MomentumLookAhead:** From `heloco_stable_v2.ipynb` (notebook Cell ~8)
- **SmartDCMLA_v2:** A more advanced direction-corrected variant (NOT implemented in this version)

---

*See ASYNC_QUICK_SUMMARY.md for how MLA fits into the async training pipeline.*
