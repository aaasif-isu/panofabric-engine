# HeLoCo Parameters Guide - Part 3: Timeline & Summary

## Practical Timeline: One Full Window Cycle

### **Steps 1 to 20 (sync_steps=20)**
```
Island 1                        Island 2
├─ Train step 1                 ├─ Train step 1
├─ Train step 2                 ├─ Train step 2
├─ ...                          ├─ ...
└─ Train step 20                └─ Train step 20
    ↓                               ↓
    Compute Δ₁ = θ₁ - θ_initial     Compute Δ₂ = θ₂ - θ_initial
    (pseudo-gradient)               (pseudo-gradient)
```

### **Window Boundary (step 20) - CORRECTION HAPPENS HERE**
```
Server:
  ├─ Receive Δ₁ from Island 1
  ├─ Apply block_correct(Δ₁, m, c_ok=0.2, k_s=0.5, k_d=1.0, ...)
  │   ├─ For each tensor block b in Δ₁:
  │   │   ├─ Compute cos_b (alignment with momentum)
  │   │   ├─ Compute conf_b (confidence)
  │   │   └─ IF cos_b ≥ 0.2: PASS
  │   │      ELIF cos_b < 0: SHRINK
  │   │      ELSE: ROTATE
  │   └─ Apply ρ scaling
  │
  ├─ Commit step: m_new = 0.9·m_old + 0.1·Δ₁_corrected
  ├─ Update: θ_new = θ_old - 0.7·(Δ₁_corrected + 0.9·m_new)
  │
  ├─ Compute look-ahead: θ̄ = θ_new - 0.7·0.9·m_new
  └─ Send θ̄ back to Island 1
```

### **Steps 21 to 40 (repeat)**
- Workers receive updated model
- Continue for another window

---

## All HeLoCo Parameters (Config + Code)

### **From heloco.yaml**

```yaml
sync_steps: 20              # Window H: inner steps before sync
num_fragments: 1            # 1 = whole model, >1 = split model
outer_lr: 0.7               # η (outer learning rate)
outer_momentum: 0.9         # μ (momentum coefficient)
rho: null                   # ρ (arrival weight) - auto: 1/√islands
coordination_method: sync   # sync or async
```

### **Hardcoded in panoengine/decentralized/heloco.py**

```python
c_ok: float = 0.2           # Alignment threshold
k_s: float = 0.5            # Shrinkage strength
k_d: float = 1.0            # Rotation strength
kappa: float = 3.0          # Confidence momentum scale
beta_max: float = 0.5       # Max shrinkage
eps: float = 1e-8           # Numerical floor
```

---

## When Everything Happens - Quick Reference

| Event | Timing | Key Parameters |
|-------|--------|-----------------|
| Training on each island | Steps 1 to sync_steps | `sync_steps` = 20 |
| Pseudo-gradient sent to server | End of window | - |
| **SHRINKING** applied | On server, for each anti-aligned block (cos_b < 0) | `c_ok`, `k_s`, `beta_max`, `kappa` |
| **ROTATION** applied | On server, for each weakly-aligned block (0 ≤ cos_b < c_ok) | `c_ok`, `k_d`, `kappa` |
| **PASS THROUGH** applied | On server, for each well-aligned block (cos_b ≥ c_ok) | `c_ok` |
| Arrival weight applied | After block correction | `rho` |
| Model update on server | After correction | `outer_lr`, `outer_momentum` |
| Look-ahead model sent to workers | After update | `outer_lr`, `outer_momentum` |

---

## Key Takeaways

1. **Correction happens every `sync_steps`** (e.g., every 20 training iterations)

2. **Three possible corrections per block:**
   - ✅ **PASS** if cos_b ≥ c_ok (default 0.2)
   - 📉 **SHRINK** if cos_b < 0 (using k_s, beta_max)
   - 🔄 **ROTATE** if 0 ≤ cos_b < c_ok (using k_d)

3. **All parameters in heloco.yaml are configurable** (easy to change)

4. **Block correction parameters are hardcoded** in heloco.py (need code edit to change)

5. **Default settings are conservative**: gentle correction, not aggressive

---

## Code Location Reference

| What | File | Lines |
|------|------|-------|
| HeLoCo optimization logic | `panoengine/decentralized/heloco.py` | 40-88 |
| Block correction algorithm | `panoengine/decentralized/heloco.py` | 90-156 |
| HeLoCoServer with defaults | `panoengine/decentralized/heloco.py` | 159-216 |
| Look-ahead snapshot | `panoengine/decentralized/heloco.py` | 218-254 |
| Config file | `heloco.yaml` | 74-93 |
| Main launcher | `run_heloco.py` | - |

