# HeLoCo Block Correction - Decision Tree

## What Happens to Each Block?

```
Block Δ_b arrives from worker
with current momentum m_b
         │
         ▼
┌─────────────────────────────┐
│ Compute alignment:          │
│ cos_b = dot(Δ_b, m_b)      │
│         / (‖Δ_b‖·‖m_b‖)   │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Is cos_b ≥ c_ok (0.2)?      │
└───────┬─────────────────┬───┘
        │ YES             │ NO
        ▼                 ▼
┌──────────────┐   ┌────────────────────┐
│ ✅ PASS      │   │ Is cos_b < 0?      │
│ Use as-is    │   └────┬────────────┬──┘
│ (Eq. 9)      │        │ YES        │ NO
└──────────────┘        ▼            ▼
        │         ┌──────────┐  ┌─────────────┐
        │         │📉 SHRINK │  │🔄 ROTATE   │
        │         └──────────┘  └─────────────┘
        │                │              │
        └────┬───────────┴──────────────┘
             ▼
    Apply ρ scaling
             │
             ▼
    Update server &
    send look-ahead θ̄
```

### **Decision Rules (Hardcoded Defaults)**

| Condition | Action | Parameters |
|-----------|--------|-----------|
| cos_b ≥ 0.2 | ✅ PASS | `c_ok = 0.2` |
| cos_b < 0 | 📉 SHRINK | `k_s = 0.5, beta_max = 0.5` |
| 0 ≤ cos_b < 0.2 | 🔄 ROTATE | `k_d = 1.0` |

---

## Parameter Control Points

### **c_ok = 0.2 (Alignment Threshold)**
- If cos_b ≥ 0.2 → PASS
- Lower c_ok → more blocks pass (less correction)
- Higher c_ok → more blocks get corrected (more aggressive)

### **k_s = 0.5 (Shrink Strength)**
- Controls magnitude of shrinking for anti-aligned blocks
- β_b = clamp(k_s · (-cos_b) · conf_b, max=beta_max)
- Higher k_s → more aggressive shrinking

### **k_d = 1.0 (Rotate Strength)**
- Controls rotation angle for weakly-aligned blocks
- λ_b = clamp(k_d · (1 - cos_b) · conf_b, max=1.0)
- Higher k_d → more rotation toward momentum

### **kappa = 3.0 (Confidence Scale)**
- conf_b = ‖Δ_b‖ / (‖Δ_b‖ + κ·‖m_b‖)
- Higher κ → lower confidence in gradient → gentler correction
- Lower κ → higher confidence in gradient → more aggressive

### **beta_max = 0.5 (Shrinkage Cap)**
- Maximum amount a block can be shrunk (~50%)
- Hard limit on β_b ≤ 0.5

### **rho = 1/√K (Arrival Weight)**
- ρ applied after correction
- K = number of islands
- 2 islands → ρ ≈ 0.707
- 4 islands → ρ = 0.5

---

## Timeline: One Window Cycle

**Steps 1-20:** Train locally
**Step 20:** Send pseudo-gradient to server
**Step 20 (server):** Apply corrections above
**Step 21-40:** Train with updated model
(repeat)

---

## Default Parameter Values

```
FROM heloco.yaml:
sync_steps: 20               (WHEN correction happens)
outer_lr: 0.7                (server learning rate)
outer_momentum: 0.9          (server momentum)

FROM heloco.py (lines 180-215):
c_ok: 0.2                    (PASS threshold)
k_s: 0.5                     (SHRINK strength)
k_d: 1.0                     (ROTATE strength)
kappa: 3.0                   (confidence scale)
beta_max: 0.5                (max shrinkage)
eps: 1e-8                    (numerical floor)
```

