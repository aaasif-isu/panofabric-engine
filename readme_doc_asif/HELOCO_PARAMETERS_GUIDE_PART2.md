# HeLoCo Parameters Guide - Part 2: Block Correction Logic

## HeLoCo Block Correction Parameters

These parameters control **WHEN rotation/shrinking happens**. They are hardcoded in HeLoCoServer.

### **Block Correction Decision Logic (Algorithm 2)**

For each tensor block in the pseudo-gradient:

```
cos_b = dot(Δ_b, m_b) / (‖Δ_b‖ · ‖m_b‖)    [cosine similarity]
conf_b = ‖Δ_b‖ / (‖Δ_b‖ + κ·‖m_b‖ + ε)    [confidence]
```

**Decision Tree:**

```
IF cos_b ≥ c_ok:
    ✅ PASS THROUGH (Eq. 9)
ELIF cos_b < 0:
    📉 SHRINK (Eqs. 10-11)
ELSE:
    🔄 ROTATE (Eqs. 12-14)
```

---

## When Each Operation Happens

### **🔴 SHRINKING - When cos_b < 0**

Block points opposite to momentum direction → reduce its magnitude

**Formula**: 
```
β_b = clamp(k_s · (-cos_b) · conf_b, max=β_max)
Δ̂_b = Δ_b - β_b · cos_b · ‖Δ_b‖ · v̂_b
```

**Parameters**:
- `k_s` = 0.5 (strength)
- `beta_max` = 0.5 (max ~50% removal)
- `kappa` = 3.0 (affects confidence)

---

### **🟢 ROTATION - When 0 ≤ cos_b < c_ok**

Block weakly-aligned → rotate toward momentum, keep magnitude

**Formula**:
```
λ_b = clamp(k_d · (1 - cos_b) · conf_b, max=1.0)
Δ̂_b = ‖Δ_b‖ · (direction_mix) / ‖direction_mix‖
```

**Parameters**:
- `k_d` = 1.0 (strength)
- `c_ok` = 0.2 (threshold)
- `kappa` = 3.0 (affects confidence)

---

### **⚪ PASS THROUGH - When cos_b ≥ c_ok**

Block well-aligned → use as-is

**Parameter**: `c_ok` = 0.2

---

## Current Default Values (Hardcoded)

```python
c_ok: float = 0.2         # Pass if cos_b ≥ 0.2
k_s: float = 0.5          # Shrinkage strength
k_d: float = 1.0          # Rotation strength  
kappa: float = 3.0        # Confidence momentum scale
beta_max: float = 0.5     # Max shrinkage ~50%
eps: float = 1e-8         # Numerical floor
```

All defined in: `panoengine/decentralized/heloco.py` lines 180-215

