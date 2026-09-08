# HeLoCo Parameters - Documentation Index

Complete documentation explaining when/how block correction happens and what parameters control it.

---

## 📚 Documentation Files

### **START HERE** ⭐
**HELOCO_QUICK_REFERENCE.md** - One-page summary with all essentials

### **VISUAL GUIDES**
**HELOCO_CORRECTION_FLOWCHART.md** - ASCII flowchart and decision tree

### **DETAILED GUIDES** (3 parts)
- **HELOCO_PARAMETERS_GUIDE_PART1.md** - Overview & timing
- **HELOCO_PARAMETERS_GUIDE_PART2.md** - Block correction logic
- **HELOCO_PARAMETERS_GUIDE_PART3.md** - Timeline & summary

---

## 🎯 Quick Answers

### **When does tensorwise correction happen?**
Every `sync_steps` training iterations (default: 20), on the parameter server.

### **When does SHRINKING happen?**
When `cos_b < 0` (anti-aligned block, pointing opposite to momentum).
- Controlled by: `k_s` (strength), `beta_max` (max ~50%)

### **When does ROTATION happen?**
When `0 ≤ cos_b < c_ok` (weakly-aligned block).
- Controlled by: `k_d` (strength), `c_ok` (threshold=0.2)

### **When does PASS THROUGH happen?**
When `cos_b ≥ c_ok` (well-aligned block).
- Controlled by: `c_ok` (default 0.2)

---

## 📋 All Parameters

### **From heloco.yaml (Easy to Change)**
```yaml
sync_steps: 20              # When correction happens
outer_lr: 0.7               # Server learning rate
outer_momentum: 0.9         # Server momentum
rho: null                   # Arrival weight (auto: 1/√K)
```

### **From heloco.py (Hardcoded)**
```python
c_ok: 0.2                   # PASS threshold
k_s: 0.5                    # SHRINK strength
k_d: 1.0                    # ROTATE strength
kappa: 3.0                  # Confidence scale
beta_max: 0.5               # Max shrinkage
```

---

## 🔧 How to Modify

### **Easy (Edit heloco.yaml)**
Change `sync_steps`, `outer_lr`, etc. and re-run.

### **Hard (Edit heloco.py)**
Change `c_ok`, `k_s`, `k_d`, etc. in lines 180-215, then rebuild.

---

## 📊 Effects Table

| Parameter | Default | Controls |
|-----------|---------|----------|
| `sync_steps` | 20 | **TIMING**: How often correction happens |
| `c_ok` | 0.2 | **THRESHOLD**: When to pass through |
| `k_s` | 0.5 | **STRENGTH**: How much to shrink |
| `k_d` | 1.0 | **STRENGTH**: How much to rotate |
| `kappa` | 3.0 | **CONFIDENCE**: More = less aggressive |
| `beta_max` | 0.5 | **CAP**: Maximum shrinkage ~50% |
| `outer_lr` | 0.7 | **SERVER**: Learning rate |
| `outer_momentum` | 0.9 | **SERVER**: Momentum coefficient |

---

## 🎓 Key Formulas

```
Alignment:  cos_b = dot(Δ_b, m_b) / (‖Δ_b‖ · ‖m_b‖)
Threshold:  if cos_b ≥ c_ok → PASS
            elif cos_b < 0 → SHRINK
            else → ROTATE
```

---

## 🔍 Code References

| What | File | Lines |
|------|------|-------|
| HeLoCo optimizer | `panoengine/decentralized/heloco.py` | 40-88 |
| Block correction | `panoengine/decentralized/heloco.py` | 90-156 |
| HeLoCoServer | `panoengine/decentralized/heloco.py` | 159-216 |
| Config file | `heloco.yaml` | 74-107 |

---

## ⚡ One-Minute Summary

1. **WHEN**: Every 20 steps
2. **WHERE**: Parameter server
3. **WHAT**: Check gradient alignment with momentum
4. **HOW**:
   - cos_b ≥ 0.2 → ✅ Pass
   - cos_b < 0 → 📉 Shrink
   - 0 ≤ cos_b < 0.2 → 🔄 Rotate
5. **SCALE**: Apply ρ = 1/√K
6. **UPDATE**: Server model
7. **SEND**: Look-ahead to workers

---

## 📚 Files in This Directory

- **HELOCO_QUICK_REFERENCE.md** - Start here!
- **HELOCO_CORRECTION_FLOWCHART.md** - Visual flowchart
- **HELOCO_PARAMETERS_GUIDE_PART1.md** - Overview & config
- **HELOCO_PARAMETERS_GUIDE_PART2.md** - Block correction logic
- **HELOCO_PARAMETERS_GUIDE_PART3.md** - Timeline & summary
- **HELOCO_PARAMETERS_INDEX.md** - This file

