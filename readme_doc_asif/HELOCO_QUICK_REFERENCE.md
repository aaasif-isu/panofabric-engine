# HeLoCo Parameters - Quick Reference

## TL;DR: When Does What Happen?

### **CORRECTION TIMING**
- **Every `sync_steps` training iterations** (default: 20)
- On the parameter SERVER (not on workers)
- Applied to each incoming pseudo-gradient from each island

### **WHAT HAPPENS TO EACH BLOCK?**

Based on alignment between gradient block Δ_b and momentum m_b:

```
Alignment = cos_b = dot(Δ_b, m_b) / (‖Δ_b‖ · ‖m_b‖)

cos_b ≥ 0.2           → ✅ PASS (use as-is)
0 ≤ cos_b < 0.2       → 🔄 ROTATE toward momentum
cos_b < 0             → 📉 SHRINK (reduce magnitude)
```

---

## Configuration Parameters (heloco.yaml)

Easy to modify - just edit the file!

```yaml
sync_steps: 20              # Window size (when correction happens)
num_fragments: 1            # 1 = whole model sync
outer_lr: 0.7               # Learning rate η
outer_momentum: 0.9         # Momentum coefficient μ
rho: null                   # Arrival weight (auto: 1/√K)
coordination_method: sync   # sync or async mode
```

---

## Block Correction Parameters (Hardcoded)

In: `panoengine/decentralized/heloco.py` lines 180-215

To change, you need to edit the code:

```python
c_ok: 0.2              # Threshold for "pass" (cos_b ≥ c_ok = PASS)
k_s: 0.5               # Shrinkage strength (for anti-aligned blocks)
k_d: 1.0               # Rotation strength (for weakly-aligned blocks)
kappa: 3.0             # Affects confidence calculation
beta_max: 0.5          # Max shrinkage ~50%
eps: 1e-8              # Numerical floor
```

---

## Full Parameter Reference

| Parameter | Location | Current Value | Controls |
|-----------|----------|----------------|----------|
| `sync_steps` | heloco.yaml | 20 | **When** correction happens (every N steps) |
| `num_fragments` | heloco.yaml | 1 | Model split for fragment-wise sync |
| `outer_lr` | heloco.yaml | 0.7 | Server learning rate η |
| `outer_momentum` | heloco.yaml | 0.9 | Server momentum μ |
| `rho` | heloco.yaml | null (auto) | **Arrival weight ρ after correction** |
| `coordination_method` | heloco.yaml | sync | Sync barrier or async |
| `c_ok` | heloco.py:199 | 0.2 | **When PASS happens** (cos_b ≥ c_ok) |
| `k_s` | heloco.py:200 | 0.5 | **When/how much to SHRINK** |
| `k_d` | heloco.py:201 | 1.0 | **When/how much to ROTATE** |
| `kappa` | heloco.py:202 | 3.0 | **Affects decision thresholds** |
| `beta_max` | heloco.py:203 | 0.5 | **Max shrinkage magnitude** |
| `eps` | heloco.py:204 | 1e-8 | Numerical floor |

---

## Example: What Happens in One Window

```
Step 1-20:   Island 1 & 2 train independently
↓
Step 20:     Worker 1 computes Δ₁ = θ₁ - θ_initial
             Send to server
↓
On Server:   For each tensor block b in Δ₁:
             1. Compute alignment: cos_b
             2. Check alignment:
                • cos_b ≥ 0.2  → keep block (PASS)
                • cos_b < 0    → shrink block (SHRINK)
                • 0≤cos_b<0.2  → rotate block (ROTATE)
             3. Apply ρ scaling
↓
             Update momentum: m = 0.9·m + 0.1·Δ_corrected
             Update model: θ = θ - 0.7·(Δ_corrected + 0.9·m)
↓
             Compute look-ahead: θ̄ = θ - 0.7·0.9·m
↓
             Send θ̄ to Worker 1
↓
Step 21-40:  Worker 1 starts new window with updated θ̄
```

---

## Key Files

1. **heloco.yaml** - All configurable parameters
2. **panoengine/decentralized/heloco.py** - HeLoCo implementation
   - Lines 40-88: HeLoCoOptimizer
   - Lines 90-156: block_correct() function  
   - Lines 159-216: HeLoCoServer class with block correction defaults
3. **run_heloco.py** - Main launcher

---

## How to Modify Parameters

### Easy (Edit heloco.yaml)
```yaml
sync_steps: 30              # Change window size
outer_lr: 0.5               # Change learning rate
coordination_method: async  # Switch to async
```

### Hard (Edit heloco.py)
```python
# Line 199 in HeLoCoServer.__init__:
c_ok: float = 0.1,          # Change from 0.2 to 0.1
k_s: float = 1.0,           # Change from 0.5 to 1.0
k_d: float = 2.0,           # Change from 1.0 to 2.0
```

Then rebuild/reinstall the package.

---

## What Each Parameter Does

### TIMING Parameters
- **sync_steps**: How often correction happens
- **coordination_method**: sync (barrier) or async (no barrier)

### CORRECTION Parameters  
- **c_ok**: Threshold above which blocks pass through
- **k_s**: How aggressively to shrink anti-aligned blocks
- **k_d**: How aggressively to rotate weakly-aligned blocks
- **kappa**: Higher = trust momentum more, correct less
- **beta_max**: Cap on maximum shrinkage

### OUTER OPTIMIZER Parameters
- **outer_lr**: Server learning rate
- **outer_momentum**: Server momentum coefficient
- **rho**: Scale factor after correction

---

## References

- Paper: https://arxiv.org/pdf/2606.00271
- Implementation: `panoengine/decentralized/heloco.py`
- Full guide: See HELOCO_PARAMETERS_GUIDE_PART1.md through PART3.md

