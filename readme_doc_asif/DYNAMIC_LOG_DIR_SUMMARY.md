# Dynamic Log Directory Feature - Summary

## ✅ Implementation Complete

The `run_heloco.py` script now automatically generates **descriptive log directory names** based on your configuration parameters.

---

## What Changed

### Before
```
outputs/heloco_run/method-heloco/
```

### After  
```
outputs/heloco_run_sync_qwen3_0_6b_iid_100/method-heloco/
         ↑
         Auto-generated with:
         - coordination_method: sync
         - config: qwen3_0_6b
         - data_distribution: iid
         - steps: 100
```

---

## Directory Name Pattern

```
heloco_run_{coordination_method}_{config}_{data_distribution}_{steps}
```

### Examples

| Command | Generated Directory |
|---------|-------------------|
| `python run_heloco.py` | `heloco_run_sync_qwen3_0_6b_iid_100` |
| `--coordination-method async --steps 50` | `heloco_run_async_qwen3_0_6b_iid_50` |
| `--config llama3_15m --data-distribution non_iid` | `heloco_run_sync_llama3_15m_non_iid_100` |

---

## How to Use

### Default (Recommended)
```bash
python run_heloco.py
# Auto-generates: outputs/heloco_run_sync_qwen3_0_6b_iid_100/
```

### Customize
```bash
python run_heloco.py --coordination-method async --config llama3_15m --steps 50
# Auto-generates: outputs/heloco_run_async_llama3_15m_iid_50/
```

### Disable (Use Custom Path)
```bash
python run_heloco.py --log-dir my_experiments/run_001
# Uses: my_experiments/run_001/ (NO auto-expansion)
```

---

## Files Modified

| File | Changes |
|------|---------|
| `run_heloco.py` | Added `build_dynamic_log_dir()` function (lines 839-863) + detection in `main()` (lines 866-880) |
| `heloco.yaml` | Updated comment explaining dynamic naming (lines 105-108) |

---

## Files Created

| File | Purpose |
|------|---------|
| `DYNAMIC_LOG_DIR_GUIDE.md` | Complete user guide with examples |
| `test_dynamic_log_dir.py` | Test script (5 test cases, all passing ✅) |
| `DYNAMIC_LOG_DIR_SUMMARY.md` | This summary |

---

## Test Results ✅

```
✅ PASS: outputs/heloco_run → outputs/heloco_run_sync_qwen3_0_6b_iid_100
✅ PASS: outputs/heloco_run → outputs/heloco_run_async_qwen3_0_6b_non_iid_50
✅ PASS: outputs/heloco_run → outputs/heloco_run_sync_llama3_15m_iid_75
✅ PASS: custom_dir → custom_dir/heloco_run_sync_qwen3_0_6b_iid_100
✅ PASS: outputs/heloco_run → outputs/heloco_run_sync_llama3_8b_both_200

Results: 5 passed, 0 failed
```

---

## Parameters in Directory Name

| Parameter | Example |
|-----------|---------|
| `coordination_method` | `sync`, `async` |
| `config` | `qwen3_0_6b`, `llama3_15m` |
| `data_distribution` | `iid`, `non_iid`, `both` |
| `steps` | `50`, `100`, `200` |

---

## Example Workflow

```bash
# Run 1: Default
python run_heloco.py
# → outputs/heloco_run_sync_qwen3_0_6b_iid_100/

# Run 2: Async, 50 steps
python run_heloco.py --coordination-method async --steps 50
# → outputs/heloco_run_async_qwen3_0_6b_iid_50/

# Run 3: Different model
python run_heloco.py --config llama3_15m
# → outputs/heloco_run_sync_llama3_15m_iid_100/

# All three coexist without naming conflicts!
```

---

## Detection Logic

```python
if base_log_dir_arg.name == "heloco_run" and len(base_log_dir_arg.parts) <= 2:
    # Default → build dynamic name
    base_log_dir = build_dynamic_log_dir(base_log_dir_arg, args)
else:
    # Custom → use as-is
    base_log_dir = REPO_ROOT / args.log_dir
```

---

## Key Features

✅ Automatic detection of default vs custom paths  
✅ Self-descriptive directory names  
✅ No naming conflicts between runs  
✅ Easy to disable with custom `--log-dir`  
✅ Backward compatible  

---

## Quick Reference

**See directory name before running:**
```bash
python run_heloco.py --dry-run
```

**Use static path:**
```bash
python run_heloco.py --log-dir my_static_dir
```

**Detailed guide:**
See `DYNAMIC_LOG_DIR_GUIDE.md`

