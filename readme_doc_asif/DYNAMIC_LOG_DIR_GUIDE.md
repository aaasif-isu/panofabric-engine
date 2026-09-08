# Dynamic Log Directory Naming Guide

## Overview

The `run_heloco.py` script now automatically generates **descriptive log directory names** based on your configuration parameters.

---

## Pattern

```
heloco_run_{coordination_method}_{config}_{data_distribution}_{steps}
```

### Components

| Component | Example |
|-----------|---------|
| `coordination_method` | sync, async |
| `config` | qwen3_0_6b, llama3_15m |
| `data_distribution` | iid, non_iid |
| `steps` | 50, 100, 200 |

---

## Examples

### Example 1: Default Run
```bash
python run_heloco.py
```
**Generated:** `outputs/heloco_run_sync_qwen3_0_6b_iid_100/`

### Example 2: Async with Non-IID
```bash
python run_heloco.py --coordination-method async --data-distribution non_iid --languages english french
```
**Generated:** `outputs/heloco_run_async_qwen3_0_6b_non_iid_100/`

### Example 3: Different Model & Steps
```bash
python run_heloco.py --config llama3_15m --steps 50
```
**Generated:** `outputs/heloco_run_sync_llama3_15m_iid_50/`

---

## How It Works

### Auto-Detection

- **Default detected** (`outputs/heloco_run`): Auto-generates with parameters
- **Custom detected** (any other value): Uses path as-is

### To Disable Dynamic Naming
```bash
python run_heloco.py --log-dir my_experiments/custom_run
# Uses my_experiments/custom_run/ without modification
```

---

## Directory Structure

```
outputs/heloco_run_sync_qwen3_0_6b_iid_100/
├── method-heloco/
│   ├── metrics/
│   │   ├── island-0_steps.csv
│   │   └── island-1_steps.csv
│   ├── lighthouse.log
│   ├── param-server.log
│   └── island-{0,1}.log
├── method-diloco/
│   └── ...
├── method-mla/
│   └── ...
├── comparison_loss_iid.png
└── evaluation_summary_iid.log
```

---

## Configuration

### heloco.yaml
```yaml
log_dir: outputs/heloco_run   # Default - auto-expands
# OR
log_dir: custom_path          # Custom - used as-is
```

### Command Line
```bash
python run_heloco.py --log-dir outputs/heloco_run  # Auto-expands
python run_heloco.py --log-dir custom_path          # Static path
```

---

## Implementation

**File:** `run_heloco.py` (lines 839-880)

**Function:** `build_dynamic_log_dir()` creates the dynamic name

**Detection:** `main()` checks if path matches default pattern

---

## Best Practices

✅ Use default `outputs/heloco_run` for automatic organization  
✅ Vary `--steps`, `--config`, `--coordination-method` for comparisons  
✅ Check console output to see actual directory used  

❌ Don't manually create dynamic directory names  
❌ Don't assume directory if using custom `--log-dir`

---

## Examples Workflow

```bash
# Experiment 1
python run_heloco.py --steps 100
# → outputs/heloco_run_sync_qwen3_0_6b_iid_100/

# Experiment 2
python run_heloco.py --coordination-method async --steps 100
# → outputs/heloco_run_async_qwen3_0_6b_iid_100/

# Experiment 3
python run_heloco.py --config llama3_15m --steps 50
# → outputs/heloco_run_sync_llama3_15m_iid_50/

# All three coexist in outputs/ without conflicts!
ls outputs/
```

---

## FAQ

**Q: Can I see the directory before running?**  
A: Yes! Use `--dry-run` to print commands without executing.

**Q: How do I disable auto-naming?**  
A: Set custom `--log-dir my_manual_dir/experiment_1`

**Q: What if I'm still getting `outputs/heloco_run/`?**  
A: Check if you're using a custom `--log-dir` value.

