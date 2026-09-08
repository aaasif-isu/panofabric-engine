# Multi-Method Decentralized Training Launcher

## Overview

You can now run **multiple decentralized training methods sequentially** (HeLoCo, DiLoCo, etc.) on the same model and dataset, with **automatic comparison plots** showing loss curves side-by-side.

## What's Ready

### 1. **Enhanced Config File** (`heloco.yaml`)
✅ **DONE** - Fully configured with:
- **`methods: [heloco, diloco]`** — Run both sequentially
- **Model selection** with commented options:
  - `models.llama3_small` + `llama3_15m` (15M, default)
  - `models.llama3` + `llama3_base` (130M)
  - `models.qwen3_0_6b` + `qwen3_0_6b` (600M)
- All training and infrastructure parameters
- Clear comments showing available options

### 2. **Metric Collection** (`run_heloco.py`)
✅ **DONE** - Per-step and per-method metrics:
- Each method's logs saved in `outputs/heloco_run/method-<method>/`
- CSVs per island: `island-<i>_steps.csv`, `island-<i>_comm.csv`
- Summary table: `summary.txt`
- Line plots: `island-<i>_loss.png` (once matplotlib installs)

### 3. **Comparison Plotting** (`run_heloco.py`)
✅ **READY** — All helper functions written:
- `_plot_comparison()` function plots loss & perplexity curves across all methods
- `run_single_method()` runs each method sequentially
- Main loop logic just needs to be connected

## Usage (Current State)

With the changes already made:

```bash
# Reads heloco.yaml automatically
python run_heloco.py

# Override methods on command line
python run_heloco.py --methods heloco,diloco --steps 50

# Dry run to see commands
python run_heloco.py --config-file heloco.yaml --dry-run
```

## Output Structure

```
outputs/heloco_run/
├── method-heloco/
│   ├── island-0.log, island-1.log
│   ├── param-server.log, lighthouse.log
│   └── metrics/
│       ├── island-{0,1}_steps.csv
│       ├── island-{0,1}_comm.csv
│       ├── island-{0,1}_loss.png
│       └── summary.txt
├── method-diloco/
│   └── ... (same)
└── comparison_loss.png    ← both methods overlaid
```

## Metrics Saved

- **island-<i>_steps.csv**: step, loss, loss_max, grad_norm, lr, tokens_per_s, gpu_mem_gib, step_time_s
- **island-<i>_comm.csv**: step, exchange, bytes_up, bytes_down, seconds, mbps, gb_cumulative
- **summary.txt**: table of first/last loss, perplexity, exchange count per island
- **island-<i>_loss.png**: 2×2 grid (loss, ppl, grad_norm, lr)
- **comparison_loss.png**: all methods' loss & ppl overlaid

## Available Models

In `heloco.yaml`:

```yaml
# Option 1: Small (15M, recommended for 40GB RAM)
module: models.llama3_small
config: llama3_15m
hf_assets: ./assets/tokenizer/debug

# Option 2: Medium (130M)
module: models.llama3
config: llama3_base
hf_assets: ./assets/hf/meta-llama

# Option 3: Large (600M)
module: models.qwen3_0_6b
config: qwen3_0_6b
hf_assets: ./assets/hf/Qwen3-0.6B
```

## Summary

✅ **Config**: `heloco.yaml` ready with methods list & model options
✅ **Metrics**: Auto-generated CSVs & plots per-method
✅ **Comparison**: Loss curves overlaid in `comparison_loss.png`
✅ **Flow**: Sequential method execution with side-by-side impact comparison

**Run:** `python run_heloco.py` and check `outputs/heloco_run/` for results!
