# Comparison Plot Design - Both Methods on ONE Figure

## Answer: YES, it's a single figure with both methods!

The `comparison_loss.png` file contains **ONE figure with TWO subplots**:

### Left Subplot: Loss Over Steps
- **Blue line**: HeLoCo method loss curve (averaged across islands)
- **Orange line**: DiLoCo method loss curve (averaged across islands)
- Both methods plotted on the **SAME axes** for direct comparison
- Legend shows which line is which

### Right Subplot: Perplexity Over Steps  
- **Blue line**: HeLoCo method perplexity
- **Orange line**: DiLoCo method perplexity
- Both methods plotted on the **SAME axes**
- Same color scheme as left subplot

## Code Flow (from `run_heloco.py`)

### main() function (lines 710-745):
```python
# 1. Run each method sequentially
methods_data: dict[str, list[list[dict]]] = {}
for method in args.methods:                           # heloco, then diloco
    method_log_dir = base_log_dir / f"method-{method}"
    success = run_single_method(method, args, ...)
    
    # 2. Collect metrics from each method
    for i in range(args.islands):
        csv_path = method_log_dir / "metrics" / f"island-{i}_steps.csv"
        methods_data[method].append(rows)              # Store both methods' data

# 3. Generate ONE comparison plot with ALL methods
comparison_path = base_log_dir / "comparison_loss.png"
_plot_comparison(methods_data, comparison_path)       # Plots both on same figure
```

### _plot_comparison() function (lines 508-572):
```python
# Create ONE figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loop through all methods and plot each on the SAME axes
for color_idx, (method, islands_steps) in enumerate(methods_data.items()):
    ...
    # BOTH lines plot on the SAME subplot
    ax1.plot(sorted_steps, avg_losses, label=method, color=color)  # Same ax1!
    ax2.plot(sorted_steps, ppls, label=method, color=color)        # Same ax2!

plt.savefig(png_path)  # ONE PNG file
```

## Visual Representation

```
┌──────────────────────────────────────────────────────────────────────┐
│  Decentralized Training Method Comparison (Loss & Perplexity)        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Loss Over Steps           │   Perplexity Over Steps                │
│  (BOTH methods here)       │   (BOTH methods here)                  │
│                            │                                        │
│  Loss │                    │  PPL  │                               │
│   3.0 │ •  ← step 0        │  20   │ •  ← step 0                   │
│   2.5 │  \                 │  15   │  \                            │
│   2.0 │   •  heloco (blue) │  10   │   •  heloco (blue)            │
│   1.5 │    \               │   8   │    \                          │
│   1.0 │     • diloco ──── │   5   │     • diloco (orange) ───     │
│   0.5 │      (orange)      │   3   │      (orange)                 │
│       └────────────────    │       └────────────────               │
│       0    25   50   100   │       0    25   50   100              │
│           Steps           │            Steps                       │
│                            │                                        │
│  Legend:                   │   Legend:                             │
│  • heloco (blue)           │   • heloco (blue)                     │
│  • diloco (orange)         │   • diloco (orange)                   │
│                            │                                        │
└──────────────────────────────────────────────────────────────────────┘
```

## Key Points

✅ **One PNG file**: Easy to share and view  
✅ **Both methods visible**: Direct visual comparison  
✅ **Same axes**: Easy to see which method converges faster/better  
✅ **Color-coded**: Blue = heloco, Orange = diloco  
✅ **Legend included**: Clear labeling  
✅ **Two metrics**: Loss AND perplexity for comprehensive analysis  

## File Structure After Running

```
outputs/heloco_run/
├── method-heloco/
│   ├── logs/
│   └── metrics/
│       ├── island-0_steps.csv      ← used for comparison
│       ├── island-1_steps.csv      ← used for comparison
│       ├── island-0_loss.png       ← separate per-island plots
│       └── island-1_loss.png       ← separate per-island plots
├── method-diloco/
│   ├── logs/
│   └── metrics/
│       ├── island-0_steps.csv      ← used for comparison
│       ├── island-1_steps.csv      ← used for comparison
│       ├── island-0_loss.png       ← separate per-island plots
│       └── island-1_loss.png       ← separate per-island plots
└── comparison_loss.png             ← ★ MAIN COMPARISON PLOT ★
    (contains both heloco and diloco lines on same figure)
```

## Tested and Verified

✓ Test script `test_comparison_plot.py` generates valid PNG  
✓ PNG contains 1390x495 pixels with colored lines (both methods)  
✓ Matplotlib correctly plots multiple methods on same axes  
✓ Color scheme differentiates between methods clearly  

## How to Use

1. Run the launcher:
   ```bash
   heloco/bin/python run_heloco.py
   ```

2. Wait for both methods to complete (~1-2 hours for 100 steps each)

3. View the results:
   - **Main comparison**: `outputs/heloco_run/comparison_loss.png`
   - **Per-island details**: `outputs/heloco_run/method-*/metrics/island-*_loss.png`
   - **CSV data**: `outputs/heloco_run/method-*/metrics/island-*_steps.csv`

The comparison plot will clearly show which method converges faster!
