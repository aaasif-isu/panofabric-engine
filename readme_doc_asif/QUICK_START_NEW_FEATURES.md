# Quick Start Guide - New Features (Sept 7, 2026)

## What's New? 

Three improvements to your panofabric setup:

### 1. 📊 **Centralized Evaluation Log**
- **What**: Single log file with ALL method metrics
- **Where**: `outputs/heloco_run/evaluation_summary_*.log`
- **Contains**: Loss, perplexity for each step, all methods combined
- **Plus**: Automatic summary statistics (improvements, records count)

### 2. 🎯 **MLA Now in Comparison Plot**
- **What**: MLA method now appears in the comparison plot
- **Why**: Fixed glob pattern to find MLA's CSV files with suffixes
- **Benefit**: See all three methods (diloco, heloco, mla) on one plot

### 3. 📈 **Thinner Lines in Plot**
- **What**: Reduced line width from 2.5 → 1.0
- **Why**: Requested - lines were too thick
- **Benefit**: Cleaner, easier to read comparison plot

---

## How to Use

### Basic Usage (Same as Before)
```bash
python run_heloco.py --config-file heloco.yaml
```

### With Multiple Methods (Now Includes MLA!)
```bash
python run_heloco.py --methods diloco,heloco,mla --steps 100
```

---

## New Output Files

After running, you'll find:

```
outputs/heloco_run/
├── comparison_loss_iid.png              ← NOW includes MLA! Lines are thinner!
├── evaluation_summary_iid.log           ← NEW! Centralized metrics
├── method-diloco/
│   └── metrics/
├── method-heloco/
│   └── metrics/
└── method-mla/
    └── metrics/
```

---

## Reading the Evaluation Log

### View the whole log
```bash
cat outputs/heloco_run/evaluation_summary_iid.log
```

### View just the summary
```bash
tail -30 outputs/heloco_run/evaluation_summary_iid.log
```

### View just the CSV data
```bash
head -60 outputs/heloco_run/evaluation_summary_iid.log | tail -50
```

---

## Log Format Example

### Header
```
================================================================================
CENTRALIZED EVALUATION LOG - Generated 2026-09-07 18:36:34
================================================================================
Total records: 500
Methods: diloco, heloco, mla
================================================================================
```

### CSV Data (searchable!)
```
method,island,step,loss,perplexity
diloco,0,1,8.167489,3524.483049
diloco,0,2,8.118116,3354.695814
diloco,1,1,8.167380,3524.123456
...
heloco,0,1,8.120691,3363.345020
heloco,1,1,8.119234,3362.123456
...
mla,0,1,8.214996,3695.962690
mla,1,1,8.212345,3693.234567
```

### Summary by Method
```
Method: diloco
  Records: 200
  First Loss: 8.167489 (PPL: 3524.483049)
  Last Loss:  3.076166 (PPL: 21.675149)
  Loss improvement: 5.091323
  PPL improvement: 3502.807900

Method: heloco
  Records: 200
  First Loss: 8.120691 (PPL: 3363.345020)
  Last Loss:  2.884336 (PPL: 17.891688)
  Loss improvement: 5.236355
  PPL improvement: 3345.453332

Method: mla
  Records: 100
  First Loss: 8.214996 (PPL: 3695.962690)
  Last Loss:  2.846798 (PPL: 17.232514)
  Loss improvement: 5.368198
  PPL improvement: 3678.730176
```

---

## FAQ

**Q: Will this break my existing runs?**
A: No! The new features are additions. Your old logs and method-specific outputs are unchanged.

**Q: Can I still see individual method logs?**
A: Yes! They're in `method-*/metrics/` and `method-*/lighthouse.log` as before.

**Q: Why is MLA now in the plot?**
A: The code now supports multiple CSV filename patterns (fixed glob matching).

**Q: Where did the thicker lines go?**
A: Changed from linewidth=2.5 to linewidth=1.0 for cleaner plots.

**Q: How often is the log generated?**
A: Once per run, at the end, after all methods are complete.

**Q: Can I parse the CSV data programmatically?**
A: Yes! It's standard CSV format, easy to import into pandas, etc.

---

## What Changed in the Code?

### File: `run_heloco.py`

1. **New Function**: `_create_evaluation_log()` (lines 566-642)
   - Generates centralized log with all methods

2. **Updated Logic**: CSV collection with glob patterns (lines 885-907)
   - Now finds diloco/heloco/mla CSVs correctly

3. **Thinner Lines**: Changed linewidth (lines 691, 695)
   - From 2.5 → 1.0

4. **Integration**: New log generation call (lines 927-930)
   - Automatically creates evaluation_summary_*.log

---

## Examples

### Compare methods programmatically
```python
import pandas as pd

# Read the centralized log
df = pd.read_csv('outputs/heloco_run/evaluation_summary_iid.log', 
                  skiprows=7,  # Skip header section
                  nrows=500)   # Adjust based on your data

# Group by method
for method in df['method'].unique():
    method_df = df[df['method'] == method]
    print(f"\n{method}:")
    print(f"  First loss: {method_df['loss'].iloc[0]:.4f}")
    print(f"  Last loss:  {method_df['loss'].iloc[-1]:.4f}")
    print(f"  Min loss:   {method_df['loss'].min():.4f}")
```

### Extract specific method data
```bash
# Get just diloco data
grep "^diloco," outputs/heloco_run/evaluation_summary_iid.log

# Get just island 0
grep ",0," outputs/heloco_run/evaluation_summary_iid.log

# Get steps 50-100
awk -F',' '$3 >= 50 && $3 <= 100' outputs/heloco_run/evaluation_summary_iid.log
```

---

## Summary of Changes

| Feature | Status | Benefit |
|---------|--------|---------|
| Centralized log | ✅ Added | One file, all methods |
| MLA in plot | ✅ Fixed | See all 3 methods |
| Thinner lines | ✅ Reduced | Cleaner plots |
| Backward compatible | ✅ Yes | No breaking changes |

**Everything works automatically - just run your script as usual! 🚀**
