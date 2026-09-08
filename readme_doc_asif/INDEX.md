# PanoEngine Documentation Index

Comprehensive documentation for the PanoEngine decentralized training framework.

## Quick Navigation

### Async Implementation (Path A)
- **ASYNC_QUICK_SUMMARY.md** - 2-minute overview of async design
- **ASYNC_CONVERSION_ROADMAP.md** - Detailed 5-phase roadmap with effort estimates
- **ASYNC_ARCHITECTURE_DIAGRAM.md** - Visual architecture and staleness handling
- **ASYNC_IMPLEMENTATION_GUIDE.md** - Phase 1 implementation checklist

### Data Distribution & Performance
- **NON_IID_DATA_GUIDE.md** - Non-IID setup with multilingual C4 data
- **DYNAMIC_SCALING_GUIDE.md** - Dynamic worker allocation
- **RUN_RESULTS_2024_09_07.md** - Actual run outputs and metrics
- **COMPARISON_PLOT_EXPLANATION.md** - Loss comparison plots

### Methods & Setup
- **FINAL_IMPLEMENTATION_SUMMARY.md** - High-level implementation overview
- **MULTI_METHOD_SETUP.md** - Running multiple methods sequentially

## Key Features Added

### Coordination Methods
- **sync** (default): Window-synchronous with torchft barrier
- **async**: Fully asynchronous HTTP push/pull with staleness weighting

### Decentralized Methods
- **heloco**: HeLoCo with block corrections
- **diloco**: DiLoCo base
- **mla**: Momentum and Loss-Aware optimizer (NEW)

### Data Distributions
- **iid**: All islands see English (C4, default)
- **non_iid**: Each island sees different language
- **both**: Run IID then non_iid sequentially

## Configuration

### Master Config: heloco.yaml
- `methods`: [heloco, diloco, mla]
- `coordination_method`: sync or async
- `data_distribution`: iid, non_iid, or both
- `languages`: per-island languages for non_iid
- `mla_*`: MLA hyperparameters

### MLA Hyperparameters
- `mla_k_dir` (default: 1.0): Direction correction strength
- `mla_cos_ok` (default: 0.2): Well-aligned threshold
- `mla_cos_bad` (default: -0.2): Anti-aligned threshold
- `mla_conf_c` (default: 3.0): Confidence weight
- `mla_k_shrink` (default: 0.5): Shrinking strength
- `mla_beta_max` (default: 0.5): Max shrinking factor

### Running Examples
```bash
# Default config
python run_heloco.py

# With async coordination
python run_heloco.py --coordination-method async

# Non-IID with MLA
python run_heloco.py --methods mla --data-distribution non_iid \
  --languages english french

# Both IID and non-IID
python run_heloco.py --data-distribution both --languages english french
```

## New Files Added

- `panoengine/decentralized/async_trainer.py` - Standalone async trainer (Path A)
- `panoengine/decentralized/mla.py` - MLAOptimizer implementation

## File Reorganization

Moved from root to `readme_doc_asif/`:
- 4× ASYNC_*.md files (roadmap, quick summary, architecture, guide)
- NON_IID_DATA_GUIDE.md
- DYNAMIC_SCALING_GUIDE.md
- RUN_RESULTS_2024_09_07.md
- COMPARISON_PLOT_EXPLANATION.md
- FINAL_IMPLEMENTATION_SUMMARY.md
- MULTI_METHOD_SETUP.md

## Status

✅ MLA optimizer (panoengine/decentralized/mla.py)
✅ Async trainer entry point (panoengine/decentralized/async_trainer.py)
✅ Updated heloco.yaml with new options
✅ Updated run_heloco.py to support mla & async config
✅ Documentation reorganized

*Last Updated: 2024-09-07*
