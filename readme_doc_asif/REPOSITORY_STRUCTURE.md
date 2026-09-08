# Repository Structure - Organized

## 📁 Root Directory Organization

### 🚀 Production Code (Active)
```
run_heloco.py              - Main YAML-driven multi-method launcher
generate_plots.py          - Utility for generating comparison plots
heloco.yaml                - Configuration file with all parameters
```

### 📚 Documentation (Reference)
```
README.md                              - Original repo documentation
FORK-DELTA.md                          - Fork changes documentation
FINAL_IMPLEMENTATION_SUMMARY.md        - Architecture & design overview
DYNAMIC_SCALING_GUIDE.md               - How all parameters scale
COMPARISON_PLOT_EXPLANATION.md         - Plotting logic & troubleshooting
MULTI_METHOD_SETUP.md                  - Multi-method setup guide
RUN_RESULTS_2024_09_07.md             - Complete session results
REPOSITORY_STRUCTURE.md                - This file
```

### 📦 Build & Configuration
```
Makefile                   - Build commands
pyproject.toml             - Project configuration
uv.lock                    - Dependency lock file
.gitignore                 - Git ignore rules
.dockerignore              - Docker ignore rules
LICENSE                    - License file
run_train.sh               - Training script
```

### 🗂️ Archive (Backup & Test)
```
archive/
├── README.md                  - Archive documentation
├── run_heloco_backup.py       - Old version 1
├── run_heloco_backup2.py      - Old version 2
├── run_heloco_backup3.py      - Old version 3 (stable copy)
├── test_comparison_plot.py    - Test script
├── test_dynamic_scaling.py    - Test script
├── verify_plot_content.py     - Verification script
└── test_comparison.png        - Test output
```

### 📂 Data & Output Directories
```
outputs/
├── heloco_run/                 - Latest run results
│   ├── comparison_loss.png     - Method comparison plot
│   ├── method-heloco/          - HeLoCo results
│   │   ├── logs/
│   │   └── metrics/
│   └── method-diloco/          - DiLoCo results
│       ├── logs/
│       └── metrics/
```

---

## 📋 File Classifications

### Critical (Must Keep)
- ✅ `run_heloco.py` - Production launcher
- ✅ `heloco.yaml` - Configuration

### Important Documentation
- ✅ `README.md` - Original repo documentation
- ✅ `FINAL_IMPLEMENTATION_SUMMARY.md` - Architecture
- ✅ `DYNAMIC_SCALING_GUIDE.md` - Parameter reference

### Build System (Do Not Delete)
- ✅ `Makefile`
- ✅ `pyproject.toml`
- ✅ `uv.lock`

### Archived (Reference Only)
- 🗂️ `archive/` - All backup and test files

---

## 🚀 Quick Start

1. **View configuration:**
   ```bash
   cat heloco.yaml
   ```

2. **Run with defaults:**
   ```bash
   heloco/bin/python run_heloco.py
   ```

3. **Run with custom parameters:**
   ```bash
   heloco/bin/python run_heloco.py --islands 4 --steps 200
   ```

4. **View results:**
   ```bash
   cat outputs/heloco_run/comparison_loss.png
   ```

---

## 📊 Active Session Documentation

### Latest Run Results
- 📄 `RUN_RESULTS_2024_09_07.md` - Complete results with metrics

### Reference Guides
- 📄 `DYNAMIC_SCALING_GUIDE.md` - All 14 parameters explained
- 📄 `FINAL_IMPLEMENTATION_SUMMARY.md` - Code architecture
- 📄 `COMPARISON_PLOT_EXPLANATION.md` - Plotting details
- 📄 `MULTI_METHOD_SETUP.md` - Multi-method configuration

---

## 🔍 Finding What You Need

| Need | File |
|------|------|
| How to run | `README.md` + `DYNAMIC_SCALING_GUIDE.md` |
| Configuration options | `heloco.yaml` |
| Code structure | `FINAL_IMPLEMENTATION_SUMMARY.md` |
| Parameter scaling | `DYNAMIC_SCALING_GUIDE.md` |
| Plot generation | `COMPARISON_PLOT_EXPLANATION.md` |
| Latest results | `RUN_RESULTS_2024_09_07.md` |
| Old versions | `archive/` |
| Test scripts | `archive/` |

---

## 📈 Before & After Cleanup

### Before (Cluttered)
```
❌ run_heloco_backup.py
❌ run_heloco_backup2.py
❌ run_heloco_backup3.py
❌ test_comparison_plot.py
❌ test_dynamic_scaling.py
❌ verify_plot_content.py
❌ test_comparison.png
+ Multiple documentation files mixed in
+ Result: Confusing root directory
```

### After (Organized)
```
✅ Production files in root
✅ Documentation grouped logically
✅ Archive contains backups & tests
✅ Result: Clean, professional structure
```

---

## ✨ Summary

The repository is now well-organized:

- **Root:** Only essential production files
- **Documentation:** Clear reference guides
- **Archive:** All backup and test files
- **Structure:** Easy to navigate and maintain

Clean workspace ready for collaboration! 🎯
