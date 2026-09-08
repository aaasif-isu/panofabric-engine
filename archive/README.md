# Archive Directory

This directory contains backup files and test scripts from the development and testing phases.

## Contents

### Backup Files (Old Versions)
- `run_heloco_backup.py` - First backup version of the launcher
- `run_heloco_backup2.py` - Second backup version of the launcher
- `run_heloco_backup3.py` - Third backup version (stable copy before final session)

**Note:** The current production version is `../run_heloco.py`

### Test & Verification Scripts
- `test_comparison_plot.py` - Test script for comparison plot generation
- `test_dynamic_scaling.py` - Test script for dynamic parameter scaling
- `verify_plot_content.py` - Verification script for plot content validation

### Test Outputs
- `test_comparison.png` - Output from comparison plot testing

## Purpose

These files are kept for reference and debugging purposes, but are not required for normal operation of the launcher.

## Using Production Code

To run the multi-method launcher, use:
```bash
heloco/bin/python ../run_heloco.py
```

Not the backup files. The production version is `../run_heloco.py`.
