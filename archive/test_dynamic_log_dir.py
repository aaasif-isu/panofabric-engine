#!/usr/bin/env python3
"""Test script for dynamic log directory naming."""

import argparse
import sys
from pathlib import Path

def build_dynamic_log_dir(base_log_dir: Path, args: argparse.Namespace) -> Path:
    """Build a dynamic log directory name based on configuration parameters."""
    config_name = args.config.lower().replace(" ", "_")
    dynamic_name = (
        f"heloco_run_{args.coordination_method}_"
        f"{config_name}_{args.data_distribution}_{args.steps}"
    )
    
    if str(base_log_dir).endswith("heloco_run"):
        dynamic_dir = base_log_dir.parent / dynamic_name
    else:
        dynamic_dir = base_log_dir / dynamic_name
    
    return dynamic_dir


def test_cases():
    """Test various parameter combinations."""
    test_data = [
        # (log_dir, coordination_method, config, data_distribution, steps, expected)
        ("outputs/heloco_run", "sync", "qwen3_0_6b", "iid", 100, 
         "outputs/heloco_run_sync_qwen3_0_6b_iid_100"),
        
        ("outputs/heloco_run", "async", "qwen3_0_6b", "non_iid", 50,
         "outputs/heloco_run_async_qwen3_0_6b_non_iid_50"),
        
        ("outputs/heloco_run", "sync", "llama3_15m", "iid", 75,
         "outputs/heloco_run_sync_llama3_15m_iid_75"),
        
        ("custom_dir", "sync", "qwen3_0_6b", "iid", 100,
         "custom_dir/heloco_run_sync_qwen3_0_6b_iid_100"),
        
        ("outputs/heloco_run", "sync", "Llama3_8B", "both", 200,
         "outputs/heloco_run_sync_llama3_8b_both_200"),
    ]
    
    passed = 0
    failed = 0
    
    for log_dir, coord_method, config, dist, steps, expected in test_data:
        args = argparse.Namespace(
            coordination_method=coord_method,
            config=config,
            data_distribution=dist,
            steps=steps
        )
        
        result = build_dynamic_log_dir(Path(log_dir), args)
        result_str = str(result)
        
        if result_str == expected:
            print(f"✅ PASS: {log_dir} → {result_str}")
            passed += 1
        else:
            print(f"❌ FAIL: {log_dir}")
            print(f"   Expected: {expected}")
            print(f"   Got:      {result_str}")
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"Results: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(test_cases())

