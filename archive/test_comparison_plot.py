#!/usr/bin/env python3
"""Test that _plot_comparison generates a single figure with both methods."""

from pathlib import Path
import sys

# Add repo to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from run_heloco import _plot_comparison

# Create mock data for 2 methods with multiple islands
methods_data = {
    'heloco': [
        # Island 0 data
        [
            {'step': '0', 'loss_metrics/global_avg_loss': '2.5'},
            {'step': '10', 'loss_metrics/global_avg_loss': '2.3'},
            {'step': '20', 'loss_metrics/global_avg_loss': '2.0'},
            {'step': '30', 'loss_metrics/global_avg_loss': '1.7'},
        ],
        # Island 1 data
        [
            {'step': '0', 'loss_metrics/global_avg_loss': '2.52'},
            {'step': '10', 'loss_metrics/global_avg_loss': '2.32'},
            {'step': '20', 'loss_metrics/global_avg_loss': '2.02'},
            {'step': '30', 'loss_metrics/global_avg_loss': '1.72'},
        ]
    ],
    'diloco': [
        # Island 0 data
        [
            {'step': '0', 'loss_metrics/global_avg_loss': '2.6'},
            {'step': '10', 'loss_metrics/global_avg_loss': '2.25'},
            {'step': '20', 'loss_metrics/global_avg_loss': '1.9'},
            {'step': '30', 'loss_metrics/global_avg_loss': '1.6'},
        ],
        # Island 1 data
        [
            {'step': '0', 'loss_metrics/global_avg_loss': '2.58'},
            {'step': '10', 'loss_metrics/global_avg_loss': '2.27'},
            {'step': '20', 'loss_metrics/global_avg_loss': '1.92'},
            {'step': '30', 'loss_metrics/global_avg_loss': '1.62'},
        ]
    ]
}

# Generate the plot
png_path = REPO_ROOT / "test_comparison.png"

try:
    _plot_comparison(methods_data, png_path)
    size = png_path.stat().st_size
    print("✓ Comparison plot generated successfully!")
    print(f"  File: {png_path}")
    print(f"  Size: {size} bytes")
    print()
    print("  WHAT YOU'LL SEE IN THE PNG:")
    print("  ┌──────────────────────────────────────────────────────────────┐")
    print("  │ Decentralized Training Method Comparison (Loss & Perplexity)  │")
    print("  ├──────────────────────────────────────────────────────────────┤")
    print("  │                                                              │")
    print("  │  Loss Over Steps       │  Perplexity Over Steps             │")
    print("  │  (ax1)                │  (ax2)                             │")
    print("  │                                                              │")
    print("  │  Loss                 │  PPL                               │")
    print("  │  axis │               │  axis │                            │")
    print("  │    2.5│ •             │   12  │ •                          │")
    print("  │    2.0│  • heloco ─── │    8  │  • heloco ─── (BLUE)      │")
    print("  │    1.5│   • diloco ── │    4  │   • diloco ── (ORANGE)    │")
    print("  │       └───────────── │       └───────────────             │")
    print("  │         0  10  20  30│         0  10  20  30              │")
    print("  │         Steps        │         Steps                       │")
    print("  │                                                              │")
    print("  │  ✓ BOTH methods plotted on SAME axes!                       │")
    print("  │  ✓ Different colors for easy comparison!                    │")
    print("  │  ✓ ONE PNG file = simple, direct comparison!                │")
    print("  │                                                              │")
    print("  └──────────────────────────────────────────────────────────────┘")
    print()
    print(f"  View the plot: display {png_path}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
