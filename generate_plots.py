#!/usr/bin/env python3
"""Generate PNG plots from heloco metrics CSVs."""

import csv
import math
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not installed. Install with:")
    print("  pip install matplotlib")
    sys.exit(1)


def read_csv(path):
    """Read CSV file into list of dicts."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric columns
            for key in list(row.keys()):
                try:
                    row[key] = float(row[key])
                except (ValueError, TypeError):
                    pass
            rows.append(row)
    return rows


def plot_island(steps_data, png_path):
    """Plot loss, ppl, grad_norm, lr for one island."""
    if not steps_data:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(Path(png_path).name[:-4], fontsize=14, fontweight="bold")
    
    steps = [s['step'] for s in steps_data]
    losses = [s['loss'] for s in steps_data]
    grad_norms = [s['grad_norm'] for s in steps_data]
    lrs = [s['lr'] for s in steps_data]
    ppls = [math.exp(l) for l in losses]
    
    # Loss
    axes[0, 0].plot(steps, losses, "b-", linewidth=1.5)
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title("Loss")
    
    # Perplexity
    axes[0, 1].plot(steps, ppls, "g-", linewidth=1.5)
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Perplexity")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title("Perplexity")
    
    # Grad norm (log)
    axes[1, 0].semilogy(steps, grad_norms, "r-", linewidth=1.5)
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Gradient Norm (log scale)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title("Gradient Norm")
    
    # Learning rate
    axes[1, 1].plot(steps, lrs, "m-", linewidth=1.5)
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title("Learning Rate")
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {png_path.name}")


def main():
    """Generate all plots from existing CSVs."""
    metrics_dir = Path("outputs/heloco_run/metrics")
    if not metrics_dir.exists():
        print(f"ERROR: {metrics_dir} not found")
        sys.exit(1)
    
    print(f"\nGenerating plots from {metrics_dir}...\n")
    
    # Find all island CSV files
    for csv_file in sorted(metrics_dir.glob("island-*_steps.csv")):
        island_num = csv_file.name.split("-")[1]
        png_file = metrics_dir / f"island-{island_num}_loss.png"
        
        print(f"Processing {csv_file.name}...")
        steps_data = read_csv(csv_file)
        plot_island(steps_data, png_file)
    
    print(f"\n✓ All plots generated in {metrics_dir}/")
    print(f"  Files: island-*_loss.png")


if __name__ == "__main__":
    main()
