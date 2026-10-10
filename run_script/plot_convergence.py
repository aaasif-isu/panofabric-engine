#!/usr/bin/env python3
import csv
import os
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not installed. Install with: pip install matplotlib")
    sys.exit(1)

def get_latest_run_dir(base_path):
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=os.path.getmtime)

def read_csv(path):
    steps = []
    losses = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            losses.append(float(row['loss']))
    return steps, losses

def main():
    # Because user ran from run_script/, the outputs are inside run_script/outputs
    base_dir = Path("run_script/outputs/method_comparison")
    if not base_dir.exists():
        # Fallback to root outputs if run directly
        base_dir = Path("outputs/method_comparison")
        
    latest_run = get_latest_run_dir(base_dir)
    
    if not latest_run:
        print(f"No run found in {base_dir}")
        return
        
    print(f"Using run directory: {latest_run}")
    
    methods = ["mla", "diloco", "heloco", "decoupled_diloco", "decoupled_heloco"]
    
    plt.figure(figsize=(10, 6))
    
    for method in methods:
        csv_path = latest_run / method / "learner_0" / "steps.csv"
        if csv_path.exists():
            steps, losses = read_csv(csv_path)
            plt.plot(steps, losses, label=method, linewidth=2)
        else:
            print(f"Warning: Data not found for {method} at {csv_path}")

    plt.xlabel("Local Training Step (Round)", fontsize=12)
    plt.ylabel("Training Loss (Convergence proxy for accuracy)", fontsize=12)
    plt.title("Convergence Comparison of Distributed Training Methods", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    out_file = "convergence_plot.png"
    main()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"\nConvergence plot saved to {out_file}")

if __name__ == "__main__":
    main()