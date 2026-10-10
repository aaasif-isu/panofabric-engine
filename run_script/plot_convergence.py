#!/usr/bin/env python3
"""Regenerate global/learner validation curves from a measured comparison."""
import argparse
import json
from pathlib import Path
from run_method_comparison import generate_convergence_plot

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run_dir / "comparison.json").read_text())
    generate_convergence_plot(args.run_dir, report["runs"])

if __name__ == "__main__":
    main()
