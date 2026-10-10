#!/usr/bin/env python3
"""Regenerate measured central resources; never invent zero syncer memory."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "run_script"))
from run_method_comparison import generate_memory_plot

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    args = parser.parse_args()
    report=json.loads((args.run_dir/"comparison.json").read_text())
    generate_memory_plot(args.run_dir,report["runs"])

if __name__ == "__main__":
    main()
