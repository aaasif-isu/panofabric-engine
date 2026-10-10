#!/usr/bin/env python
"""Evaluate completed dense 15M global exports on identical held-out batches."""

import argparse
from pathlib import Path
import pickle
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True, help="completed run folders containing global_model.pt and saved run metadata")
    parser.add_argument("--batches", type=int, default=100, help="held-out batches, 1..10000 (default: 100)")
    parser.add_argument("--device", default="cuda:0", help="one logical GPU within the current allocation (default: cuda:0)")
    parser.add_argument("--validation-cache", type=Path, help="reuse validation_batches.pt from an earlier evaluation with matching settings")
    parser.add_argument("--output-dir", type=Path, help="new output directory; must not already exist")
    args = parser.parse_args(argv)
    if not 1 <= args.batches <= 10000:
        print("Evaluation error: --batches must be in 1..10000", file=sys.stderr)
        return 2
    try:
        from panoengine.decentralized.decoupled_heloco.global_validation import run_validation
        return run_validation(args.run_dirs, REPO_ROOT, batches=args.batches, device=args.device,
                              output_dir=args.output_dir, validation_cache_path=args.validation_cache)
    except ImportError as exc:
        print(f"Missing evaluation dependency: {exc}. Use the project's heloco environment with its pinned TorchTitan dependencies.", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError, pickle.UnpicklingError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Evaluation interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
