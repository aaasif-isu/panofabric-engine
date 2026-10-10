#!/usr/bin/env python
"""Train original async outer methods with the shared dense FP32 recipe."""

import argparse
from pathlib import Path
import sys
import tempfile

import yaml

from run_decoupled_heloco import _load_legacy_launcher, _parse_legacy_options
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
from panoengine.decentralized.decoupled_heloco.baseline_options import baseline_metadata, validate_baseline_options
from panoengine.decentralized.decoupled_heloco.config import ConfigError, load_config
from panoengine.decentralized.decoupled_heloco.gpu_recipe import validate_dense_options


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-config", action="store_true", help="validate without Torch/CUDA imports or launching processes")
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--check-training", action="store_true", help="CUDA, recipe and tokenizer preflight only")
    parser.add_argument("--training-timeout", type=float, default=1800.0)
    parser.add_argument("--output-dir", type=Path, help="new explicit run directory; refuses to overwrite")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config_file)
        with tempfile.TemporaryDirectory(prefix="matched-baseline-") as directory:
            path = Path(directory) / "legacy.yaml"
            path.write_text(yaml.safe_dump(config.legacy_options()))
            options = _parse_legacy_options(_load_legacy_launcher(), path)
        validate_baseline_options(config, options)
        validate_dense_options(options)
        print(f"Configuration valid: matched method={config.method}", flush=True)
        if args.check_config or args.dry_run:
            if args.dry_run:
                print(yaml.safe_dump({"recipe": "dense FP32 Llama 15M / plain AdamW", "outer_optimizer": baseline_metadata(config, options)}, sort_keys=False))
            return 0
        from panoengine.decentralized.decoupled_heloco.baseline_training import run_baseline_training
        output = {"output_dir": args.output_dir} if args.output_dir is not None else {}
        return run_baseline_training(config, options, REPO_ROOT, timeout=args.training_timeout, check_only=args.check_training, **output)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ImportError, yaml.YAMLError) as exc:
        print(f"Matched baseline error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
