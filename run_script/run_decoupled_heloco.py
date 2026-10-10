#!/usr/bin/env python
"""Separate experiment launcher; existing methods use the original workflow.

Phase 7 adds dense one-GPU learners for the from-scratch 15M Llama recipe.
Use --check-config to validate without requiring GPUs or training packages.
Use --dry-run to inspect routing and the effective legacy YAML without launch.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from panoengine.decentralized.decoupled_heloco.config import (
    ConfigError,
    DECOUPLED_METHODS,
    LEGACY_METHODS,
    METHODS,
    load_config,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def _load_legacy_launcher():
    """Read its parser only; importing does not run main or start any roles."""
    path = SCRIPT_DIR / "run_heloco.py"
    spec = importlib.util.spec_from_file_location("_decoupled_legacy_launcher", path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"cannot load existing launcher {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_legacy_options(legacy, path: Path):
    # The existing parser takes sys.argv, so scope and restore it. No old file
    # is patched, and the actual run is delegated in a fresh subprocess.
    original = sys.argv
    try:
        sys.argv = [str(SCRIPT_DIR / "run_heloco.py"), "--config-file", str(path)]
        try:
            return legacy.parse_args()
        except (SystemExit, TypeError, ValueError) as exc:
            raise ConfigError(f"invalid run settings: {exc}") from exc
    finally:
        sys.argv = original


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, default=Path(__file__).resolve().parent / "decoupled_heloco.yaml")
    parser.add_argument("--method", choices=METHODS, help="override the YAML method selector")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-config", action="store_true", help="validate only; no launch or GPU query")
    actions.add_argument("--dry-run", action="store_true", help="validate and print routing only; no GPU query")
    actions.add_argument("--smoke-test", action="store_true", help="run the synthetic CPU fragment simulation; no real recipe/network")
    actions.add_argument("--process-smoke-test", action="store_true", help="run separate CPU learner processes over localhost TCP")
    actions.add_argument("--check-training", action="store_true", help="check CUDA allocation, recipe dependencies, tokenizer and fragment sizes without launching learners")
    parser.add_argument("--smoke-ticks", type=int, default=30, help="simulation ticks for --smoke-test (default: 30)")
    parser.add_argument("--smoke-outer", choices=("heloco", "diloco", "sgd"), help="CPU simulation outer override (default: selected method)")
    parser.add_argument("--process-learners", type=int, default=2, help="CPU process demo learners (default: 2; must satisfy min_quorum)")
    parser.add_argument("--process-cycles", type=int, default=2, help="complete fragment cycles in process demo (default: 2)")
    parser.add_argument("--process-timeout", type=float, default=60.0, help="total process demo deadline in seconds (default: 60)")
    parser.add_argument("--training-timeout", type=float, default=1800.0, help="total GPU run deadline, including startup and final drain (default: 1800)")
    parser.add_argument("--output-dir", type=Path, help="new explicit GPU run directory; refuses to overwrite")
    args = parser.parse_args(argv)

    try:
        import yaml

        config = load_config(args.config_file, args.method)
        legacy = _load_legacy_launcher()
        with tempfile.TemporaryDirectory(prefix="decoupled-heloco-") as directory:
            effective = Path(directory) / "legacy.yaml"
            effective.write_text(yaml.safe_dump(config.legacy_options(), sort_keys=False), encoding="utf-8")
            resolved = _parse_legacy_options(legacy, effective)
            config.validate_run(resolved)
            if args.output_dir is not None and config.method not in DECOUPLED_METHODS:
                raise ConfigError("--output-dir requires a decoupled GPU method")

            print(f"Configuration valid: method={config.method}", flush=True)
            if args.check_config:
                return 0
            if args.process_smoke_test:
                if config.method not in DECOUPLED_METHODS:
                    raise ConfigError("--process-smoke-test requires a decoupled method")
                if args.process_learners < config.decoupled.min_quorum or args.process_cycles < 1:
                    raise ConfigError("process learners must satisfy min_quorum and process cycles must be positive")
                from panoengine.decentralized.decoupled_heloco.process_smoke import run_process_smoke

                return run_process_smoke(config, learners=args.process_learners, cycles=args.process_cycles, timeout=args.process_timeout)
            if args.smoke_test:
                if config.method not in DECOUPLED_METHODS:
                    raise ConfigError("--smoke-test requires a decoupled method")
                if args.smoke_ticks < 1:
                    raise ConfigError("--smoke-ticks must be positive")
                from panoengine.decentralized.decoupled_heloco.smoke import run_smoke

                return run_smoke(config, ticks=args.smoke_ticks, outer_method=args.smoke_outer or config.fragment_outer_method)
            if config.method in DECOUPLED_METHODS:
                if args.dry_run:
                    from panoengine.decentralized.decoupled_heloco.gpu_recipe import validate_training_options

                    validate_training_options(config, resolved)
                    print("Route: dense GPU learners + CPU syncer over localhost TCP")
                    print("Recipe: models.llama3_small / llama3_15m; one GPU per learner; IID shards; FP32/FlexAttention/plain AdamW")
                    print(f"Outer optimizer: {config.fragment_outer_method}; HeLoCo controls apply only to decoupled_heloco.")
                    print(yaml.safe_dump({"decoupled": asdict(config.decoupled)}, sort_keys=False), end="")
                    print("Use --check-training to check CUDA, dependencies, tokenizer, and fragment frame sizes.")
                    return 0
                from panoengine.decentralized.decoupled_heloco.gpu_training import run_gpu_training

                output = {"output_dir": args.output_dir} if args.output_dir is not None else {}
                return run_gpu_training(config, resolved, REPO_ROOT, timeout=args.training_timeout, check_only=args.check_training, **output)

            assert config.method in LEGACY_METHODS
            if args.check_training:
                raise ConfigError("--check-training requires a decoupled method")
            command = [sys.executable, str(SCRIPT_DIR / "run_heloco.py"), "--config-file", str(effective)]
            if args.dry_run:
                print("Route: existing run_heloco.py (unchanged)")
                print(f"Command (temporary config exists only during this invocation): {shlex.join(command)}")
                print("Effective YAML passed to the existing launcher:")
                print(effective.read_text(encoding="utf-8"), end="")
                print("Hardware, assets, and token-budget calculation are checked during an actual run.")
                return 0
            # Relative assets and outputs keep the original repo-root semantics.
            return subprocess.call(command, cwd=REPO_ROOT)
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Use the project heloco environment; GPU training also needs its pinned TorchTitan/TorchFT training dependencies.", file=sys.stderr)
        return 2
    except (ConfigError, OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
