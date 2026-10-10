#!/usr/bin/env python
"""Prepare a matched configuration from a completed run's recorded settings."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from run_decoupled_heloco import _load_legacy_launcher, _parse_legacy_options
from panoengine.decentralized.decoupled_heloco.config import ConfigError, ExperimentConfig, METHODS, load_config


MATCHED_RUN_FIELDS = (
    "islands", "gpus_per_island", "gpus", "module", "config", "hf_assets",
    "dataset", "seq_len", "batch", "seed", "data_distribution", "languages",
    "island_slowness_factors", "coordination_method", "async_interval",
    "max_wait_time", "sync_steps", "num_fragments", "outer_lr", "outer_momentum",
    "should_quantize", "correction_workers", "correction_scope",
    "correction_heatmap", "other_islands_method", "extra",
)


def prepare_comparison(reference_run, output_config, *, method="decoupled_diloco"):
    folder, destination = Path(reference_run), Path(output_config)
    if method not in METHODS:
        raise ConfigError(f"method must be one of {METHODS}")
    if destination.exists():
        raise FileExistsError(destination)
    required = [folder / name for name in ("experiment.yaml", "run-options.json", "summary.json", "global_model.pt")]
    if any(not path.is_file() for path in required):
        raise ConfigError("reference run must contain experiment.yaml, run-options.json, summary.json and global_model.pt")
    source = load_config(required[0])
    saved = json.loads(required[1].read_text())
    summary = json.loads(required[2].read_text())
    if summary.get("status") != "passed" or summary.get("degraded", False):
        raise ConfigError("reference must be a successful, non-degraded run")
    budget = summary.get("steps_per_learner")
    if type(budget) is not int or budget <= 0 or budget != saved.get("steps"):
        raise ConfigError("reference run budget disagrees with its resolved settings")
    with tempfile.TemporaryDirectory() as directory:
        parsed = Path(directory) / "legacy.yaml"
        parsed.write_text(yaml.safe_dump(source.legacy_options()))
        original = _parse_legacy_options(_load_legacy_launcher(), parsed)
    for field in MATCHED_RUN_FIELDS:
        if field not in saved or saved[field] != getattr(original, field):
            raise ConfigError(f"recorded run settings disagree with experiment.yaml: {field}")
    if original.tokens_per_parameter is None and original.steps != budget:
        raise ConfigError("reference run budget disagrees with experiment.yaml")
    run = {**source.run, **{field: saved[field] for field in MATCHED_RUN_FIELDS},
           "steps": budget, "tokens_per_parameter": None, "log_dir": f"outputs/{method}"}
    target = ExperimentConfig(method, run=run, decoupled=source.decoupled)
    with tempfile.TemporaryDirectory() as directory:
        parsed = Path(directory) / "legacy.yaml"
        parsed.write_text(yaml.safe_dump(target.legacy_options()))
        target.validate_run(_parse_legacy_options(_load_legacy_launcher(), parsed))
    # Exclusive creation protects both the reference and an existing output.
    with destination.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(asdict(target), stream, sort_keys=False)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, default=Path("decoupled_diloco.yaml"))
    parser.add_argument("--method", choices=METHODS, default="decoupled_diloco")
    args = parser.parse_args(argv)
    try:
        prepare_comparison(args.reference_run, args.output_config, method=args.method)
        print(f"Prepared matched configuration: {args.output_config}")
        return 0
    except (ConfigError, OSError, ValueError) as exc:
        print(f"configuration preparation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
