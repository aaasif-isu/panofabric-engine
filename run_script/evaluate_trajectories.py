#!/usr/bin/env python
"""Evaluate saved trajectory checkpoints on identical held-out batches."""

import argparse
from pathlib import Path
import json
import secrets
import sys
from datetime import datetime, timezone

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from panoengine.decentralized.decoupled_heloco.config import ConfigError, load_config
from panoengine.decentralized.decoupled_heloco.evaluation import evaluate_model, freeze_batches, load_validation_cache, parameter_fingerprint, validation_cache
from panoengine.decentralized.decoupled_heloco.gpu_recipe import build_initial_model, build_recipe, validate_tokenizer
from panoengine.decentralized.decoupled_heloco.global_validation import load_run, check_compatible_runs, tokenizer_fingerprint

def run_trajectory_validation(run_dirs, repo_root, *, batches=100, device="cuda:0", output_dir=None, validation_cache_path=None):
    if type(batches) is not int or not 1 <= batches <= 10000:
        raise ConfigError("--batches must be in 1..10000")
    if not run_dirs:
        raise ConfigError("supply at least one completed run directory")
    runs = [load_run(folder) for folder in run_dirs]
    check_compatible_runs(runs)
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    from torchtitan.distributed.utils import set_spmd_backend
    from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader

    set_spmd_backend("partial_dtensor")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output = Path(output_dir) if output_dir is not None else Path(repo_root) / "outputs/trajectory_evaluation" / f"{stamp}-{secrets.token_hex(3)}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "starting", "dataset": "c4_validation", "split": "validation",
              "device": str(device), "precision": "float32", "matmul_precision": "highest",
              "metric_definitions": {"loss": "sum of cross-entropies / valid target tokens",
                                     "perplexity": "exp(loss)", "next_token_accuracy": "correct top-1 predictions / valid target tokens"},
              "trajectories": {}}
    print(f"Trajectory evaluation output: {output}", flush=True)
    try:
        options = runs[0]["options"]
        recipe = build_recipe(options, repo_root, output)
        tokenizer = recipe.tokenizer.build(tokenizer_path=recipe.hf_assets_path)
        vocab_size = recipe.model_spec.model.vocab_size
        validate_tokenizer(tokenizer, vocab_size)
        tokenizer_sha = tokenizer_fingerprint(tokenizer, recipe.hf_assets_path)
        model = build_initial_model(recipe, options.seed)
        initial_sha = parameter_fingerprint(dict(model.named_parameters()))
        initial_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
        
        cache_metadata = {"dataset": "c4_validation", "split": "validation", "seq_len": options.seq_len,
                          "batch_size": options.batch, "batch_count": batches, "tokenizer_sha256": tokenizer_sha}
        if validation_cache_path is not None:
            cache_path = Path(validation_cache_path).resolve()
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            frozen = load_validation_cache(cache, cache_metadata, vocab_size=vocab_size)
            print(f"Reusing {batches} frozen held-out batches", flush=True)
        else:
            print(f"Loading {batches} held-out C4 validation batches", flush=True)
            loader_config = HuggingFaceTextDataLoader.Config(dataset="c4_validation", infinite=False, num_workers=0)
            loader = loader_config.build(dp_world_size=1, dp_rank=0, tokenizer=tokenizer,
                                         seq_len=options.seq_len, local_batch_size=options.batch)
            frozen = freeze_batches(loader, batches, seq_len=options.seq_len, batch_size=options.batch, vocab_size=vocab_size)
            cache = validation_cache(frozen, cache_metadata)
            cache_path = output / "validation_batches.pt"
            torch.save(cache, cache_path)
            
        report.update(model={"module": options.module, "config": options.config, "seed": options.seed},
                      validation={**cache_metadata, "batches_sha256": cache["batches_sha256"], "cache_path": str(cache_path)},
                      initial_parameters_sha256=initial_sha)
        model.to(device=device)

        def measure(label):
            def progress(done, total):
                if done == total:
                    pass
            metrics = evaluate_model(model, frozen, device=device, vocab_size=vocab_size, progress=progress)
            ppl = "overflow" if metrics["perplexity"] is None else f"{metrics['perplexity']:.3f}"
            print(f"{label}: loss={metrics['loss']:.6f}, perplexity={ppl}, tokens={metrics['valid_tokens']}", flush=True)
            return metrics

        for run in runs:
            method_label = run["folder"].name
            report["trajectories"][method_label] = []
            
            traj_dir = run["folder"] / "trajectory"
            if not traj_dir.is_dir():
                raise RuntimeError(f"Missing trajectory directory for {method_label}")
                
            checkpoints = sorted(traj_dir.glob("*.pt"))
            if not checkpoints:
                raise RuntimeError(f"Trajectory directory is empty for {method_label}")
                
            if not any("initial" in p.name or "00000000" in p.name for p in checkpoints):
                raise RuntimeError(f"Missing initial checkpoint for {method_label}")
            if not any("final" in p.name for p in checkpoints):
                raise RuntimeError(f"Missing final checkpoint for {method_label}")
                
            interval = run["config"].monitoring.get("global_every_updates", 10)
            sync_updates = run["summary"].get("sync_updates", 0)
            expected_steps = set(range(interval, sync_updates, interval))
            found_steps = set()
                
            print(f"\nEvaluating {len(checkpoints)} trajectory points for {method_label}...", flush=True)
            for ckpt_path in checkpoints:
                payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                found_steps.add(payload["step"])
                
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        p.copy_(payload["parameters"][name])
                
                metrics = measure(f"{method_label} @ step {payload['step']}")
                
                point = {
                    "step": payload["step"],
                    "processed_tokens": payload["processed_tokens"],
                    "elapsed_s": payload["elapsed_s"],
                    "snapshot_event": payload.get("snapshot_event", "interval"),
                    **metrics
                }
                report["trajectories"][method_label].append(point)
                
            missing = expected_steps - found_steps
            if missing:
                raise RuntimeError(f"Missing expected intermediate checkpoints for {method_label} at steps: {sorted(missing)}")
                
            with torch.no_grad():
                for name, p in model.named_parameters():
                    p.copy_(initial_parameters[name])

        report["status"] = "passed"
        with (output / "trajectory_evaluation.json").open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        return 0
    finally:
        if report["status"] != "passed":
            report["status"] = "failed"
            try:
                with (output / "trajectory_evaluation.json").open("w", encoding="utf-8") as stream:
                    json.dump(report, stream, indent=2, allow_nan=False)
            except Exception:
                pass

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True, help="completed run folders")
    parser.add_argument("--batches", type=int, default=100, help="held-out batches")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        return run_trajectory_validation(args.run_dirs, REPO_ROOT, batches=args.batches, device=args.device,
                              output_dir=args.output_dir, validation_cache_path=args.validation_cache)
    except Exception as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())
