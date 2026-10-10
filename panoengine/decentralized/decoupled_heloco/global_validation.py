"""Single-GPU held-out validation of completed dense 15M global exports."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
from types import SimpleNamespace

import torch

from .config import ConfigError, DECOUPLED_METHODS, load_config
from .baseline_options import BASELINE_SCOPE, validate_baseline_options
from .evaluation import (evaluate_model, freeze_batches, load_global_parameters,
                         load_validation_cache, parameter_fingerprint, validation_cache)
from .gpu_recipe import build_initial_model, build_recipe, validate_dense_options, validate_tokenizer, validate_training_options


def load_run(folder):
    folder = Path(folder).resolve()
    config = load_config(folder / "experiment.yaml")
    options = SimpleNamespace(**json.loads((folder / "run-options.json").read_text()))
    summary = json.loads((folder / "summary.json").read_text())
    if summary.get("status") != "passed":
        raise ConfigError(f"run did not finish successfully: {folder}")
    if summary.get("method", config.method) != config.method:
        raise ConfigError("run summary method differs from the saved experiment")
    if config.method in DECOUPLED_METHODS:
        validate_training_options(config, options)
    else:
        if summary.get("scope") != BASELINE_SCOPE:
            raise ConfigError("legacy evaluation requires a completed run from run_matched_baselines.py")
        validate_baseline_options(config, options)
        validate_dense_options(options)
    for name in ("seed", "seq_len", "batch"):
        value = getattr(options, name, None)
        if type(value) is not int or (name != "seed" and value < 1):
            raise ConfigError(f"saved run option {name} is invalid")
    for name in ("module", "config", "seed", "seq_len", "batch", "hf_assets"):
        if name in config.run and config.run[name] != getattr(options, name):
            raise ConfigError(f"saved experiment and run-options disagree on {name}")
    checkpoint = folder / "global_model.pt"
    if not checkpoint.is_file():
        raise ConfigError(f"global checkpoint does not exist: {checkpoint}")
    return {"folder": folder, "config": config, "options": options, "summary": summary,
            "checkpoint": checkpoint}


def check_compatible_runs(runs):
    first = runs[0]["options"]
    for run in runs[1:]:
        for name in ("module", "config", "seed", "seq_len", "batch"):
            if getattr(run["options"], name) != getattr(first, name):
                raise ConfigError(f"shared evaluation requires matching {name}; {run['folder']} differs")
    if len({str(r["folder"]) for r in runs}) != len(runs):
        raise ConfigError("the same run directory was supplied more than once")


def fragment_count(run):
    return run["config"].decoupled.num_fragments if run["config"].method in DECOUPLED_METHODS else 1


def file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def tokenizer_fingerprint(tokenizer, assets):
    data = {"vocabulary": tokenizer.get_vocab(), "bos_id": getattr(tokenizer, "bos_id", None),
            "eos_id": getattr(tokenizer, "eos_id", None), "files": {}}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                 "vocab.json", "merges.txt", "tokenizer.model", "sentencepiece.model"):
        path = Path(assets) / name
        if path.is_file():
            data["files"][name] = file_fingerprint(path)
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def residual_work(summary):
    """Per-fragment work overlaps: deliberately provide no summed tail total."""
    return [{"learner_id": learner["learner_id"], "fragments": [
        {"fragment_id": f["fragment_id"], "local_steps": f["local_steps"], "tokens": f["tokens"]}
        for f in learner["fragments"]]} for learner in summary["learners"]]


def run_validation(run_dirs, repo_root, *, batches=100, device="cuda:0", output_dir=None, validation_cache_path=None):
    if type(batches) is not int or not 1 <= batches <= 10000:
        raise ConfigError("--batches must be in 1..10000")
    if not run_dirs:
        raise ConfigError("supply at least one completed run directory")
    runs = [load_run(folder) for folder in run_dirs]
    check_compatible_runs(runs)
    device = torch.device(device)
    if device.type != "cuda" or device.index is None or not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
        raise ConfigError("evaluation requires a visible GPU; use --device cuda:0 inside your GPU allocation")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    from torchtitan.distributed.utils import set_spmd_backend
    from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader

    set_spmd_backend("partial_dtensor")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output = Path(output_dir) if output_dir is not None else Path(repo_root) / "outputs/decoupled_heloco_evaluation" / f"{stamp}-{secrets.token_hex(3)}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "dataset": "c4_validation", "split": "validation",
              "device": str(device), "precision": "float32", "matmul_precision": "highest",
              "metric_definitions": {"loss": "sum of cross-entropies / valid target tokens",
                                     "perplexity": "exp(loss)", "next_token_accuracy": "correct top-1 predictions / valid target tokens"},
              "models": []}
    print(f"Evaluation output: {output}", flush=True)
    try:
        options = runs[0]["options"]
        recipe = build_recipe(options, repo_root, output)
        tokenizer = recipe.tokenizer.build(tokenizer_path=recipe.hf_assets_path)
        vocab_size = recipe.model_spec.model.vocab_size
        validate_tokenizer(tokenizer, vocab_size)
        tokenizer_sha = tokenizer_fingerprint(tokenizer, recipe.hf_assets_path)
        # Historical exports did not fingerprint tokenizer assets. Current
        # paths must contain the original training tokenizer.
        for run in runs[1:]:
            other_assets = (Path(repo_root) / run["options"].hf_assets).resolve()
            other_tokenizer = recipe.tokenizer.build(tokenizer_path=str(other_assets))
            if tokenizer_fingerprint(other_tokenizer, other_assets) != tokenizer_sha:
                raise ConfigError("runs use different tokenizer assets")
        model = build_initial_model(recipe, options.seed)
        initial_sha = parameter_fingerprint(dict(model.named_parameters()))
        initial_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
        for run in runs:
            expected = run["summary"].get("initial_parameters_sha256")
            if expected is not None and initial_sha != expected:
                raise ConfigError("reconstructed initial model differs from the actual training initialization")
        if any("initial_parameters_sha256" not in r["summary"] for r in runs):
            print("Historical run: initial reference reconstructed from its saved seed; keep the original model code and tokenizer assets.", flush=True)
        cache_metadata = {"dataset": "c4_validation", "split": "validation", "seq_len": options.seq_len,
                          "batch_size": options.batch, "batch_count": batches, "tokenizer_sha256": tokenizer_sha}
        if validation_cache_path is not None:
            cache_path = Path(validation_cache_path).resolve()
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            frozen = load_validation_cache(cache, cache_metadata, vocab_size=vocab_size)
            print(f"Reusing {batches} frozen held-out batches", flush=True)
        else:
            print(f"Loading {batches} held-out C4 validation batches (first use may download data)", flush=True)
            loader_config = HuggingFaceTextDataLoader.Config(dataset="c4_validation", infinite=False, num_workers=0)
            loader = loader_config.build(dp_world_size=1, dp_rank=0, tokenizer=tokenizer,
                                         seq_len=options.seq_len, local_batch_size=options.batch)
            frozen = freeze_batches(loader, batches, seq_len=options.seq_len, batch_size=options.batch, vocab_size=vocab_size)
            cache = validation_cache(frozen, cache_metadata)
            cache_path = output / "validation_batches.pt"
            torch.save(cache, cache_path)
        report.update(model={"module": options.module, "config": options.config, "seed": options.seed},
                      validation={**cache_metadata, "batches_sha256": cache["batches_sha256"], "cache_path": str(cache_path)},
                      initial_parameters_sha256=initial_sha,
                      initial_reference="reconstructed using the syncer CPU initializer and saved seed")
        model.to(device=device)  # Keep complex RoPE buffers complex.
        for run in runs:
            # Check all exports before any expensive evaluation kernel work.
            payload = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
            load_global_parameters(model, payload, num_fragments=fragment_count(run),
                                   expected_revisions=run["summary"]["fragment_revisions"], expected_method=run["config"].method)
        # Restore the initial reference after the prevalidation loop.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(initial_parameters[name])
        del initial_parameters

        def measure(label):
            def progress(done, total):
                if done == 1 or done % 20 == 0 or done == total:
                    print(f"{label}: batch {done}/{total}", flush=True)
            metrics = evaluate_model(model, frozen, device=device, vocab_size=vocab_size, progress=progress)
            ppl = "overflow" if metrics["perplexity"] is None else f"{metrics['perplexity']:.3f}"
            print(f"{label}: loss={metrics['loss']:.6f}, perplexity={ppl}, next_token_accuracy={metrics['next_token_accuracy']:.4%}, tokens={metrics['valid_tokens']}", flush=True)
            return metrics

        report["models"].append({"label": "initial", "kind": "initial", "checkpoint": None, **measure("initial")})
        for run in runs:
            payload = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
            load_global_parameters(model, payload, num_fragments=fragment_count(run),
                                   expected_revisions=run["summary"]["fragment_revisions"], expected_method=run["config"].method)
            label = run["folder"].name
            report["models"].append({"label": label, "kind": "global", "method": run["config"].method,
                                    "run_dir": str(run["folder"]), "checkpoint": str(run["checkpoint"]),
                                    "checkpoint_sha256": file_fingerprint(run["checkpoint"]),
                                    "initial_fingerprint_verified": run["summary"].get("initial_parameters_sha256") == initial_sha,
                                    "stopping_policy": run["summary"].get("stopping_policy", "fixed_local_budgets_drain_to_first_unavailable_round_robin_quorum"),
                                    "outer_optimizer": run["summary"].get("outer_optimizer"),
                                    "local_optimizer_steps": [r["total_local_steps"] for r in run["summary"]["learners"]] if all("total_local_steps" in r for r in run["summary"]["learners"]) else None,
                                    "processed_tokens": sum(r["total_tokens"] for r in run["summary"]["learners"]) if all("total_tokens" in r for r in run["summary"]["learners"]) else None,
                                    "sync_updates": run["summary"].get("sync_updates"),
                                    "fragment_revisions": list(payload["fragment_revisions"]),
                                    "unmerged_work": residual_work(run["summary"]), **measure(label)})
        report["status"] = "passed"
        with (output / "evaluation.csv").open("w", newline="") as stream:
            fields = ("label", "kind", "loss", "perplexity", "next_token_accuracy", "valid_tokens", "batches", "checkpoint")
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(report["models"])
        print("GLOBAL VALIDATION COMPLETED: identical held-out batches; no optimizer steps.", flush=True)
        print(f"Results: {output / 'evaluation.json'}", flush=True)
        return 0
    except KeyboardInterrupt:
        report.update(status="interrupted", error="interrupted by user")
        raise
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
