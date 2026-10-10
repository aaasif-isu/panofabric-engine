#!/usr/bin/env python
"""One YAML, sequential matched training, one shared global evaluation."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time

import yaml

from run_decoupled_heloco import _load_legacy_launcher, _parse_legacy_options
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
from panoengine.decentralized.decoupled_heloco.baseline_options import validate_baseline_options
from panoengine.decentralized.decoupled_heloco.config import ConfigError, DECOUPLED_METHODS, METHODS, load_config
from panoengine.decentralized.decoupled_heloco.gpu_recipe import validate_dense_options, validate_training_options


def positive_seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{name} must be finite and positive")
    return float(value)


def load_comparison(path):
    """Validate EVERY method before creating outputs or importing Torch."""
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    if not isinstance(data, dict) or data.keys() - {"method_run", "run", "decoupled", "evaluation"}:
        raise ConfigError("comparison YAML accepts method_run, run, decoupled, and evaluation sections")
    methods = data.get("method_run")
    if not isinstance(methods, list) or not methods or any(not isinstance(m, str) or m not in METHODS for m in methods):
        raise ConfigError(f"method_run must be a nonempty list drawn from {METHODS}")
    if len(set(methods)) != len(methods):
        raise ConfigError("method_run contains duplicate methods")
    evaluation = data.get("evaluation", {})
    if not isinstance(evaluation, dict) or evaluation.keys() - {"enabled", "batches", "device", "validation_cache", "timeout"}:
        raise ConfigError("invalid evaluation section")
    evaluation = {"enabled": True, "batches": 100, "device": "cuda:0", "validation_cache": None,
                  "timeout": 900.0, **evaluation}
    if type(evaluation["enabled"]) is not bool:
        raise ConfigError("evaluation.enabled must be true or false")
    if type(evaluation["batches"]) is not int or not 1 <= evaluation["batches"] <= 10000:
        raise ConfigError("evaluation.batches must be in 1..10000")
    if not isinstance(evaluation["device"], str) or re.fullmatch(r"cuda:[0-9]+", evaluation["device"]) is None:
        raise ConfigError("evaluation.device must be a logical CUDA device such as cuda:0")
    if evaluation["validation_cache"] is not None and (not isinstance(evaluation["validation_cache"], str) or not evaluation["validation_cache"].strip()):
        raise ConfigError("evaluation.validation_cache must be a path or null")
    positive_seconds(evaluation["timeout"], "evaluation.timeout")
    configs, options = [], []
    legacy = _load_legacy_launcher()
    with tempfile.TemporaryDirectory(prefix="five-methods-validate-") as directory:
        effective = Path(directory) / "effective.yaml"
        parsed = Path(directory) / "legacy.yaml"
        for method in methods:
            effective.write_text(yaml.safe_dump({"method": method, "run": data.get("run", {}), "decoupled": data.get("decoupled", {})}))
            config = load_config(effective)
            parsed.write_text(yaml.safe_dump(config.legacy_options()))
            resolved = _parse_legacy_options(legacy, parsed)
            config.validate_run(resolved)
            validate_dense_options(resolved)
            if resolved.tokens_per_parameter is not None:
                raise ConfigError("comparison uses a fixed shared run.steps budget; set tokens_per_parameter: null")
            if method in DECOUPLED_METHODS:
                validate_training_options(config, resolved)
                if resolved.steps < config.decoupled.overlap_steps:
                    raise ConfigError("run.steps must be at least decoupled.overlap_steps")
            else:
                validate_baseline_options(config, resolved)
            configs.append(config)
            options.append(resolved)
    return {"methods": methods, "configs": configs, "options": options,
            "evaluation": evaluation, "source": content,
            "source_sha256": hashlib.sha256(content.encode()).hexdigest()}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def stop_process_group(process):
    """Stop only the fresh process session created for this method."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait(timeout=2.0)
    # A failed coordinator may leave learners after it has already exited.
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_command(command, log_path, repo_root, timeout):
    """Stream child output while enforcing a wall-clock process deadline."""
    process = subprocess.Popen(command, cwd=repo_root, env=dict(os.environ, PYTHONUNBUFFERED="1"),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               start_new_session=True)
    lines = queue.Queue()

    def read_output():
        try:
            for line in process.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout
    try:
        with Path(log_path).open("w", encoding="utf-8") as stream:
            eof = False
            while not eof or process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"process deadline exceeded; inspect {log_path}")
                try:
                    line = lines.get(timeout=0.1)
                except queue.Empty:
                    continue
                if line is None:
                    eof = True
                else:
                    stream.write(line)
                    stream.flush()
                    print(line, end="", flush=True)
            status = process.wait()
            if status:
                raise RuntimeError(f"command failed with exit code {status}; inspect {log_path}")
    except BaseException:
        stop_process_group(process)
        raise
    finally:
        reader.join(timeout=1.0)
        process.stdout.close()


def completed_run(folder, config, options):
    """Require actual budgets, method and initialization metadata, not stdout."""
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text())
    if summary.get("status") != "passed" or summary.get("method") != config.method:
        raise RuntimeError(f"{config.method} did not produce a matching passed summary")
    if not (folder / "global_model.pt").is_file():
        raise RuntimeError(f"{config.method} did not export global_model.pt")
    learners = summary.get("learners", [])
    if (summary.get("steps_per_learner") != options.steps or len(learners) != options.islands
            or [r.get("learner_id") for r in learners] != list(range(options.islands))
            or any(r.get("total_local_steps") != options.steps or type(r.get("total_tokens")) is not int or r["total_tokens"] < 1 for r in learners)):
        raise RuntimeError(f"{config.method} did not complete the shared local training budget")
    initial = summary.get("initial_parameters_sha256")
    if not isinstance(initial, str) or re.fullmatch(r"[0-9a-f]{64}", initial) is None:
        raise RuntimeError(f"{config.method} lacks a valid initial-model fingerprint")
    return summary


def comparison_rows(report, manifest):
    if report.get("status") != "passed":
        raise RuntimeError("shared global evaluation did not finish successfully")
    models = report.get("models", [])
    globals_list = [row for row in models if row.get("kind") == "global"]
    globals_by_method = {row.get("method"): row for row in globals_list}
    if len(globals_list) != len(manifest["methods"]) or len(globals_by_method) != len(manifest["methods"]) or set(globals_by_method) != set(manifest["methods"]):
        raise RuntimeError("evaluation methods differ from the configured method_run")
    initial = [r for r in models if r.get("kind") == "initial"]
    if len(initial) != 1:
        raise RuntimeError("evaluation requires one shared initial reference")
    rows = [{"method": "initial", **{k: initial[0][k] for k in ("loss", "perplexity", "next_token_accuracy", "valid_tokens")}}]
    for method in manifest["methods"]:
        result = globals_by_method[method]
        run = manifest["runs"][method]
        if Path(result["run_dir"]).resolve() != Path(run["folder"]).resolve() or result.get("initial_fingerprint_verified") is not True:
            raise RuntimeError(f"evaluation did not verify the recorded {method} run")
        summary = run["summary"]
        rows.append({"method": method, **{k: result[k] for k in ("loss", "perplexity", "next_token_accuracy", "valid_tokens")},
                     "processed_tokens": sum(r["total_tokens"] for r in summary["learners"]),
                     "steps_per_learner": summary["steps_per_learner"], "sync_updates": summary["sync_updates"],
                     "training_elapsed_s": summary["elapsed_s"], "run_dir": run["folder"],
                     "checkpoint": result["checkpoint"]})
    return rows

def generate_convergence_plot(folder, runs):
    try:
        import matplotlib.pyplot as plt
        import csv
    except ImportError:
        print("Skipping convergence plot: matplotlib is not installed. Run 'pip install matplotlib' in your environment to enable it.", flush=True)
        return

    plt.figure(figsize=(10, 6))
    plotted = False

    for method, run_data in runs.items():
        if run_data.get("status") != "passed":
            continue
        
        run_dir = Path(run_data["folder"])
        csv_path = run_dir / "learner_0" / "steps.csv"
        
        if csv_path.exists():
            steps, losses = [], []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    steps.append(int(row['step']))
                    losses.append(float(row['loss']))
            if steps:
                plt.plot(steps, losses, label=method, linewidth=2)
                plotted = True
        else:
            print(f"Warning: Data not found for {method} at {csv_path}", flush=True)

    if plotted:
        plt.xlabel("Local Training Step (Round)", fontsize=12)
        plt.ylabel("Training Loss (Convergence)", fontsize=12)
        plt.title("Convergence Comparison of Distributed Training Methods", fontsize=14, fontweight="bold")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        out_file = folder / "convergence_plot.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        print(f"Convergence plot saved to: {out_file}", flush=True)
    else:
        print("No valid steps.csv found to generate convergence plot.", flush=True)
    plt.close()

def generate_memory_plot(folder, runs):
    try:
        import torch
        import glob
        import json
        import csv
    except ImportError as e:
        print(f"Skipping memory footprint: required packages not installed: {e}", flush=True)
        return

    param_count = None
    for method, run_data in runs.items():
        if run_data.get("status") == "passed":
            model_path = Path(run_data["folder"]) / "global_model.pt"
            if model_path.exists():
                try:
                    checkpoint = torch.load(model_path, map_location="cpu")
                    parameters = checkpoint.get("parameters", {})
                    param_count = sum(p.numel() for p in parameters.values())
                    break
                except Exception:
                    pass
    
    if not param_count:
        print("No valid global_model.pt found to calculate memory footprint.", flush=True)
        return

    bytes_per_param = 4
    base_model_size_mb = (param_count * bytes_per_param) / (1024 ** 2)
    
    # The CPU-based Server or Coordinator process maintains the FP32 model and outer FP32 momentum,
    # plus staging buffers for aggregation or dispatch. Hence, theoretical CPU memory is ~3x model size.
    ps_multiplier = 3
    estimated_cpu_memory_mb = base_model_size_mb * ps_multiplier

    methods = []
    gpu_memory = {}
    csv_rows = []

    for method, run_data in runs.items():
        if run_data.get("status") != "passed":
            continue
            
        methods.append(method)
        m_max = 0
        run_dir = Path(run_data["folder"])
        log_files = glob.glob(str(run_dir / "**/*.log"), recursive=True)
        for f in log_files:
            try:
                for l in open(f):
                    if "PFMETRICS" in l:
                        data = json.loads(l.split("PFMETRICS ")[1])
                        m_max = max(m_max, data.get("memory/max_reserved(GiB)", 0))
            except Exception:
                pass
                
        gpu_memory_mb = m_max * 1024  # Convert GiB to MB
        gpu_memory[method] = gpu_memory_mb
        
        csv_rows.append({
            "Method": method,
            "Central_CPU_Memory_MB": round(estimated_cpu_memory_mb, 2),
            "CPU_Memory_Type": "Theoretical Estimate (~3x model size)",
            "Peak_Learner_GPU_Memory_MB": round(gpu_memory_mb, 2),
            "GPU_Memory_Type": "Empirical Measurement (max_reserved from logs)",
            "GPU_Source_Logs": len(log_files)
        })

    if not methods:
        return

    # Write CSV
    csv_path = folder / "memory_footprint.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "Central_CPU_Memory_MB", "CPU_Memory_Type", "Peak_Learner_GPU_Memory_MB", "GPU_Memory_Type", "GPU_Source_Logs"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Memory numerical results saved to: {csv_path}", flush=True)

    # Plot
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"Skipping memory plot visualization: matplotlib not installed: {e}", flush=True)
        return

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    
    cpu_vals = [estimated_cpu_memory_mb for _ in methods]
    gpu_vals = [gpu_memory[m] for m in methods]
    
    rects1 = ax.bar(x - width/2, cpu_vals, width, label='Estimated Central Syncer (CPU)', color='salmon')
    rects2 = ax.bar(x + width/2, gpu_vals, width, label='Measured Peak Learner (GPU)', color='skyblue')
    
    ax.set_ylabel('Memory (MB)', fontsize=12)
    model_m = param_count / 1e6
    ax.set_title(f"Memory Footprint by Method ({model_m:.1f}M Params)", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in rects1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}", ha='center', va='bottom', fontweight='bold', fontsize=9)
    for bar in rects2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}", ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.figtext(0.5, 0.01, 
             "Note: CPU memory is a theoretical estimate (~3x model size) as all methods currently use an in-process syncer.\n"
             "GPU memory is empirically measured via max_reserved(GiB) in PyTorch across all learner nodes.", 
             ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.subplots_adjust(bottom=0.15)
    out_file = folder / "parameter_server_memory.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Memory plot saved to: {out_file}", flush=True)
    plt.close()



def run_comparison(comparison, repo_root, *, training_timeout=1800.0, output_dir=None, command_runner=run_command):
    training_timeout = positive_seconds(training_timeout, "--training-timeout")
    repo_root = Path(repo_root).resolve()
    cache = comparison["evaluation"]["validation_cache"]
    cache_path = (repo_root / cache).resolve() if cache is not None else None
    if comparison["evaluation"]["enabled"] and cache_path is not None and not cache_path.is_file():
        raise ConfigError(f"validation cache does not exist: {cache_path}")
    if output_dir is None:
        parent = repo_root / comparison["options"][0].log_dir
        folder = parent / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(3))
    else:
        folder = repo_root / output_dir
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    configs_dir = folder / "configs"
    configs_dir.mkdir()
    (folder / "comparison.yaml").write_text(comparison["source"], encoding="utf-8")
    manifest = {"status": "starting", "methods": list(comparison["methods"]),
                "config_sha256": comparison["source_sha256"], "runs": {},
                "evaluation_enabled": comparison["evaluation"]["enabled"], "output_dir": str(folder)}
    print(f"Comparison output: {folder}", flush=True)
    started = time.monotonic()
    try:
        initial_sha = None
        for index, (config, options) in enumerate(zip(comparison["configs"], comparison["options"]), 1):
            method = config.method
            run_dir = folder / method
            effective = configs_dir / f"{method}.yaml"
            effective.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
            launcher = "run_decoupled_heloco.py" if method in DECOUPLED_METHODS else "run_matched_baselines.py"
            command = [sys.executable, str(SCRIPT_DIR / launcher), "--config-file", str(effective),
                       "--output-dir", str(run_dir), "--training-timeout", str(training_timeout)]
            manifest.update(status="training", current_method=method)
            manifest["runs"][method] = {"status": "running", "folder": str(run_dir)}
            write_json(folder / "comparison.json", manifest)
            print(f"\nMethod {index}/{len(comparison['methods'])}: {method}", flush=True)
            # The child's own deadline includes startup/drain. Allow a small
            # margin to publish diagnostics before terminating its session.
            command_runner(command, folder / f"{method}.log", repo_root, training_timeout + 30.0)
            summary = completed_run(run_dir, config, options)
            if initial_sha is not None and summary["initial_parameters_sha256"] != initial_sha:
                raise RuntimeError("methods started from different initial parameters")
            initial_sha = summary["initial_parameters_sha256"]
            manifest["runs"][method].update(status="passed", summary=summary)
            write_json(folder / "comparison.json", manifest)
        manifest["initial_parameters_sha256"] = initial_sha
        manifest.pop("current_method", None)
        evaluation = comparison["evaluation"]
        if evaluation["enabled"]:
            manifest["status"] = "evaluating"
            write_json(folder / "comparison.json", manifest)
            evaluation_dir = folder / "evaluation"
            command = [sys.executable, str(SCRIPT_DIR / "evaluate_decoupled_heloco.py"), "--run-dirs",
                       *[manifest["runs"][m]["folder"] for m in comparison["methods"]],
                       "--batches", str(evaluation["batches"]), "--device", evaluation["device"], "--output-dir", str(evaluation_dir)]
            if cache_path is not None:
                command += ["--validation-cache", str(cache_path)]
            print("\nEvaluating all global models on one frozen held-out stream", flush=True)
            command_runner(command, folder / "evaluation.log", repo_root, evaluation["timeout"])
            report = json.loads((evaluation_dir / "evaluation.json").read_text())
            rows = comparison_rows(report, manifest)
            fields = ("method", "loss", "perplexity", "next_token_accuracy", "valid_tokens", "processed_tokens",
                      "steps_per_learner", "sync_updates", "training_elapsed_s", "run_dir", "checkpoint")
            with (folder / "comparison.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            manifest.update(status="passed", evaluation=report, results=rows)
            print(f"\nComparison metrics: {folder / 'comparison.csv'}", flush=True)
        else:
            manifest["status"] = "trained_only"
        manifest["elapsed_s"] = time.monotonic() - started
        print(f"COMPARISON {'PASSED' if evaluation['enabled'] else 'TRAINING COMPLETED'}: {folder / 'comparison.json'}", flush=True)
        generate_convergence_plot(folder, manifest.get("runs", {}))
        generate_memory_plot(folder, manifest.get("runs", {}))
        return 0
    except KeyboardInterrupt:
        manifest.update(status="interrupted", error="interrupted by user")
        return 130
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        if manifest.get("current_method") in manifest["runs"]:
            manifest["runs"][manifest["current_method"]].update(status="failed", error=str(exc))
        print(f"Comparison failed: {exc}\nSaved runs and logs: {folder}", file=sys.stderr, flush=True)
        return 2
    finally:
        write_json(folder / "comparison.json", manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, default=Path(__file__).resolve().parent / "method_comparison.yaml")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-config", action="store_true", help="validate all methods; no Torch/CUDA imports or processes")
    actions.add_argument("--dry-run", action="store_true", help="print sequential routing without training")
    parser.add_argument("--training-timeout", type=float, default=1800.0, help="deadline PER method, including startup/drain")
    parser.add_argument("--output-dir", type=Path, help="new comparison directory; refuses to overwrite")
    args = parser.parse_args(argv)
    try:
        positive_seconds(args.training_timeout, "--training-timeout")
        comparison = load_comparison(args.config_file)
        print(f"Configuration valid: method_run={comparison['methods']}", flush=True)
        if args.check_config or args.dry_run:
            if args.dry_run:
                for config in comparison["configs"]:
                    launcher = "run_decoupled_heloco.py" if config.method in DECOUPLED_METHODS else "run_matched_baselines.py"
                    print(f"{config.method}: {launcher}")
                print(f"Shared held-out evaluation: {comparison['evaluation']}")
            return 0
        return run_comparison(comparison, REPO_ROOT, training_timeout=args.training_timeout, output_dir=args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ImportError, yaml.YAMLError) as exc:
        print(f"Comparison configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
