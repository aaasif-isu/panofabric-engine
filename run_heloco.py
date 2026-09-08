#!/usr/bin/env python
"""One-command local HeLoCo run: lighthouse + parameter server + N trainer islands.

This is the single-node simulation of decentralized training. Every role that
would normally live in its own shell (or on its own cluster) is spawned here as
a subprocess and wired together over loopback:

    torchft_lighthouse                              (rendezvous / FT quorum)
    python -m panoengine.decentralized.parameter_server --outer_method heloco
    torchrun -m torchtitan.train ... --fault_tolerance.semi_sync_method=heloco   x N

The GPUs are carved into ``--islands`` replicas of ``--gpus-per-island`` GPUs
each via CUDA_VISIBLE_DEVICES. Each island is exactly what ``run_train.sh``
would launch; the parameter server's addresses are read from its stdout and
exported to the trainers as $DILOCO_SERVER_ADDR / $DILOCO_HB_ADDR (the
contract torchtitan's FT manager reads for semi_sync_method='heloco').

Usage (from the repo root, with the project venv's python):

    python run_heloco.py --config-file heloco.yaml       # everything from the YAML
    python run_heloco.py --config-file heloco.yaml --steps 5   # CLI flags override it
    python run_heloco.py --islands 4 --gpus-per-island 1 # no file: defaults + flags
    python run_heloco.py --config-file heloco.yaml --dry-run   # print commands only

Logs: every role streams to the console with a [role] prefix and to
outputs/heloco_run/<role>.log. Ctrl-C tears everything down.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
BIN_DIR = PYTHON.parent  # the venv's bin/: torchrun + torchft_lighthouse live here

_ADDR_RE = re.compile(r"^(DILOCO_SERVER_ADDR|DILOCO_HB_ADDR)=(\S+)\s*$")


# --------------------------------------------------------------------------- args
def _load_config_file(path: str) -> dict:
    """Read a YAML config file into a flat dict of {dest_name: value}.

    Keys use the CLI option names with underscores (``gpus_per_island``,
    ``sync_steps``, ...). ``gpus`` may be a list or a comma-separated string;
    ``extra`` must be a list of torchtitan flags.
    """
    import yaml  # torchtitan dependency, always present in the venv

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: top level must be a mapping of option: value")
    out: dict = {}
    for key, value in data.items():
        dest = str(key).replace("-", "_")
        if dest == "gpus" and isinstance(value, list):
            value = ",".join(str(g) for g in value)
        if dest == "extra" and value is not None and not isinstance(value, list):
            raise SystemExit(f"{path}: 'extra' must be a list of strings")
        out[dest] = value
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    default_cfg = REPO_ROOT / "heloco.yaml"
    p.add_argument("--config-file",
                   default=str(default_cfg) if default_cfg.exists() else None,
                   help="YAML file whose keys are these options (underscored, e.g. "
                        "gpus_per_island). CLI flags given explicitly override it. "
                        "Defaults to ./heloco.yaml when that file exists; pass "
                        "--config-file '' to ignore it.")
    topo = p.add_argument_group("topology")
    topo.add_argument("--islands", type=int, default=2,
                      help="number of trainer replicas (HeLoCo workers)")
    topo.add_argument("--gpus-per-island", type=int, default=2,
                      help="GPUs per replica (FSDP-sharded inside the island)")
    topo.add_argument("--gpus", type=str, default=None,
                      help="comma-separated GPU ids to use (default: 0..islands*gpus_per_island-1)")

    recipe = p.add_argument_group("recipe")
    recipe.add_argument("--module", default="models.qwen3",
                        help="recipe package with a config_registry")
    recipe.add_argument("--config", default="qwen3_0_6b",
                        help="preset function inside the recipe's config_registry")
    recipe.add_argument("--hf-assets", default="./assets/hf/Qwen3-0.6B",
                        help="tokenizer (and optionally weights) directory")
    recipe.add_argument("--dataset", default="c4",
                        help="torchtitan dataset name (c4 streams from HF Hub)")
    recipe.add_argument("--steps", type=int, default=30, help="inner training steps per island")
    recipe.add_argument("--seq-len", type=int, default=1024)
    recipe.add_argument("--batch", type=int, default=1, help="local batch size per rank")
    recipe.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="anything after --extra is appended verbatim to every torchrun")

    async_grp = p.add_argument_group("coordination (async trainer Path A)")
    async_grp.add_argument("--coordination-method", choices=["sync", "async"], default="sync",
                           help="sync (torchft barrier) or async (HTTP push/pull). (default: sync)")
    async_grp.add_argument("--async-interval", type=int, default=1,
                           help="check staleness every N window pulses (async only). (default: 1)")
    async_grp.add_argument("--max-wait-time", type=float, default=0.0,
                           help="max seconds to wait for PS response; 0.0 = no timeout. (default: 0.0)")

    algo = p.add_argument_group("heloco / diloco / mla")
    algo.add_argument("--sync-steps", type=int, default=10,
                      help="window length H: inner steps between pushes to the PS")
    algo.add_argument("--num-fragments", type=int, default=1,
                      help="fragment-wise sync (must divide --sync-steps); 1 = whole model")
    algo.add_argument("--outer-lr", type=float, default=0.7)
    algo.add_argument("--outer-momentum", type=float, default=0.9)
    algo.add_argument("--rho", type=float, default=None,
                      help="HeLoCo arrival weight (default: 1/sqrt(islands))")
    algo.add_argument("--outer-method", choices=["heloco", "diloco", "mla"], default="heloco",
                      help="server-side outer optimizer (diloco = plain async DiLoCo; "
                           "trainers use the same 'heloco' wire path either way)")
    algo.add_argument("--should-quantize", action="store_true",
                      help="int8 pseudo-gradient upload (set on both ends automatically)")

    net = p.add_argument_group("ports / misc")
    net.add_argument("--host", default="127.0.0.1")
    net.add_argument("--lighthouse-port", type=int, default=29510)
    net.add_argument("--ps-port", type=int, default=29520)
    net.add_argument("--rdzv-base-port", type=int, default=29600,
                     help="island i rendezvous on rdzv_base_port + i")
    net.add_argument("--log-dir", default="outputs/heloco_run")
    net.add_argument("--ps-timeout", type=float, default=600.0,
                     help="seconds to wait for the parameter server to print its address")
    net.add_argument("--dry-run", action="store_true", help="print commands, launch nothing")
    net.add_argument("--methods", default="heloco",
                     help="comma-separated methods to run sequentially: heloco, diloco, etc. "
                          "(default: heloco). Runs each and produces a comparison_loss.png.")
    
    data = p.add_argument_group("data distribution")
    data.add_argument("--data-distribution", default="iid", choices=["iid", "non_iid", "both"],
                      help="iid (all islands same language), non_iid (each island different), "
                           "or both (run IID then non_iid sequentially)")
    data.add_argument("--languages", default=None, nargs="+",
                      help="space-separated languages per island (for non_iid/both modes). "
                           "Example: --languages english french german")

    # Hidden option: only for heloco.yaml config file (no CLI argument exposed)
    p.add_argument("--island-slowness-factors", default=None, nargs="+", type=float, 
                   help=argparse.SUPPRESS)  # Hidden from help

    args = p.parse_args()
    
    # Load config file first (before parsing methods)
    if args.config_file:
        file_values = _load_config_file(args.config_file)
        unknown = [k for k in file_values if not hasattr(args, k) or k == "config_file"]
        if unknown:
            raise SystemExit(f"{args.config_file}: unknown option(s) {unknown}")
        # Precedence: script defaults < config file < flags typed on the CLI.
        # An option counts as "typed" if its value differs from the parser default.
        defaults = {a.dest: a.default for a in p._actions}
        for key, value in file_values.items():
            if getattr(args, key) == defaults.get(key):
                setattr(args, key, value)

    # Parse methods: allow both comma-separated string and YAML lists
    if isinstance(args.methods, str):
        args.methods = [m.strip() for m in args.methods.split(",")]
    elif isinstance(args.methods, list):
        pass  # already a list from YAML
    else:
        args.methods = [str(args.methods)]
    
    # Parse and validate languages for non-IID data distribution
    if args.languages is None:
        args.languages = []
    elif isinstance(args.languages, str):
        args.languages = [lang.strip() for lang in args.languages.split(",")]
    elif isinstance(args.languages, list):
        args.languages = [str(lang).strip() for lang in args.languages]
    
    # Validate data distribution configuration
    if args.data_distribution in ("non_iid", "both"):
        if not args.languages:
            raise SystemExit(
                f"--data-distribution {args.data_distribution} requires --languages to be specified. "
                f"Provide one language per island. Available: english, french, german, spanish, italian, "
                f"portuguese, romanian, dutch, greek, czech, polish, hungarian, croatian, swedish, finnish, "
                f"danish, bulgarian, lithuanian, slovene, slovak, irish, maltese"
            )
        if len(args.languages) != args.islands:
            raise SystemExit(
                f"Number of languages ({len(args.languages)}) must match number of islands ({args.islands}). "
                f"Provided: {args.languages}"
            )
    
    # Parse and validate island slowness factors for heterogeneity simulation (from config file)
    if args.island_slowness_factors is None:
        args.island_slowness_factors = [1.0] * args.islands  # Default: all islands same speed
    else:
        args.island_slowness_factors = [float(f) for f in args.island_slowness_factors]
        if len(args.island_slowness_factors) != args.islands:
            raise SystemExit(
                f"Number of slowness factors in heloco.yaml ({len(args.island_slowness_factors)}) must match "
                f"number of islands ({args.islands}). "
                f"Provided: {args.island_slowness_factors}"
            )
        # Validate that all factors are positive
        if any(f <= 0 for f in args.island_slowness_factors):
            raise SystemExit(
                f"All slowness factors in heloco.yaml must be positive (> 0). Got: {args.island_slowness_factors}"
            )

    return args



# ------------------------------------------------------------------ preflight
def preflight(args: argparse.Namespace) -> list[int]:
    problems: list[str] = []

    for tool in ("torchrun", "torchft_lighthouse"):
        if not (BIN_DIR / tool).exists() and shutil.which(tool) is None:
            problems.append(f"'{tool}' not found in {BIN_DIR} or on PATH -- run with the "
                            f"project venv's python (e.g. heloco/bin/python run_heloco.py)")

    if args.gpus:
        gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    else:
        gpus = list(range(args.islands * args.gpus_per_island))
    needed = args.islands * args.gpus_per_island
    if len(gpus) < needed:
        problems.append(f"need {needed} GPUs ({args.islands} islands x {args.gpus_per_island}), "
                        f"but only {len(gpus)} listed: {gpus}")
    try:
        visible = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                                 timeout=30).stdout.count("GPU ")
        if visible < needed:
            problems.append(f"nvidia-smi reports {visible} GPU(s); need {needed}")
    except Exception as exc:  # noqa: BLE001 - nvidia-smi absent is itself the message
        problems.append(f"could not query nvidia-smi ({exc})")

    assets = Path(args.hf_assets)
    if not assets.is_dir() or not any(assets.glob("tokenizer*")):
        problems.append(
            f"no tokenizer in {assets}. Fetch one, e.g.:\n"
            f"    {BIN_DIR}/hf download Qwen/Qwen3-0.6B tokenizer.json tokenizer_config.json "
            f"vocab.json merges.txt config.json --local-dir {assets}"
        )

    if args.sync_steps % args.num_fragments != 0:
        problems.append(f"--sync-steps ({args.sync_steps}) must be divisible by "
                        f"--num-fragments ({args.num_fragments})")
    if args.num_fragments > 1 and args.outer_method != "heloco":
        problems.append("--num-fragments > 1 requires --outer-method heloco")

    if problems:
        print("preflight failed:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        sys.exit(2)
    return gpus[:needed]


# ------------------------------------------------------------------- commands
def lighthouse_cmd(args) -> list[str]:
    return [
        str(BIN_DIR / "torchft_lighthouse"),
        f"--bind={args.host}:{args.lighthouse_port}",
        "--min_replicas", "1",
        "--quorum_tick_ms", "100",
        "--join_timeout_ms", "10000",
    ]


def ps_cmd(args) -> list[str]:
    rho = args.rho if args.rho is not None else 1.0 / math.sqrt(args.islands)
    cmd = [
        str(PYTHON), "-m", "panoengine.decentralized.parameter_server",
        "--module", args.module,
        "--config", args.config,
        "--hf_assets_path", args.hf_assets,
        "--outer_method", args.outer_method,
        "--port", str(args.ps_port),
        "--lr", str(args.outer_lr),
        "--momentum", str(args.outer_momentum),
    ]
    if args.outer_method == "heloco":
        cmd += ["--rho", f"{rho:.6f}"]
    else:
        # DelayedNesterov milestone: >= number of workers.
        cmd += ["--nesterov_period", str(args.islands)]
    if args.num_fragments > 1:
        cmd += ["--num_fragments", str(args.num_fragments)]
    if args.should_quantize:
        cmd.append("--should_quantize")
    return cmd


def trainer_cmd(args, island: int) -> list[str]:
    # Mirrors run_train.sh. The trainer-side strategy is always 'heloco' (=
    # AsyncDiLoCo talking to the PS); --outer-method only changes the server.
    cmd = [
        str(BIN_DIR / "torchrun"),
        f"--nproc_per_node={args.gpus_per_island}",
        "--nnodes", "1",
        "--rdzv_id", f"heloco-{island}",
        "--rdzv_backend", "c10d",
        f"--rdzv_endpoint={args.host}:{args.rdzv_base_port + island}",
        "--local-ranks-filter", "0",
        "--role", "rank",
        "--tee", "3",
        "-m", "torchtitan.train",
        "--module", args.module,
        "--config", args.config,
        f"--hf_assets_path={args.hf_assets}",
        f"--dump_folder={Path(args.log_dir) / f'island-{island}'}",
        f"--dataloader.dataset={args.dataset}",
        f"--training.steps={args.steps}",
        f"--training.seq_len={args.seq_len}",
        f"--training.local_batch_size={args.batch}",
        "--fault_tolerance.enable",
        f"--fault_tolerance.replica_id={island}",
        f"--fault_tolerance.group_size={args.islands}",
        "--fault_tolerance.semi_sync_method=heloco",
        f"--fault_tolerance.sync_steps={args.sync_steps}",
        f"--fault_tolerance.num_fragments={args.num_fragments}",
        "--fault_tolerance.process_group=gloo",
    ]
    if args.should_quantize:
        cmd.append("--fault_tolerance.should_quantize")
    cmd += args.extra
    return cmd


def trainer_env(args, island: int, gpus: list[int], ps_addr: str, hb_addr: str) -> dict:
    lo = island * args.gpus_per_island
    env = os.environ.copy()
    env_dict = {
        "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpus[lo: lo + args.gpus_per_island]),
        "DILOCO_SERVER_ADDR": ps_addr,
        "DILOCO_HB_ADDR": hb_addr,
        "TORCHFT_LIGHTHOUSE": f"http://{args.host}:{args.lighthouse_port}",
        "GLOO_SOCKET_IFNAME": "lo",
        "NCCL_SOCKET_IFNAME": "lo",
        "PYTHONSAFEPATH": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "LOG_RANK": "0",
    }
    
    # For non-IID data distribution, set language for this island
    if args.data_distribution == "non_iid" and island < len(args.languages):
        env_dict["ISLAND_LANGUAGE"] = args.languages[island]

    # Heterogeneity simulation: per-island artificial slowdown, read by our
    # own AsyncDiLoCo worker (panoengine/decentralized/async_diloco.py) --
    # NOT a torchtitan CLI option, so it must travel as an env var (same
    # pattern as ISLAND_LANGUAGE and PF_WIRE_BF16).
    factors = getattr(args, "island_slowness_factors", None)
    if factors and island < len(factors):
        env_dict["PF_ISLAND_SLOWNESS_FACTOR"] = str(factors[island])

    env.update(env_dict)
    return env


# ------------------------------------------------------------- process plumbing
class Role:
    """A spawned subprocess whose stdout/stderr is tee'd to the console (with a
    [name] prefix) and to a log file, with an optional per-line callback."""

    def __init__(self, name: str, cmd: list[str], env: dict, log_dir: Path,
                 on_line=None):
        self.name = name
        self.cmd = cmd
        self.log_path = log_dir / f"{name}.log"
        self._log = open(self.log_path, "w", buffering=1)
        self._on_line = on_line
        self.proc = subprocess.Popen(
            cmd, env=env, cwd=REPO_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, start_new_session=True,  # own process group -> clean teardown
        )
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._log.write(line)
            sys.stdout.write(f"[{self.name}] {line}")
            sys.stdout.flush()
            if self._on_line is not None:
                self._on_line(line)
        self._log.close()

    def poll(self):
        return self.proc.poll()

    def _signal_group(self, sig: int) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                pass

    def terminate(self) -> None:
        self._signal_group(signal.SIGTERM)

    def kill(self) -> None:
        self._signal_group(signal.SIGKILL)


def shutdown(roles: list[Role], grace: float = 15.0) -> None:
    for r in reversed(roles):  # trainers first, then PS, then lighthouse
        r.terminate()
    deadline = time.time() + grace
    while time.time() < deadline and any(r.poll() is None for r in roles):
        time.sleep(0.2)
    for r in roles:
        r.kill()


def fmt(cmd: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([c]) for c in cmd)


# --------------------------------------------------------------- metrics export
_PFMETRICS_RE = re.compile(r"PFMETRICS (\{.*\})\s*$")

# Console-friendly names for the per-step metrics torchtitan emits.
_STEP_COLUMNS = [
    ("step", "step"),
    ("loss_metrics/global_avg_loss", "loss"),
    ("loss_metrics/global_max_loss", "loss_max"),
    ("grad_norm", "grad_norm"),
    ("lr", "lr"),
    ("throughput(tps)", "tokens_per_s"),
    ("n_tokens_seen", "tokens_seen"),
    ("memory/max_reserved(GiB)", "gpu_mem_gib"),
    ("time_metrics/end_to_end(s)", "step_time_s"),
]
_COMM_COLUMNS = [
    ("step", "step"),
    ("comm/exchange", "exchange"),
    ("comm/bytes_up", "bytes_up"),
    ("comm/bytes_down", "bytes_down"),
    ("comm/seconds", "seconds"),
    ("comm/mbps", "mbps"),
    ("comm/gb_cumulative", "gb_cumulative"),
]


def _parse_pfmetrics(log_path: Path) -> tuple[list[dict], list[dict]]:
    """Split a role log into (per-step training records, HeLoCo comm records)."""
    import json

    steps: list[dict] = []
    comms: list[dict] = []
    with open(log_path, errors="replace") as f:
        for line in f:
            m = _PFMETRICS_RE.search(line)
            if not m:
                continue
            try:
                rec = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if "loss_metrics/global_avg_loss" in rec:
                steps.append(rec)
            elif "comm/exchange" in rec:
                comms.append(rec)
    return steps, comms


def _write_csv(path: Path, rows: list[dict], columns: list[tuple[str, str]]) -> None:
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([name for _, name in columns])
        for r in rows:
            w.writerow([r.get(key, "") for key, _ in columns])


def export_metrics(log_dir: Path, islands: int, data_distribution: str = "iid", 
                   model_name: str = "") -> None:
    """Turn the PFMETRICS lines of every island log into CSVs, plots + a short summary.

    Written to ``<log_dir>/metrics/``:
      island-<i>_steps_{mode}_{model}.csv      one row per training step (loss, grad_norm, lr, ...)
      island-<i>_comm_{mode}_{model}.csv       one row per parameter-server exchange (bytes, seconds)
      island-<i>_loss_{mode}_{model}.png       line plot: loss, perplexity, grad_norm, lr over steps
      summary_{mode}_{model}.txt               first/last loss per island and the exchange count
    
    Args:
        log_dir: Directory containing island logs
        islands: Number of islands
        data_distribution: "iid" or "non_iid" (appended to filenames)
        model_name: Model identifier (e.g., "llama3_15m")
    """
    import math as _math

    out = log_dir / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    
    # Build filename suffix: mode + model
    mode_suffix = f"_iid" if data_distribution == "iid" else "_noniid"
    model_suffix = f"_{model_name}" if model_name else ""
    file_suffix = f"{mode_suffix}{model_suffix}"
    
    lines = [f"{'island':<10}{'steps':>6}{'first_loss':>12}{'last_loss':>11}"
             f"{'min_loss':>10}{'last_ppl':>10}{'exchanges':>11}"]
    for i in range(islands):
        log_path = log_dir / f"island-{i}.log"
        if not log_path.exists():
            continue
        steps, comms = _parse_pfmetrics(log_path)
        _write_csv(out / f"island-{i}_steps{file_suffix}.csv", steps, _STEP_COLUMNS)
        _write_csv(out / f"island-{i}_comm{file_suffix}.csv", comms, _COMM_COLUMNS)
        if not steps:
            lines.append(f"island-{i:<4}{0:>6}{'-':>12}{'-':>11}{'-':>10}{'-':>10}{len(comms):>11}")
            continue
        losses = [s["loss_metrics/global_avg_loss"] for s in steps]
        lines.append(
            f"island-{i:<4}{len(steps):>6}{losses[0]:>12.4f}{losses[-1]:>11.4f}"
            f"{min(losses):>10.4f}{_math.exp(losses[-1]):>10.2f}{len(comms):>11}"
        )
        _plot_island(steps, out / f"island-{i}_loss{file_suffix}.png")
    summary = "\n".join(lines)
    (out / "summary{file_suffix}.txt").write_text(summary + "\n")
    print("\n== metrics summary (loss = cross-entropy, ppl = exp(loss))")
    print(summary)
    print(f"== per-step CSVs & plots saved to: {out}/")


def _plot_island(steps: list[dict], png_path: Path) -> None:
    """Plot loss, perplexity, grad_norm, and learning-rate curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return  # matplotlib not available; skip plots
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{png_path.name[:-4]}", fontsize=14, fontweight="bold")
    
    # Loss
    ax = axes[0, 0]
    s = [st["step"] for st in steps]
    loss = [st["loss_metrics/global_avg_loss"] for st in steps]
    ax.plot(s, loss, "b-", linewidth=1.5, label="loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Perplexity
    ax = axes[0, 1]
    import math
    ppl = [math.exp(l) for l in loss]
    ax.plot(s, ppl, "g-", linewidth=1.5, label="perplexity")
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Grad norm
    ax = axes[1, 0]
    grad = [st.get("grad_norm", 0) for st in steps]
    ax.semilogy(s, grad, "r-", linewidth=1.5, label="grad_norm")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm (log scale)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Learning rate
    ax = axes[1, 1]
    lr = [st.get("lr", 0) for st in steps]
    ax.plot(s, lr, "m-", linewidth=1.5, label="learning_rate")
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=100, bbox_inches="tight")
    plt.close()


def _create_evaluation_log(methods_data: dict[str, list[list[dict]]], log_path: Path, sync_steps: int = None) -> None:
    """Create a centralized log file with evaluation metrics for all methods.
    
    This log contains evaluation checkpoint data (at sync_steps intervals)
    across all methods and islands for easy comparison.
    
    Args:
        methods_data: {method_name: [island_0_steps, island_1_steps, ...]}
        log_path: output log file path
        sync_steps: if provided, only include steps that are multiples of sync_steps
                   (e.g., with sync_steps=10, includes steps 10, 20, 30, ... only)
    """
    try:
        import csv as csv_module
        import math
        from datetime import datetime
    except ImportError:
        return
    
    # Collect all data points (filtered by sync_steps if provided)
    all_records = []
    
    for method_name, islands_steps in methods_data.items():
        for island_idx, island_steps in enumerate(islands_steps):
            for record in island_steps:
                try:
                    step = int(record["step"]) if isinstance(record["step"], str) else record["step"]
                    
                    # Filter by sync_steps: only keep checkpoint steps
                    if sync_steps is not None and step % sync_steps != 0:
                        continue
                    
                    # Try different column names for loss
                    loss_key = "loss_metrics/global_avg_loss" if "loss_metrics/global_avg_loss" in record else "loss"
                    loss = float(record[loss_key]) if isinstance(record[loss_key], str) else record[loss_key]
                    perplexity = math.exp(loss)
                    
                    all_records.append({
                        'method': method_name,
                        'island': island_idx,
                        'step': step,
                        'loss': f"{loss:.6f}",
                        'perplexity': f"{perplexity:.6f}"
                    })
                except (ValueError, KeyError):
                    continue
    
    # Sort by method, island, then step
    all_records.sort(key=lambda x: (x['method'], x['island'], x['step']))
    
    # Write log file
    with open(log_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"CENTRALIZED EVALUATION LOG - Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
        if sync_steps is not None:
            f.write(f"Evaluation Checkpoints (sync_steps={sync_steps})\n")
        f.write(f"Total records: {len(all_records)}\n")
        f.write(f"Methods: {', '.join(sorted(set(r['method'] for r in all_records)))}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write as CSV
        writer = csv_module.DictWriter(f, fieldnames=['method', 'island', 'step', 'loss', 'perplexity'])
        writer.writeheader()
        writer.writerows(all_records)
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY BY METHOD\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary statistics
        for method in sorted(set(r['method'] for r in all_records)):
            method_records = [r for r in all_records if r['method'] == method]
            if method_records:
                first_loss = float(method_records[0]['loss'])
                last_loss = float(method_records[-1]['loss'])
                first_ppl = float(method_records[0]['perplexity'])
                last_ppl = float(method_records[-1]['perplexity'])
                
                f.write(f"Method: {method}\n")
                f.write(f"  Records: {len(method_records)}\n")
                f.write(f"  First Loss: {first_loss:.6f} (PPL: {first_ppl:.6f})\n")
                f.write(f"  Last Loss:  {last_loss:.6f} (PPL: {last_ppl:.6f})\n")
                f.write(f"  Loss improvement: {first_loss - last_loss:.6f}\n")
                f.write(f"  PPL improvement: {first_ppl - last_ppl:.6f}\n")
                f.write("\n")


def _plot_comparison(methods_data: dict[str, list[list[dict]]], png_path: Path) -> None:
    """Plot comparison of loss curves across multiple methods.
    
    Args:
        methods_data: {method_name: [island_0_steps, island_1_steps, ...]}
        png_path: output PNG path
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    import math
    
    # Colors for different methods
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Decentralized Training Method Comparison (Loss & Perplexity)", 
                 fontsize=14, fontweight="bold")
    
    for color_idx, (method, islands_steps) in enumerate(methods_data.items()):
        # Average across islands
        all_steps = []
        for island_steps in islands_steps:
            if island_steps:
                all_steps.extend(island_steps)
        
        if not all_steps:
            continue
        
        # Group by step and average
        from collections import defaultdict
        step_to_losses = defaultdict(list)
        for rec in all_steps:
            step = int(rec["step"]) if isinstance(rec["step"], str) else rec["step"]
            # Try different column names (loss_metrics/global_avg_loss or just loss)
            loss_key = "loss_metrics/global_avg_loss" if "loss_metrics/global_avg_loss" in rec else "loss"
            loss = float(rec[loss_key]) if isinstance(rec[loss_key], str) else rec[loss_key]
            step_to_losses[step].append(loss)
        
        sorted_steps = sorted(step_to_losses.keys())
        avg_losses = [sum(step_to_losses[s]) / len(step_to_losses[s]) for s in sorted_steps]
        color = colors[color_idx % len(colors)]
        
        # Loss
        ax1.plot(sorted_steps, avg_losses, linewidth=1.0, label=method, color=color)
        
        # Perplexity
        ppls = [math.exp(l) for l in avg_losses]
        ax2.plot(sorted_steps, ppls, linewidth=1.0, label=method, color=color)
    
    ax1.set_xlabel("Step", fontsize=11)
    ax1.set_ylabel("Loss (cross-entropy)", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_title("Loss Over Steps")
    
    ax2.set_xlabel("Step", fontsize=11)
    ax2.set_ylabel("Perplexity", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_title("Perplexity Over Steps")
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=100, bbox_inches="tight")
    plt.close()


def print_dry_run(args, gpus: list[int]) -> None:
    ps_addr = f"http://{args.host}:{args.ps_port}/sync"
    hb_addr = f"http://{args.host}:{args.ps_port}/heartbeat"
    print("\n# 1. lighthouse\n" + fmt(lighthouse_cmd(args)))
    print("\n# 2. parameter server (prints DILOCO_SERVER_ADDR / DILOCO_HB_ADDR)\n"
          + fmt(ps_cmd(args)))
    keys = ("CUDA_VISIBLE_DEVICES", "DILOCO_SERVER_ADDR", "DILOCO_HB_ADDR",
            "TORCHFT_LIGHTHOUSE", "GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME")
    for i in range(args.islands):
        env = trainer_env(args, i, gpus, ps_addr, hb_addr)
        print(f"\n# 3.{i} trainer island {i}\n"
              + " ".join(f"{k}={env[k]}" for k in keys)
              + " \\\n  " + fmt(trainer_cmd(args, i)))



# ------------------------------------------------------------------------ main
def run_single_method(method: str, args: argparse.Namespace, method_log_dir: Path, 
                      gpus: list[int]) -> bool:
    """Run a single decentralized training method.
    
    Returns True if successful, False if failed.
    """
    # Update args to reflect the current method
    args_copy = argparse.Namespace(**vars(args))
    args_copy.outer_method = method
    
    print(f"\n{'='*70}")
    print(f"== RUNNING METHOD: {method.upper()}")
    print(f"{'='*70}")
    
    method_log_dir.mkdir(parents=True, exist_ok=True)
    roles: list[Role] = []
    base_env = os.environ.copy()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    
    try:
        # 1. lighthouse
        lighthouse = Role("lighthouse", lighthouse_cmd(args_copy), base_env, method_log_dir)
        roles.append(lighthouse)
        time.sleep(1.0)
        if lighthouse.poll() is not None:
            print(f"== lighthouse exited immediately; see {lighthouse.log_path}", 
                  file=sys.stderr)
            return False
        
        # 2. parameter server
        addrs: dict[str, str] = {}
        got_addrs = threading.Event()
        
        def on_ps_line(line: str) -> None:
            m = _ADDR_RE.match(line.strip())
            if m:
                addrs[m.group(1)] = m.group(2)
                if "DILOCO_SERVER_ADDR" in addrs and "DILOCO_HB_ADDR" in addrs:
                    got_addrs.set()
        
        ps = Role("param-server", ps_cmd(args_copy), base_env, method_log_dir, 
                  on_line=on_ps_line)
        roles.append(ps)
        t0 = time.time()
        while not got_addrs.is_set():
            if stop.is_set():
                return False
            if ps.poll() is not None:
                print(f"== parameter server exited before announcing address; see {ps.log_path}", 
                      file=sys.stderr)
                return False
            if time.time() - t0 > args_copy.ps_timeout:
                print(f"== parameter server did not announce within {args_copy.ps_timeout}s", 
                      file=sys.stderr)
                return False
            time.sleep(0.2)
        
        ps_addr, hb_addr = addrs["DILOCO_SERVER_ADDR"], addrs["DILOCO_HB_ADDR"]
        print(f"== parameter server ready: {ps_addr} (heartbeat {hb_addr})")
        
        # 3. trainer islands
        trainers: list[Role] = []
        for i in range(args_copy.islands):
            env = trainer_env(args_copy, i, gpus, ps_addr, hb_addr)
            r = Role(f"island-{i}", trainer_cmd(args_copy, i), env, method_log_dir)
            roles.append(r)
            trainers.append(r)
            print(f"== island {i} launched on CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
            time.sleep(0.5)
        
        # 4. wait for completion
        while True:
            if stop.is_set():
                print("== interrupted; shutting down")
                return False
            for infra in (lighthouse, ps):
                if infra.poll() is not None:
                    print(f"== {infra.name} died (exit {infra.poll()}); aborting", 
                          file=sys.stderr)
                    return False
            codes = [t.poll() for t in trainers]
            failed = [(t.name, c) for t, c in zip(trainers, codes) if c not in (None, 0)]
            if failed:
                print(f"== island(s) failed: {failed}; aborting", file=sys.stderr)
                return False
            if all(c == 0 for c in codes):
                print(f"== {method.upper()} completed successfully")
                return True
            time.sleep(1.0)
    finally:
        shutdown(roles)
        try:
            export_metrics(method_log_dir, args_copy.islands, args_copy.data_distribution, args_copy.config)
        except Exception as exc:
            print(f"== metrics export failed: {exc}", file=sys.stderr)


def build_dynamic_log_dir(base_log_dir: Path, args: argparse.Namespace) -> Path:
    """Build a dynamic log directory name based on configuration parameters.
    
    Pattern: heloco_run_{coordination_method}_{config}_{data_distribution}_{steps}
    
    Examples:
      - heloco_run_sync_qwen3_0_6b_iid_100
      - heloco_run_async_llama3_15m_non_iid_50
    """
    # Extract config name (e.g., "qwen3_0_6b" from "qwen3_0_6b" or just use config as-is)
    config_name = args.config.lower().replace(" ", "_")
    
    # Build the dynamic name
    dynamic_name = (
        f"heloco_run_{args.coordination_method}_"
        f"{config_name}_{args.data_distribution}_{args.steps}"
    )
    
    # If base_log_dir is just "outputs/heloco_run", replace it; otherwise append
    if str(base_log_dir).endswith("heloco_run"):
        dynamic_dir = base_log_dir.parent / dynamic_name
    else:
        dynamic_dir = base_log_dir / dynamic_name
    
    return dynamic_dir


def main() -> int:
    args = parse_args()
    # Build dynamic log directory or use the provided one
    base_log_dir_arg = Path(args.log_dir)
    
    # Check if the user provided a custom log_dir or using the default
    # If using default pattern "outputs/heloco_run", build dynamic name
    if base_log_dir_arg.name == "heloco_run" and len(base_log_dir_arg.parts) <= 2:
        # Default pattern detected, build dynamic directory
        base_log_dir = build_dynamic_log_dir(base_log_dir_arg, args)
        print(f"== using dynamic log directory: {base_log_dir}")
    else:
        # Custom log directory provided, use as-is
        base_log_dir = REPO_ROOT / args.log_dir
    
    gpus = preflight(args)

    print(f"== config file: {args.config_file or '(none - built-in defaults + CLI flags)'}")
    print(f"== methods: {', '.join(args.methods)}")
    print(f"== recipe: --module {args.module} --config {args.config} "
          f"(assets {args.hf_assets}, dataset {args.dataset})")
    print(f"== topology: {args.islands} islands x {args.gpus_per_island} GPU(s) "
          f"on GPUs {gpus}; {args.steps} steps, sync every {args.sync_steps}, "
          f"seq_len {args.seq_len}, batch {args.batch}")
    print(f"== data distribution: {args.data_distribution}")
    if args.data_distribution in ("non_iid", "both") and args.languages:
        print(f"   languages (per island): {', '.join(args.languages)}")

    if args.dry_run:
        print_dry_run(args, gpus)
        return 0

    base_log_dir.mkdir(parents=True, exist_ok=True)

    # Handle "both" mode: run IID first, then non_IID
    data_distributions_to_run = []
    if args.data_distribution == "both":
        data_distributions_to_run = ["iid", "non_iid"]
    else:
        data_distributions_to_run = [args.data_distribution]
    
    methods_data: dict[str, list[list[dict]]] = {}
    
    for dist_mode in data_distributions_to_run:
        # Temporarily set data_distribution for this iteration
        original_dist = args.data_distribution
        args.data_distribution = dist_mode
        
        # Add mode suffix to method names when running "both" mode
        method_suffix = f"_{dist_mode}" if original_dist == "both" else ""
        
        print(f"\n{'='*70}")
        print(f"== DATA DISTRIBUTION MODE: {dist_mode.upper()}")
        print(f"{'='*70}")
        
        # Run each method sequentially
        for method in args.methods:
            method_key = f"{method}{method_suffix}"
            method_log_dir = base_log_dir / f"method-{method_key}"
            success = run_single_method(method, args, method_log_dir, gpus)
            if not success:
                print(f"== method {method} ({dist_mode} mode) failed; aborting", file=sys.stderr)
                return 1

            # Collect metrics for comparison plot
            try:
                import csv as csv_module
                import glob
                methods_data[method_key] = []
                for i in range(args.islands):
                    # Try multiple file name patterns to support different methods/modes
                    # Pattern 1: island-{i}_steps.csv (default)
                    # Pattern 2: island-{i}_steps_*.csv (for method-specific suffixes like mla)
                    metrics_dir = method_log_dir / "metrics"
                    csv_path = metrics_dir / f"island-{i}_steps.csv"
                    
                    if not csv_path.exists():
                        # Look for alternative file patterns (e.g., island-0_steps_noniid_llama3_15m.csv)
                        pattern = str(metrics_dir / f"island-{i}_steps*.csv")
                        matches = glob.glob(pattern)
                        if matches:
                            csv_path = Path(matches[0])  # Use first match
                    
                    if csv_path.exists():
                        rows = []
                        with open(csv_path) as f:
                            for row in csv_module.DictReader(f):
                                rows.append(row)
                        methods_data[method_key].append(rows)
                        print(f"   ✓ Collected metrics for {method_key} island-{i} from {csv_path.name}")
            except Exception as exc:
                print(f"== warning: failed to collect metrics for {method} ({dist_mode}): {exc}", file=sys.stderr)

    # Generate comparison plot and centralized evaluation log
    print()
    print("="*70)
    print("== Generating comparison plot and evaluation log")
    print("="*70)
    print()
    try:
        # Build comparison plot filename with mode and model suffixes
        # Mode suffix: _iid, _noniid, or _both
        mode_suffix = f"_{args.data_distribution}"
        model_suffix = f"_{args.config}" if args.config else ""
        plot_suffix = f"{mode_suffix}{model_suffix}"
        comparison_path = base_log_dir / f"comparison_loss{plot_suffix}.png"
        _plot_comparison(methods_data, comparison_path)
        print(f"== comparison plot saved to {comparison_path}")
        
        # Create centralized evaluation log (filtered by sync_steps)
        eval_log_path = base_log_dir / f"evaluation_summary{plot_suffix}.log"
        _create_evaluation_log(methods_data, eval_log_path, sync_steps=args.sync_steps)
        print(f"== evaluation log saved to {eval_log_path} (sync_steps={args.sync_steps})")
    except Exception as exc:
        print(f"== warning: comparison plot/log generation failed: {exc}", file=sys.stderr)

    print()
    print(f"== all methods completed. logs in {base_log_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

