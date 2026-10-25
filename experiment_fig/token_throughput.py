#!/usr/bin/env python3

# python token_throughput.py \
#   --on ../outputs/method_comparison/qwen600m_rl_true_20261023T023426-287c63 \
#   --off ../outputs/method_comparison/qwen600m_rl_false_20261023T020000-b79d82 \
#   --method both \
#   --limit 100 \
#   --output qwen600m_token_throughput

# python token_throughput.py \
#   --on ../outputs/method_comparison/llama15m_rl_true_20261023T013826-e28756 \
#   --off ../outputs/method_comparison/llama_15m_rl_false_20261023T014937-a67113 \
#   --method both \
#   --limit 100 \
#   --output llama15m_token_throughput

"""Plot reported token counts per outer synchronization, ON vs OFF.

Inputs: each run root contains <method>/syncs.csv, where 'tokens' is a
serialized list aligned with 'learner_ids'. The plot reproduces the reported
sync-token-volume preview; it is NOT a tokens/second throughput estimate.

Example, from panofabric-engine repository root:
  python experiment_fig/token_throughput.py \
    --on outputs/method_comparison/<rebase_ledger_on_run> \
    --off outputs/method_comparison/<rebase_ledger_off_run> \
    --method both --output experiment_fig/qwen600m
"""
import argparse
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Safe on headless HPC nodes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ("decoupled_heloco", "decoupled_diloco")
COLORS = {"ON": "#277DA1", "OFF": "#E07A5F"}


def parse_token_list(value, csv_path: Path, row_number: int) -> float:
    """Sum per-learner token counts reported on a synchronization row."""
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
        if not isinstance(parsed, (list, tuple)):
            raise ValueError("expected list of token counts")
        values = np.asarray(parsed, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("token counts must be finite and nonnegative")
        return float(values.sum())
    except (ValueError, SyntaxError, TypeError) as exc:
        raise ValueError(f"Invalid tokens at {csv_path}, CSV row {row_number}: {value!r}") from exc


def read_sync_tokens(run_root: Path, method: str, limit: int | None):
    csv_path = run_root / method / "syncs.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing synchronization records: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"sync_step", "tokens"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    df = df.sort_values("sync_step", kind="stable").reset_index(drop=True)
    if limit is not None:
        df = df.iloc[:limit].copy()
    if df.empty:
        raise ValueError(f"No synchronization records in {csv_path}")
    volume = np.array([
        parse_token_list(value, csv_path, row + 2)
        for row, value in enumerate(df["tokens"])
    ], dtype=float)
    # Use chronological 1-based sync index, consistent across run variants.
    sync_numbers = np.arange(1, len(df) + 1)
    return sync_numbers, volume


def plot_method(on_root, off_root, method, output_dir, limit, window, fmt, dpi):
    result = {
        "ON": read_sync_tokens(on_root, method, limit),
        "OFF": read_sync_tokens(off_root, method, limit),
    }
    method_name = "HeLoCo" if method == "decoupled_heloco" else "DiLoCo"
    fig, axes = plt.subplots(
        2, 1, figsize=(10.2, 7.3), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.1]},
    )
    ax_round, ax_total = axes
    for setting in ("OFF", "ON"):
        steps, volumes = result[setting]
        color = COLORS[setting]
        smoothed = pd.Series(volumes).rolling(window, center=True, min_periods=1).mean().to_numpy()
        ax_round.plot(steps, volumes / 1000, color=color, alpha=0.23, lw=1.1)
        ax_round.plot(
            steps, smoothed / 1000, color=color, lw=2.25,
            label=f"Rebase + ledger {setting} ({window}-sync average)",
        )
        cumulative = np.cumsum(volumes)
        ax_total.plot(
            steps, cumulative / 1e6, color=color, lw=2.5,
            label=f"{setting}: {cumulative[-1] / 1e6:.2f}M summed sync-token counts",
        )
        print(f"{method_name} {setting}: {len(steps)} syncs, "
              f"{cumulative[-1]:,.0f} summed sync-token counts")

    ax_round.set_ylabel("Summed reported tokens per sync (thousands)")
    ax_total.set_ylabel("Cumulative reported sync-token counts (millions)")
    ax_total.set_xlabel("Outer synchronization (chronological)")
    ax_round.set_title(f"{method_name}: per-sync reported token volume")
    for ax in axes:
        ax.grid(alpha=0.16)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Rebase + ledger ON vs OFF", fontsize=12, y=1.005)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{method}_roundwise_tokens.{fmt}"
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {target.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Compare sync-reported token volume (ON vs OFF)")
    parser.add_argument("--on", type=Path, required=True, help="Rebase+ledger ON experiment root")
    parser.add_argument("--off", type=Path, required=True, help="Rebase+ledger OFF experiment root")
    parser.add_argument("--method", choices=("both", *METHODS), default="both")
    parser.add_argument("--output", type=Path, default=Path("figures/tokens"))
    parser.add_argument("--limit", type=int, default=None, help="First N syncs, default: all")
    parser.add_argument("--window", type=int, default=7, help="Rolling-average sync window")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if args.window < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--window and --limit must be positive")
    methods = METHODS if args.method == "both" else (args.method,)
    for method in methods:
        plot_method(args.on, args.off, method, args.output, args.limit,
                    args.window, args.format, args.dpi)
    print("NOTE: The plotted token counts are summed from syncs.csv. If counters are cumulative, "
          "the cumulative line double-counts earlier tokens; it is not unique processed tokens ")

if __name__ == "__main__":
    main()
