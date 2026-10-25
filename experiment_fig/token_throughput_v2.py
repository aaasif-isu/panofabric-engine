
#!/usr/bin/env python3
"""
Paper-ready round-wise reported token-volume comparison.

Compares:
    Without Rebase + Ledger
    With Rebase + Ledger

Upper panel:
    Reported token volume per outer optimization step.

Lower panel:
    Cumulative sum of reported synchronization-token counts.

Input:
    <experiment_root>/<method>/syncs.csv

IMPORTANT:
    These are sync-reported token counts, NOT necessarily unique
    processed tokens and NOT tokens/second throughput.


python token_throughput_v2.py \
  --on ../outputs/method_comparison/llama15m_rl_true_20261023T013826-e28756 \
  --off ../outputs/method_comparison/llama_15m_rl_false_20261023T014937-a67113 \
  --method both \
  --limit 100 \
  --window 5 \
  --output llama15m_token_throughput \
  --format pdf \
  --font-base 16

python experiment_fig/token_throughput_v2.py \
  --on outputs/method_comparison/llama15m_rl_true_20261023T013826-e28756 \
  --off outputs/method_comparison/llama_15m_rl_false_20261023T014937-a67113 \
  --method both \
  --limit 100 \
  --window 5 \
  --output experiment_fig/llama15m_token_throughput \
  --format pdf \
  --font-base 16

"""

import argparse
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


METHODS = ("decoupled_heloco", "decoupled_diloco")

COLORS = {
    "OFF": "#D76A4E",
    "ON": "#2878A8",
}

LABELS = {
    "OFF": "Without Rebase + Ledger",
    "ON": "With Rebase + Ledger",
}


def configure_style(font_size):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
    })


def parse_token_list(value, csv_path, row_number):
    """Sum token counts reported for learners at one sync."""

    try:
        parsed = (
            ast.literal_eval(value)
            if isinstance(value, str)
            else value
        )

        if not isinstance(parsed, (list, tuple)):
            raise ValueError("Expected a list of token counts")

        values = np.asarray(parsed, dtype=float)

        if not np.all(np.isfinite(values)):
            raise ValueError("Token counts must be finite")

        if np.any(values < 0):
            raise ValueError("Token counts must be nonnegative")

        return float(values.sum())

    except (ValueError, SyntaxError, TypeError) as exc:
        raise ValueError(
            f"Invalid tokens in {csv_path}, row {row_number}: "
            f"{value!r}"
        ) from exc


def read_sync_tokens(run_root, method, limit):
    """Load chronological synchronization-token counts."""

    csv_path = Path(run_root) / method / "syncs.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing synchronization records: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required = {"sync_step", "tokens"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{csv_path} is missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(f"Empty synchronization file: {csv_path}")

    if df["sync_step"].duplicated().any():
        raise ValueError(
            f"Duplicate sync_step values in {csv_path}"
        )

    df = df.sort_values(
        "sync_step",
        kind="stable"
    ).reset_index(drop=True)

    if limit is not None:
        df = df.iloc[:limit].copy()

    if df.empty:
        raise ValueError(
            f"No synchronization records selected: {csv_path}"
        )

    volumes = np.array([
        parse_token_list(value, csv_path, index + 2)
        for index, value in enumerate(df["tokens"])
    ], dtype=float)

    steps = np.arange(
        1,
        len(volumes) + 1
    )

    return steps, volumes


def plot_method(
    on_root,
    off_root,
    method,
    output_dir,
    limit,
    window,
    fmt,
    dpi,
    font_size,
):
    result = {
        "OFF": read_sync_tokens(off_root, method, limit),
        "ON": read_sync_tokens(on_root, method, limit),
    }

    configure_style(font_size)

    # --------------------------------------------------
    # Figure layout
    # --------------------------------------------------

    fig, (ax_round, ax_total) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.05, 1.0],
            "hspace": 0.16,
        },
    )

    # --------------------------------------------------
    # Plot both configurations
    # --------------------------------------------------

    endpoints = {}

    for setting in ("OFF", "ON"):

        steps, volumes = result[setting]
        color = COLORS[setting]

        smoothed = (
            pd.Series(volumes)
            .rolling(
                window=window,
                center=True,
                min_periods=1
            )
            .mean()
            .to_numpy()
        )

        # Upper: raw observations (faint)
        ax_round.plot(
            steps,
            volumes / 1_000,
            color=color,
            alpha=0.20,
            linewidth=1.1,
            zorder=1,
        )

        # Upper: rolling average (prominent)
        ax_round.plot(
            steps,
            smoothed / 1_000,
            color=color,
            linewidth=2.8,
            solid_capstyle="round",
            zorder=3,
        )

        # Lower: cumulative sum of sync-reported counts
        cumulative = np.cumsum(volumes)

        ax_total.plot(
            steps,
            cumulative / 1_000_000,
            color=color,
            linewidth=3.0,
            solid_capstyle="round",
            zorder=3,
        )

        endpoints[setting] = (
            steps[-1],
            cumulative[-1] / 1_000_000
        )

        print(
            f"{method} {LABELS[setting]}: "
            f"{len(steps)} synchronizations, "
            f"{cumulative[-1]:,.0f} summed sync-token counts"
        )

    # --------------------------------------------------
    # Axis labels
    # --------------------------------------------------

    ax_round.set_ylabel(
        "Tokens per Outer Step (K)",
        fontsize=font_size,
        labelpad=10,
    )

    ax_total.set_ylabel(
        "Cumulative Token Volume (M)",
        fontsize=font_size,
        labelpad=10,
    )

    ax_total.set_xlabel(
        "Outer Optimization Step",
        fontsize=font_size,
        labelpad=9,
    )

    # --------------------------------------------------
    # Clean axis style
    # --------------------------------------------------

    max_steps = max(
        len(result["OFF"][0]),
        len(result["ON"][0])
    )

    ax_total.set_xlim(
        0,
        max_steps * 1.09
    )

    ax_round.margins(y=0.10)

    ax_total.set_ylim(
        bottom=0
    )

    ticks = np.unique(
        np.round(
            np.linspace(0, max_steps, 6)
        ).astype(int)
    )

    ax_total.set_xticks(ticks)

    for ax in (ax_round, ax_total):

        ax.grid(
            axis="both",
            color="#D9D9D9",
            linewidth=0.75,
            alpha=0.55,
        )

        ax.set_axisbelow(True)

        ax.tick_params(
            axis="both",
            labelsize=font_size - 2,
            length=4,
            width=0.8,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # # --------------------------------------------------
    # # Endpoint labels for cumulative curves
    # # --------------------------------------------------

    # for setting in ("OFF", "ON"):

    #     end_x, end_y = endpoints[setting]

    #     ax_total.annotate(
    #         f"{end_y:.2f}M",
    #         xy=(end_x, end_y),
    #         xytext=(8, 0),
    #         textcoords="offset points",
    #         color=COLORS[setting],
    #         fontsize=font_size - 1,
    #         fontweight="bold",
    #         va="center",
    #         ha="left",
    #     )

    # --------------------------------------------------
    # One shared legend, above the panels
    # --------------------------------------------------

    legend_handles = [
        Line2D(
            [0], [0],
            color=COLORS["OFF"],
            linewidth=3,
            label=LABELS["OFF"],
        ),
        Line2D(
            [0], [0],
            color=COLORS["ON"],
            linewidth=3,
            label=LABELS["ON"],
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.99),
        ncol=2,
        fontsize=font_size - 2,
        frameon=False,
        columnspacing=2.5,
        handlelength=2.7,
    )

    # --------------------------------------------------
    # Spacing
    # --------------------------------------------------

    fig.subplots_adjust(
        left=0.15,
        right=0.91,
        top=0.89,
        bottom=0.11,
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    target = (
        output_dir /
        f"{method}_roundwise_tokens.{fmt}"
    )

    fig.savefig(
        target,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.14,
    )

    plt.close(fig)

    print(f"Saved: {target.resolve()}")


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--on",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--off",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--method",
        choices=("both", *METHODS),
        default="both",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/tokens"),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--window",
        type=int,
        default=7,
        help="Moving-average window (default: 7)",
    )

    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="pdf",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--font-base",
        type=int,
        default=16,
        help="Base font size (default: 16)",
    )

    args = parser.parse_args()

    if args.window < 1:
        parser.error("--window must be positive")

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    if args.font_base < 8:
        parser.error("--font-base must be at least 8")

    methods = (
        METHODS
        if args.method == "both"
        else (args.method,)
    )

    for method in methods:

        plot_method(
            on_root=args.on,
            off_root=args.off,
            method=method,
            output_dir=args.output,
            limit=args.limit,
            window=args.window,
            fmt=args.format,
            dpi=args.dpi,
            font_size=args.font_base,
        )

    print(
        "\nNOTE: These curves use summed token counts from "
        "syncs.csv, not verified unique processed tokens. "
        "If the counters overlap across synchronizations, "
        "the cumulative values can double-count training work."
    )


if __name__ == "__main__":
    main()
