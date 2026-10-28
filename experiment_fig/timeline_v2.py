#!/usr/bin/env python3
"""
Paper-ready side-by-side learner participation timeline.

Left:  Rebase + Ledger OFF
Right: Rebase + Ledger ON

Upper panels:
    Accepted learner participation at each outer optimization step.

Lower panels:
    Cumulative number of synchronizations in which each learner
    contributed an accepted update.

Input:
    <experiment_directory>/<method>/syncs.csv

python experiment_fig/timeline_v2.py \
  --on outputs/method_comparison/llama_15m_rl_true_adap_true_20261027T165227-7647ae \
  --off outputs/method_comparison/llama_15m_rl_false_adap_true_20261027T162146-87d608 \
  --method both \
  --output experiment_fig/llama15m_timeline
"""

import argparse
import ast
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def configure_style(font_size: int):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
    })


def read_contributions(run_dir: Path, method: str):
    path = run_dir / method / "syncs.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing synchronization records: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Empty synchronization file: {path}")

    rows.sort(key=lambda row: int(row["sync_step"]))

    sync_ids = [int(row["sync_step"]) for row in rows]
    if len(sync_ids) != len(set(sync_ids)):
        raise ValueError(f"Duplicate sync_step values found in {path}")

    contributors = []
    for row in rows:
        ids = ast.literal_eval(row["learner_ids"])
        if not isinstance(ids, (list, tuple)):
            raise ValueError(f"Invalid learner_ids format in {path}")
        contributors.append([int(i) for i in ids])

    return sync_ids, contributors


def learner_labels(n_learners: int, speeds):
    if not speeds:
        return [f"L{i}" for i in range(n_learners)]

    if len(speeds) != n_learners:
        raise ValueError(
            f"Expected {n_learners} speed factors, got {len(speeds)}"
        )

    fastest = min(speeds)
    slowest = max(speeds)

    labels = []
    for i, s in enumerate(speeds):
        if s == fastest:
            kind = "Fast"
        elif s == slowest:
            kind = "Slowest"
        else:
            kind = "Slow"
        labels.append(f"L{i} ({kind})")
    return labels


def build_matrix(contributors, n_learners: int):
    n_steps = len(contributors)
    matrix = np.zeros((n_learners, n_steps), dtype=np.int8)

    for step, members in enumerate(contributors):
        if len(members) != len(set(members)):
            raise ValueError(f"Duplicate learner ID at sync step {step}")

        for learner_id in members:
            if learner_id < 0 or learner_id >= n_learners:
                raise ValueError(f"Invalid learner ID {learner_id}")
            matrix[learner_id, step] = 1

    return matrix


def draw_top_panel(ax, matrix, labels, colors, title, max_steps, font_size, side="left"):
    n_learners, n_steps = matrix.shape
    counts = matrix.sum(axis=1)
    percentages = counts / n_steps * 100.0
    steps = np.arange(1, n_steps + 1)

    for i in range(n_learners):
        active_steps = steps[matrix[i] == 1]
        bars = [(float(step) - 0.45, 0.90) for step in active_steps]
        ax.broken_barh(
            bars,
            (i - 0.33, 0.66),
            facecolors=colors[i],
            edgecolors="none",
            zorder=3
        )

    ax.set_xlim(0.5, max_steps + 0.5)
    ax.set_ylim(n_learners - 0.5, -0.5)
    ax.set_xticks([])
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=font_size + 2, fontweight="bold", pad=10)

    tick_text = [f"{labels[i]}  ·  {percentages[i]:.0f}%" for i in range(n_learners)]
    ax.set_yticks(np.arange(n_learners))
    ax.set_yticklabels(tick_text, fontsize=font_size - 1)

    if side == "left":
        ax.tick_params(axis="y", left=False, right=False, labelleft=True, labelright=False, pad=4, length=0)
    elif side == "right":
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=True, pad=4, length=0)
    else:
        raise ValueError("side must be either 'left' or 'right'")

    return counts, percentages


def draw_bottom_panel(ax, matrix, colors, max_steps, font_size, show_ylabel=False):
    n_learners, n_steps = matrix.shape
    x = np.arange(n_steps + 1)

    for i in range(n_learners):
        cumulative = np.r_[0, np.cumsum(matrix[i])]
        ax.step(
            x,
            cumulative,
            where="post",
            linewidth=2.1,
            color=colors[i],
            label=f"L{i}",
            zorder=3
        )

    ax.set_xlim(0, max_steps)
    ax.set_ylim(0, max_steps * 1.03)
    ax.grid(color="#D9D9D9", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Outer Optimization Step", fontsize=font_size, labelpad=6)

    if show_ylabel:
        ax.set_ylabel("Cumulative Participations", fontsize=font_size, labelpad=8)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)

    step_ticks = np.unique(np.round(np.linspace(0, max_steps, 6)).astype(int))
    ax.set_xticks(step_ticks)


def plot_comparison(
    run_off: Path,
    run_on: Path,
    method: str,
    output_dir: Path,
    n_learners: int,
    speeds,
    limit,
    fmt: str,
    font_size: int,
):
    _, off_contributors = read_contributions(run_off, method)
    _, on_contributors = read_contributions(run_on, method)

    if limit is not None:
        off_contributors = off_contributors[:limit]
        on_contributors = on_contributors[:limit]

    if not off_contributors or not on_contributors:
        raise ValueError("Both runs must contain synchronization records.")

    if n_learners is None:
        all_ids = [
            i for run in (off_contributors, on_contributors)
            for members in run for i in members
        ]
        if not all_ids:
            raise ValueError("No learner IDs found.")
        n_learners = max(all_ids) + 1

    matrix_off = build_matrix(off_contributors, n_learners)
    matrix_on = build_matrix(on_contributors, n_learners)

    max_steps = max(matrix_off.shape[1], matrix_on.shape[1])

    labels = learner_labels(n_learners, speeds)
    palette = plt.get_cmap("tab10")
    colors = [palette(i % 10) for i in range(n_learners)]

    configure_style(font_size)

    fig = plt.figure(figsize=(19, max(7.0, n_learners * 0.9 + 3.6)))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[2.0, 1.0],
        width_ratios=[1, 1],
        hspace=0.14,
        wspace=0.08,
    )

    ax_top_off = fig.add_subplot(gs[0, 0])
    ax_bottom_off = fig.add_subplot(gs[1, 0])

    ax_top_on = fig.add_subplot(gs[0, 1])
    ax_bottom_on = fig.add_subplot(gs[1, 1])

    counts_off, pct_off = draw_top_panel(
        ax_top_off,
        matrix_off,
        labels,
        colors,
        "(a) Without Rebase + Ledger",
        max_steps,
        font_size,
        side="left",
    )

    counts_on, pct_on = draw_top_panel(
        ax_top_on,
        matrix_on,
        labels,
        colors,
        "(b) With Rebase + Ledger",
        max_steps,
        font_size,
        side="right",
    )

    draw_bottom_panel(
        ax_bottom_off,
        matrix_off,
        colors,
        max_steps,
        font_size,
        show_ylabel=True,
    )

    draw_bottom_panel(
        ax_bottom_on,
        matrix_on,
        colors,
        max_steps,
        font_size,
        show_ylabel=False,
    )

    # Shared legend only for learners (no full-participation line)
    handles = [
        Line2D([0], [0], color=colors[i], linewidth=2.5, label=f"L{i}")
        for i in range(n_learners)
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=min(n_learners, 6),
        frameon=False,
        fontsize=font_size - 1,
        handlelength=2.6,
        columnspacing=2.0,
    )

    fig.subplots_adjust(
        left=0.11,
        right=0.95,
        top=0.92,
        bottom=0.17,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{method}_rebase_ledger_comparison.{fmt}"

    fig.savefig(target, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    print(f"\nSaved: {target}")
    print("OFF:", {i: f"{int(counts_off[i])} ({pct_off[i]:.1f}%)" for i in range(n_learners)})
    print("ON :", {i: f"{int(counts_on[i])} ({pct_on[i]:.1f}%)" for i in range(n_learners)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--on", type=Path, required=True,
                        help="Run directory with rebase=true and ledger=true")
    parser.add_argument("--off", type=Path, required=True,
                        help="Run directory with rebase=false and ledger=false")
    parser.add_argument("--method", choices=["decoupled_heloco", "decoupled_diloco", "both"],
                        default="both")
    parser.add_argument("--output", type=Path, default=Path("timeline_figures"))
    parser.add_argument("--learners", type=int, default=4,
                        help="Number of learners (default: 4)")
    parser.add_argument("--speeds", default="1,2,3,1",
                        help="Slowness factors, e.g. 1,2,3,1")
    parser.add_argument("--limit", type=int, default=None,
                        help="Plot only the first N synchronization steps")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="pdf")
    parser.add_argument("--font-base", type=int, default=16,
                        help="Base font size")

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    speeds = [float(x) for x in args.speeds.split(",") if x.strip()]
    methods = ["decoupled_heloco", "decoupled_diloco"] if args.method == "both" else [args.method]

    for method in methods:
        plot_comparison(
            run_off=args.off,
            run_on=args.on,
            method=method,
            output_dir=args.output,
            n_learners=args.learners,
            speeds=speeds,
            limit=args.limit,
            fmt=args.format,
            font_size=args.font_base,
        )


if __name__ == "__main__":
    main()