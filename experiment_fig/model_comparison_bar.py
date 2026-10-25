
#!/usr/bin/env python3
"""
Grouped bar chart comparing three methods across model sizes.

Metric: Validation Loss (lower is better).

Manually fill in the values in RESULTS below.

Output:
    PNG, PDF, or SVG for publication.

python model_comparison_bar.py \
  --output model_comparison \
  --format pdf \
  --font-base 16 \
  --show-values
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. ENTER YOUR EXPERIMENT RESULTS HERE
# ============================================================

MODELS = [
    "Llama 15M",
    "Gemma 270M",
    "Qwen 0.6B",
]

RESULTS = {
    "Decoupled DiLoCo": [
        3.40,  # Llama 15M
        3.65,  # Gemma 270M
        3.20,  # Qwen 0.6B
    ],

    "Decoupled HeLoCo": [
        3.29,
        2.88,
        3.10,
    ],

    "Decoupled HeLoCo + Rebase + Ledger": [
        3.21,
        2.83,
        2.95,
    ],
}


# ============================================================
# 2. FIGURE SETTINGS
# ============================================================

COLORS = {
    "Decoupled DiLoCo": "#D28A65",
    "Decoupled HeLoCo": "#6497BE",
    "Decoupled HeLoCo + Rebase + Ledger": "#27638A",
}


def configure_style(font_size):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
    })


# ============================================================
# 3. VALIDATE DATA
# ============================================================

def validate_results():

    for method, values in RESULTS.items():

        if len(values) != len(MODELS):
            raise ValueError(
                f"{method}: expected {len(MODELS)} values."
            )

        missing = [
            MODELS[i]
            for i, value in enumerate(values)
            if value is None
        ]

        if missing:
            raise ValueError(
                f"Missing validation loss for {method}: "
                f"{', '.join(missing)}"
            )

        for value in values:
            if not np.isfinite(value):
                raise ValueError(
                    f"{method}: values must be finite numbers."
                )


# ============================================================
# 4. GENERATE GROUPED BAR CHART
# ============================================================

def plot_comparison(output, fmt, font_size, dpi, show_values):

    validate_results()
    configure_style(font_size)

    fig, ax = plt.subplots(figsize=(11.5, 5.8))

    x = np.arange(len(MODELS))

    methods = list(RESULTS.keys())

    n_methods = len(methods)

    bar_width = 0.23

    offsets = (
        np.arange(n_methods) - (n_methods - 1) / 2
    ) * bar_width

    all_values = []

    for i, method in enumerate(methods):

        values = np.asarray(RESULTS[method], dtype=float)

        all_values.extend(values)

        bars = ax.bar(
            x + offsets[i],
            values,
            width=bar_width,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.7,
            label=method,
            zorder=3,
        )

        if show_values:
            ax.bar_label(
                bars,
                labels=[f"{v:.2f}" for v in values],
                padding=4,
                fontsize=font_size - 3,
            )

    # --------------------------------------------------------
    # Axis settings
    # --------------------------------------------------------

    ax.set_xticks(x)

    ax.set_xticklabels(
        MODELS,
        fontsize=font_size,
    )

    ax.set_ylabel(
        "Validation Loss",
        fontsize=font_size,
        labelpad=12,
    )

    # Zero baseline for an honest bar-chart comparison.
    ax.set_ylim(
        0,
        max(all_values) * (1.20 if show_values else 1.12),
    )

    ax.set_xlim(
        -0.6,
        len(MODELS) - 0.4,
    )

    # --------------------------------------------------------
    # Professional formatting
    # --------------------------------------------------------

    ax.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.8,
        alpha=0.6,
        zorder=0,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        labelsize=font_size - 1,
        length=4,
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.26),
        ncol=3,
        frameon=False,
        fontsize=font_size - 3,
        handlelength=1.5,
        columnspacing=1.3,
    )

    # No figure title: use the LaTeX caption instead.

    fig.subplots_adjust(
        left=0.11,
        right=0.98,
        top=0.79,
        bottom=0.16,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    target = output / f"model_validation_loss.{fmt}"

    fig.savefig(
        target,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.15,
    )

    plt.close(fig)

    print(f"Saved: {target.resolve()}")


# ============================================================
# 5. COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Plot validation loss across model families."
    )

    parser.add_argument(
        "--output",
        default="model_comparison",
    )

    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="pdf",
    )

    parser.add_argument(
        "--font-base",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Show numeric loss above each bar.",
    )

    args = parser.parse_args()

    plot_comparison(
        output=args.output,
        fmt=args.format,
        font_size=args.font_base,
        dpi=args.dpi,
        show_values=args.show_values,
    )


if __name__ == "__main__":
    main()
