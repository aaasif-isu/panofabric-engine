#!/usr/bin/env python3

# python experiment_fig/timeline.py \
#   --on outputs/method_comparison/llama15m_rl_true_20261023T013826-e28756 \
#   --off outputs/method_comparison/llama_15m_rl_false_20261023T014937-a67113 \
#   --method both \
#   --output experiment_fig/llama15m

# python timeline.py \
#   --on ../outputs/method_comparison/qwen600m_rl_true_20261023T023426-287c63 \
#   --off ../outputs/method_comparison/qwen600m_rl_false_20261023T020000-b79d82 \
#   --method both \
#   --output qwen600m

"""Plot learner contribution timelines and cumulative counts from comparison runs.

Use one comparison run directory for rebase+ledger ON and one for OFF.
Each run must contain <method>/syncs.csv (and preferably configs/<method>.yaml).
This plots accepted global contributions, NOT GPU compute intervals.
"""
import argparse
import ast
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_contributions(run_dir, method):
    path = run_dir / method / 'syncs.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing synchronization records: {path}')
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Empty synchronization file: {path}')
    ids = [int(r['sync_step']) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate sync_step in {path}')
    rows.sort(key=lambda r: int(r['sync_step']))
    sync_ids = [int(r['sync_step']) for r in rows]
    parsed = [ast.literal_eval(r['learner_ids']) for r in rows]
    if any(not isinstance(x, (list, tuple)) for x in parsed):
        raise ValueError('Expected learner_ids formatted as lists')
    return sync_ids, parsed


def labels_for(n, speeds):
    if not speeds:
        return [f'Learner {i}' for i in range(n)]
    if len(speeds) != n:
        raise ValueError(f'--speeds needs {n} comma-separated numbers, got {len(speeds)}')
    slowest = max(speeds)
    names = []
    for i, s in enumerate(speeds):
        label = 'fast' if s == 1 else ('slowest' if s == slowest and s > 1 else 'slow')
        names.append(f'Learner {i} ({label})')
    return names


def plot_one(run, method, enabled, out_dir, n_learners, speeds, limit, fmt):
    syncs, contributors = read_contributions(run, method)
    if limit is not None:
        syncs, contributors = syncs[:limit], contributors[:limit]
    if not syncs:
        raise ValueError('No syncs in selected range')
    if n_learners is None:
        n_learners = 1 + max((i for members in contributors for i in members), default=-1)
    if not n_learners:
        raise ValueError('No learners found')
    if any(i < 0 or i >= n_learners for members in contributors for i in members):
        raise ValueError('A contributor ID is outside the specified learner count')
    matrix = np.zeros((n_learners, len(syncs)), dtype=np.int8)
    for j, members in enumerate(contributors):
        for i in members:
            matrix[i, j] = 1
    counts = matrix.sum(axis=1)
    n = len(syncs)
    pos = np.arange(n)
    palette = plt.get_cmap('tab10')
    colors = [palette(i % 10) for i in range(n_learners)]
    y_labels = labels_for(n_learners, speeds)

    fig = plt.figure(figsize=(15, max(6.2, 1.0*n_learners + 2.5)))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1], hspace=0.11)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for i, color in enumerate(colors):
        xs = pos[matrix[i] == 1]
        ax.broken_barh([(float(j)+0.05, 0.90) for j in xs], (i - 0.36, 0.72), facecolors=color)
        cumulative = np.cumsum(matrix[i])
        ax2.step(np.arange(n+1), np.r_[0,cumulative], where='post',
                 linewidth=1.75, color=color, label=f'Learner {i}')
    ax.set_yticks(np.arange(n_learners),
                  [f'{y_labels[i]}  ·  {counts[i]}/{n}' for i in range(n_learners)])
    ax.set_ylim(n_learners-0.45, -0.55)
    ax.set_ylabel('Accepted contributor')
    ax.grid(axis='x', alpha=0.15)
    ax.set_xlim(0, n)
    ax.tick_params(axis='x', labelbottom=False)
    ax2.plot([0,n],[0,n], color='0.5', ls=':', linewidth=1.6, label='100% participation')
    ax2.set_xlim(0,n)
    ax2.set_ylim(0, n*1.04)
    ax2.set_ylabel('Cumulative contributions')
    ax2.set_xlabel('Synchronization number (chronological, 1 to N)')
    ax2.grid(alpha=0.17)
    ax2.legend(ncol=min(n_learners+1, 5), fontsize=9, loc='upper left')
    condition = 'Rebase + Ledger ON' if enabled else 'Rebase + Ledger OFF'
    name = method.replace('decoupled_', 'Decoupled ').replace('_', ' ').replace('heloco','HeLoCo').replace('diloco','DiLoCo')
    fig.suptitle(f'{name} — {condition} | {n} synchronization steps', fontsize=15)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f'{method}_rebase_ledger_{"on" if enabled else "off"}.{fmt}'
    fig.savefig(target, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'{target}: {dict(enumerate(map(int,counts)))}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--on', type=Path, required=True, help='Comparison run directory where rebase=true and ledger=true')
    ap.add_argument('--off', type=Path, required=True, help='Comparison run directory where rebase=false and ledger=false')
    ap.add_argument('--method', choices=['decoupled_heloco','decoupled_diloco','both'], default='decoupled_heloco')
    ap.add_argument('--output', type=Path, default=Path('figures/sync_participation'))
    ap.add_argument('--learners', type=int, default=4, help='Number of learners (default: 4)')
    ap.add_argument('--speeds', default='1,2,3,1', help='Slowness factors for row labels, or empty string to omit (default: 1,2,3,1)')
    ap.add_argument('--limit', type=int, default=None, help='Plot first N syncs (default: all)')
    ap.add_argument('--format', choices=['png','pdf','svg'], default='png')
    args = ap.parse_args()
    speeds = [float(x) for x in args.speeds.split(',') if x.strip()]
    methods = ['decoupled_heloco','decoupled_diloco'] if args.method == 'both' else [args.method]
    for method in methods:
        for path, enabled in [(args.on,True), (args.off,False)]:
            plot_one(path, method, enabled, args.output, args.learners, speeds, args.limit, args.format)

if __name__ == '__main__':
    main()
