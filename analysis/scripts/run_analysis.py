import os
import sys
import json
import ast
import yaml
import argparse
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Fail-Fast Validation and Data Loading
# -----------------------------------------------------------------------------
def hash_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def verify_experiments(orig_dir, rebase_dir):
    print("Verifying data provenance and experimental configurations...")
    
    # 1. Verify Initial Hashes
    comp_orig = safe_load_json(os.path.join(orig_dir, "comparison.json"))
    comp_rebase = safe_load_json(os.path.join(rebase_dir, "comparison.json"))
    if comp_orig.get("initial_parameters_sha256") != comp_rebase.get("initial_parameters_sha256"):
        raise ValueError("Initial parameters mismatch. Experiments did not start from the same seed/model.")

    for method in ["decoupled_heloco", "decoupled_diloco"]:
        # 2. Verify run-options.json
        opt_orig = safe_load_json(os.path.join(orig_dir, method, "run-options.json"))
        opt_rebase = safe_load_json(os.path.join(rebase_dir, method, "run-options.json"))
        
        keys_to_check = ["seed", "dataset", "module", "config", "island_slowness_factors", "coordination_method"]
        for k in keys_to_check:
            if opt_orig.get(k) != opt_rebase.get(k):
                raise ValueError(f"Mismatch in {method} run option {k}: {opt_orig.get(k)} vs {opt_rebase.get(k)}")
                
        # 3. Verify experiment.yaml control variables and flags
        with open(os.path.join(orig_dir, method, "experiment.yaml")) as f:
            cfg_orig = yaml.safe_load(f)
        with open(os.path.join(rebase_dir, method, "experiment.yaml")) as f:
            cfg_rebase = yaml.safe_load(f)
            
        dec_orig = cfg_orig.get("decoupled", {})
        dec_rebase = cfg_rebase.get("decoupled", {})
        
        if dec_orig.get("rebase", False) is not False or dec_orig.get("ledger", False) is not False:
            raise ValueError(f"Original {method} config must have rebase=False, ledger=False")
        if dec_rebase.get("rebase", False) is not True or dec_rebase.get("ledger", False) is not True:
            raise ValueError(f"Rebase+Ledger {method} config must have rebase=True, ledger=True")
            
        control_keys = ["num_fragments", "min_quorum", "min_local_steps", "scheduler", "stopping"]
        for k in control_keys:
            if dec_orig.get(k) != dec_rebase.get(k):
                raise ValueError(f"Mismatch in {method} decoupled config {k}: {dec_orig.get(k)} vs {dec_rebase.get(k)}")

    # 4. Verify Validation Caches
    for lang in ["en", "fr", "de", "es"]:
        f1 = os.path.join(orig_dir, f"validation_cache_{lang}.pt")
        f2 = os.path.join(rebase_dir, f"validation_cache_{lang}.pt")
        if os.path.exists(f1) and os.path.exists(f2):
            if hash_file(f1) != hash_file(f2):
                raise ValueError(f"Validation cache for {lang} does not match between runs!")

    print("Verification passed.")
    return opt_orig.get("island_slowness_factors", [1, 1, 1, 1])

def safe_load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required CSV data: {path}")
    return pd.read_csv(path)

def safe_load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required JSON data: {path}")
    with open(path) as f:
        return json.load(f)

# -----------------------------------------------------------------------------
# Data Processing and Calculation
# -----------------------------------------------------------------------------
def load_syncs(run_dir, method):
    path = os.path.join(run_dir, method, "syncs.csv")
    df = safe_load_csv(path)
    df["learner_ids_list"] = df["learner_ids"].apply(ast.literal_eval)
    df["weights_list"] = df["weights"].apply(ast.literal_eval)
    df["quorum_size"] = df["learner_ids_list"].apply(len)
    return df

def compute_cumulative_stats(df, num_learners=4):
    cum_counts = {i: [0] for i in range(num_learners)}
    cum_weights = {i: [0.0] for i in range(num_learners)}
    elapsed = [0.0]
    steps = [0]
    
    for i, row in df.iterrows():
        elapsed.append(row["elapsed_s"])
        steps.append(row["sync_step"])
        for l_id in range(num_learners):
            added_count = 1 if l_id in row["learner_ids_list"] else 0
            added_weight = 0.0
            if added_count:
                idx = row["learner_ids_list"].index(l_id)
                added_weight = row["weights_list"][idx]
            
            cum_counts[l_id].append(cum_counts[l_id][-1] + added_count)
            cum_weights[l_id].append(cum_weights[l_id][-1] + added_weight)
            
    return elapsed, steps, cum_counts, cum_weights

def compute_starvation(df, num_learners=4):
    stats = []
    for l_id in range(num_learners):
        participated_steps = df[df["learner_ids_list"].apply(lambda x: l_id in x)]["sync_step"].tolist()
        if not participated_steps:
            stats.append({"learner_id": l_id, "first_sync": -1, "max_gap": -1, "total_syncs": 0})
            continue
        first_sync = participated_steps[0]
        gaps = np.diff([0] + participated_steps) 
        max_gap = max(gaps) if len(gaps) > 0 else 0
        stats.append({"learner_id": l_id, "first_sync": first_sync, "max_gap": max_gap, "total_syncs": len(participated_steps)})
    return pd.DataFrame(stats)

def jains_index(values):
    values = np.array(values)
    if sum(values) == 0: return 0.0
    return float((np.sum(values)**2) / (len(values) * np.sum(values**2)))

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def save_fig(fig, output_dir, name):
    fig.savefig(os.path.join(output_dir, "figures", f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "figures", f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_participation_heatmap(df_orig, df_rebase, method, output_dir):
    """Scientific Question: When and how frequently did each worker participate in synchronization?"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for ax, df, title in zip(axes, [df_orig, df_rebase], ["Original (rebase=False, ledger=False)", "Rebase + Ledger"]):
        matrix = np.zeros((4, len(df)))
        for i, row in df.iterrows():
            for l_id in row["learner_ids_list"]:
                matrix[l_id, i] = 1
        ax.imshow(matrix, aspect='auto', cmap='Blues', interpolation='none')
        ax.set_yticks([0, 1, 2, 3])
        ax.set_ylabel("Learner ID")
        ax.set_title(f"{method}: {title}")
    axes[1].set_xlabel("Synchronization Step")
    plt.tight_layout()
    save_fig(fig, output_dir, f"{method}_participation_heatmap")

def plot_cumulative_contributions(df_orig, df_rebase, method, output_dir):
    """Scientific Question: How does the influence of each worker accumulate mathematically over synchronization steps?"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    elapsed_orig, steps_orig, counts_orig, weights_orig = compute_cumulative_stats(df_orig)
    elapsed_rebase, steps_rebase, counts_rebase, weights_rebase = compute_cumulative_stats(df_rebase)
    
    for l_id in range(4):
        axes[0,0].plot(steps_orig, counts_orig[l_id], label=f"Learner {l_id}")
        axes[0,1].plot(steps_rebase, counts_rebase[l_id], label=f"Learner {l_id}")
        axes[1,0].plot(steps_orig, weights_orig[l_id], label=f"Learner {l_id}")
        axes[1,1].plot(steps_rebase, weights_rebase[l_id], label=f"Learner {l_id}")
        
    axes[0,0].set_title("Original: Cum. Contributions")
    axes[0,1].set_title("Rebase+Ledger: Cum. Contributions")
    axes[1,0].set_title("Original: Cum. Weights")
    axes[1,1].set_title("Rebase+Ledger: Cum. Weights")
    
    for ax in axes.flat:
        ax.set_xlabel("Sync Step")
        ax.legend()
        
    axes[0,0].set_ylabel("Accepted Count")
    axes[1,0].set_ylabel("Cumulative Weight")
    
    plt.tight_layout()
    save_fig(fig, output_dir, f"{method}_cumulative_contributions")

def plot_time_behavior(df_orig, df_rebase, method, output_dir, slowness_factors):
    """Scientific Question: Does Rebase+Ledger delay global sync pacing, and when do slow workers contribute?"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Panel 1: Sync Progress vs Elapsed Time
    axes[0].plot(df_orig["elapsed_s"], df_orig["sync_step"], label="Original", color="blue")
    axes[0].plot(df_rebase["elapsed_s"], df_rebase["sync_step"], label="Rebase+Ledger", color="orange")
    axes[0].set_xlabel("Elapsed Time (s)")
    axes[0].set_ylabel("Global Sync Step")
    axes[0].set_title("Global Synchronization Pacing")
    axes[0].legend()
    
    # Panel 2: Slow Worker Contributions vs Elapsed Time
    slow_learners = [i for i, f in enumerate(slowness_factors) if f > 1]
    
    if slow_learners:
        el_orig, st_orig, c_orig, _ = compute_cumulative_stats(df_orig)
        el_rebase, st_rebase, c_rebase, _ = compute_cumulative_stats(df_rebase)
        
        slow_cum_orig = np.sum([c_orig[i] for i in slow_learners], axis=0)
        slow_cum_rebase = np.sum([c_rebase[i] for i in slow_learners], axis=0)
        
        axes[1].plot(el_orig, slow_cum_orig, label="Original", color="blue")
        axes[1].plot(el_rebase, slow_cum_rebase, label="Rebase+Ledger", color="orange")
        axes[1].set_xlabel("Elapsed Time (s)")
        axes[1].set_ylabel("Cumulative Accepted Contributions")
        axes[1].set_title("Slow Workers' Participation Rate")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No slow workers configured", ha='center', va='center')
        
    plt.tight_layout()
    save_fig(fig, output_dir, f"{method}_time_behavior")

def plot_efficiency_tradeoff(eff_df, method, output_dir):
    """Scientific Question: What is the system cost of Rebase+Ledger in time, throughput, and communication?"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    subset = eff_df[eff_df["method"] == method]
    x = np.arange(2)
    labels = ["Original", "Rebase+Ledger"]
    
    vals_time = subset["elapsed_s"].values
    vals_tps = subset["tokens_per_sec"].values
    vals_tx = subset["tx_bytes"].values / (1024**3) # GB
    vals_rx = subset["rx_bytes"].values / (1024**3) # GB
    vals_cps = subset["accepted_per_sec"].values
    
    axes[0].bar(x, vals_time, color=["blue", "orange"], alpha=0.7)
    axes[0].set_title("Wall-Clock Time")
    axes[0].set_ylabel("Seconds")
    
    axes[1].bar(x, vals_tps, color=["blue", "orange"], alpha=0.7)
    axes[1].set_title("Token Throughput")
    axes[1].set_ylabel("Tokens / Sec")
    
    width = 0.35
    axes[2].bar(x - width/2, vals_tx, width, label='TX (Server->Worker)', color="lightblue")
    axes[2].bar(x + width/2, vals_rx, width, label='RX (Worker->Server)', color="darkblue")
    axes[2].set_title("Communication Traffic")
    axes[2].set_ylabel("Gigabytes (GB)")
    axes[2].legend()
    
    axes[3].bar(x, vals_cps, color=["blue", "orange"], alpha=0.7)
    axes[3].set_title("Aggregations / Sec")
    axes[3].set_ylabel("Contributions / Sec")
    
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        
    plt.tight_layout()
    save_fig(fig, output_dir, f"{method}_efficiency_tradeoff")

def analyze_evaluations(orig_dir, rebase_dir, output_dir):
    print("Analyzing Evaluations...")
    data = []
    langs = ["evaluation", "evaluation_de", "evaluation_en", "evaluation_es", "evaluation_fr"]
    
    for run_dir, exp_name in [(orig_dir, "Original"), (rebase_dir, "Rebase+Ledger")]:
        for lang in langs:
            path = os.path.join(run_dir, lang, "evaluation.json")
            if not os.path.exists(path): continue
            eval_data = safe_load_json(path)
            lang_name = "overall" if lang == "evaluation" else lang.split("_")[1]
            
            for row in eval_data.get("models", []):
                if row.get("method") in ["decoupled_heloco", "decoupled_diloco"]:
                    data.append({
                        "method": row["method"],
                        "language": lang_name,
                        "experiment": exp_name,
                        "loss": row.get("loss"),
                        "accuracy": row.get("next_token_accuracy"),
                        "perplexity": row.get("perplexity")
                    })
                    
    df_eval = pd.DataFrame(data)
    df_eval.to_csv(os.path.join(output_dir, "data", "evaluation_summary.csv"), index=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, method in zip(axes, ["decoupled_heloco", "decoupled_diloco"]):
        subset = df_eval[df_eval["method"] == method]
        if subset.empty: continue
        
        orig_loss = subset[subset["experiment"] == "Original"].set_index("language")["loss"]
        rebase_loss = subset[subset["experiment"] == "Rebase+Ledger"].set_index("language")["loss"]
        
        df_plot = pd.DataFrame({"Original": orig_loss, "Rebase+Ledger": rebase_loss}).dropna()
        df_plot.plot(kind="bar", ax=ax, alpha=0.8, color=["blue", "orange"])
        ax.set_title(f"{method}: Validation Loss")
        ax.set_ylabel("Cross Entropy Loss")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        # add exact values
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}", (p.get_x() * 1.005, p.get_height() * 1.005), fontsize=8)
            
    plt.tight_layout()
    save_fig(fig, output_dir, "evaluation_loss_comparison")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze Rebase+Ledger Ablation")
    parser.add_argument("--original", required=True, help="Path to original run directory")
    parser.add_argument("--rebase-ledger", required=True, help="Path to Rebase+Ledger run directory")
    parser.add_argument("--output", required=True, help="Path to output analysis directory")
    args = parser.parse_args()
    
    orig_dir = os.path.abspath(args.original)
    rebase_dir = os.path.abspath(args.rebase_ledger)
    out_dir = os.path.abspath(args.output)
    
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    
    slowness_factors = verify_experiments(orig_dir, rebase_dir)
    
    eff_records = []
    
    for method in ["decoupled_heloco", "decoupled_diloco"]:
        print(f"Analyzing {method}...")
        df_orig = load_syncs(orig_dir, method)
        df_rebase = load_syncs(rebase_dir, method)
        
        plot_participation_heatmap(df_orig, df_rebase, method, out_dir)
        plot_cumulative_contributions(df_orig, df_rebase, method, out_dir)
        plot_time_behavior(df_orig, df_rebase, method, out_dir, slowness_factors)
        
        # Quorum dist
        q_orig = df_orig["quorum_size"].value_counts().to_dict()
        q_rebase = df_rebase["quorum_size"].value_counts().to_dict()
        pd.DataFrame({"Original": q_orig, "Rebase+Ledger": q_rebase}).fillna(0).to_csv(os.path.join(out_dir, "data", f"{method}_quorum_sizes.csv"))
        
        # Starvation
        starv_orig = compute_starvation(df_orig)
        starv_rebase = compute_starvation(df_rebase)
        starv_orig["experiment"] = "Original"
        starv_rebase["experiment"] = "Rebase+Ledger"
        pd.concat([starv_orig, starv_rebase]).to_csv(os.path.join(out_dir, "data", f"{method}_starvation.csv"), index=False)
        
        # Efficiency
        s_orig = safe_load_json(os.path.join(orig_dir, method, "summary.json"))
        s_rebase = safe_load_json(os.path.join(rebase_dir, method, "summary.json"))
        _, _, c_orig, w_orig = compute_cumulative_stats(df_orig)
        _, _, c_rebase, w_rebase = compute_cumulative_stats(df_rebase)
        
        for s, exp, cnt, wt in [
            (s_orig, "Original", [c_orig[i][-1] for i in range(4)], [w_orig[i][-1] for i in range(4)]), 
            (s_rebase, "Rebase+Ledger", [c_rebase[i][-1] for i in range(4)], [w_rebase[i][-1] for i in range(4)])
        ]:
            tokens = sum(l["total_tokens"] for l in s["learners"])
            elapsed = s["elapsed_s"]
            total_accepted = sum(cnt)
            eff_records.append({
                "method": method,
                "experiment": exp,
                "elapsed_s": elapsed,
                "total_tokens": tokens,
                "tokens_per_sec": tokens / elapsed if elapsed else 0,
                "tx_bytes": s.get("wire_sent_bytes", 0),
                "rx_bytes": s.get("wire_received_bytes", 0),
                "accepted_contributions": total_accepted,
                "accepted_per_sec": total_accepted / elapsed if elapsed else 0,
                "jains_fairness_counts": jains_index(cnt),
                "jains_fairness_weights": jains_index(wt)
            })

    eff_df = pd.DataFrame(eff_records)
    eff_df.to_csv(os.path.join(out_dir, "data", "efficiency_metrics.csv"), index=False)
    
    for method in ["decoupled_heloco", "decoupled_diloco"]:
        plot_efficiency_tradeoff(eff_df, method, out_dir)
        
    analyze_evaluations(orig_dir, rebase_dir, out_dir)
    
    # Write Report
    report_path = os.path.join(out_dir, "README.md")
    with open(report_path, "w") as f:
        f.write(f"""# Rebase + Ledger Experimental Analysis

## Data Provenance
*   **Original Run:** `{orig_dir}`
*   **Rebase+Ledger Run:** `{rebase_dir}`

The analysis script verified that both experiments started from the identical initialization hash, evaluated against identical validation caches, and exactly matched in all control parameters (quorum size, scheduler, stopping criterion, slowness factors) except the `rebase` and `ledger` flags.

## Generated Figures

1. **`[method]_participation_heatmap.[png/pdf]`**
   * *Scientific Question:* When and how frequently did each worker participate in synchronization?
   * *Data Source:* `{method}/syncs.csv` (`learner_ids`).

2. **`[method]_cumulative_contributions.[png/pdf]`**
   * *Scientific Question:* How does the influence of each worker accumulate mathematically over synchronization steps?
   * *Data Source:* `{method}/syncs.csv` (`learner_ids`, `weights`).

3. **`[method]_time_behavior.[png/pdf]`**
   * *Scientific Question:* Does Rebase+Ledger delay global sync pacing, and when do slow workers contribute?
   * *Data Source:* `{method}/syncs.csv` (`sync_step`, `elapsed_s`, `learner_ids`).
   * *Consolidated Information:* Shows overall wall-clock synchronization timeline alongside slow-worker contribution rates, replacing independent throughput plots.

4. **`[method]_efficiency_tradeoff.[png/pdf]`**
   * *Scientific Question:* What is the system cost of Rebase+Ledger in time, throughput, and communication?
   * *Data Source:* `{method}/summary.json` (`elapsed_s`, `total_tokens`, `wire_sent_bytes`, `wire_received_bytes`).
   * *Presentation:* Side-by-side axes mapping incompatible units (Seconds, Tokens/Sec, GB, Aggregations/Sec) to evaluate resource trade-offs.

5. **`evaluation_loss_comparison.[png/pdf]`**
   * *Scientific Question:* Does the inclusion of delayed, rebased gradients improve or degrade the final model generalization per language?
   * *Data Source:* `evaluation_{{lang}}/evaluation.json` (`loss`).
   * *Limitation:* The experiment did not save intermediate checkpoints, precluding plotting convergence versus wall-clock time. This plot evaluates final convergence only.

## Generated Data (CSVs)
*   **`[method]_quorum_sizes.csv`**: Distribution of learners aggregated per commit.
*   **`[method]_starvation.csv`**: Maximum synchronization gaps per learner.
*   **`efficiency_metrics.csv`**: Tokens/sec, TX/RX bytes, Jain's fairness index, and wall-clock time metrics.
*   **`evaluation_summary.csv`**: Parsed validation loss and accuracy scores.
""")
        
    print(f"Analysis complete. All outputs saved to {out_dir}")

if __name__ == "__main__":
    main()
