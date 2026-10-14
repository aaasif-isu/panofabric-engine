# Rebase + Ledger Experimental Analysis

## Data Provenance
*   **Original Run:** `/lustre/hdd/LAS/jannesar-lab/aaasif/panofabric_run/panofabric-engine/outputs/method_comparison/20261013T094124-9a2f23`
*   **Rebase+Ledger Run:** `/lustre/hdd/LAS/jannesar-lab/aaasif/panofabric_run/panofabric-engine/outputs/method_comparison/20261013T094907-7cfedb`

The analysis script verified that both experiments started from the identical initialization hash, evaluated against identical validation caches, and exactly matched in all control parameters (quorum size, scheduler, stopping criterion, slowness factors) except the `rebase` and `ledger` flags.

## Generated Figures

1. **`[method]_participation_heatmap.[png/pdf]`**
   * *Scientific Question:* When and how frequently did each worker participate in synchronization?
   * *Data Source:* `decoupled_diloco/syncs.csv` (`learner_ids`).

2. **`[method]_cumulative_contributions.[png/pdf]`**
   * *Scientific Question:* How does the influence of each worker accumulate mathematically over synchronization steps?
   * *Data Source:* `decoupled_diloco/syncs.csv` (`learner_ids`, `weights`).

3. **`[method]_time_behavior.[png/pdf]`**
   * *Scientific Question:* Does Rebase+Ledger delay global sync pacing, and when do slow workers contribute?
   * *Data Source:* `decoupled_diloco/syncs.csv` (`sync_step`, `elapsed_s`, `learner_ids`).
   * *Consolidated Information:* Shows overall wall-clock synchronization timeline alongside slow-worker contribution rates, replacing independent throughput plots.

4. **`[method]_efficiency_tradeoff.[png/pdf]`**
   * *Scientific Question:* What is the system cost of Rebase+Ledger in time, throughput, and communication?
   * *Data Source:* `decoupled_diloco/summary.json` (`elapsed_s`, `total_tokens`, `wire_sent_bytes`, `wire_received_bytes`).
   * *Presentation:* Side-by-side axes mapping incompatible units (Seconds, Tokens/Sec, GB, Aggregations/Sec) to evaluate resource trade-offs.

5. **`evaluation_loss_comparison.[png/pdf]`**
   * *Scientific Question:* Does the inclusion of delayed, rebased gradients improve or degrade the final model generalization per language?
   * *Data Source:* `evaluation_{lang}/evaluation.json` (`loss`).
   * *Limitation:* The experiment did not save intermediate checkpoints, precluding plotting convergence versus wall-clock time. This plot evaluates final convergence only.

## Generated Data (CSVs)
*   **`[method]_quorum_sizes.csv`**: Distribution of learners aggregated per commit.
*   **`[method]_starvation.csv`**: Maximum synchronization gaps per learner.
*   **`efficiency_metrics.csv`**: Tokens/sec, TX/RX bytes, Jain's fairness index, and wall-clock time metrics.
*   **`evaluation_summary.csv`**: Parsed validation loss and accuracy scores.


## How to run

python analysis/scripts/run_analysis.py \
  --original outputs/method_comparison/20261013T094124-9a2f23 \
  --rebase-ledger outputs/method_comparison/20261013T094907-7cfedb \
  --output analysis/results/oct13_comparison
