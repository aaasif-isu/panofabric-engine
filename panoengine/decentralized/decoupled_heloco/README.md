# Decoupled HeLoCo and DiLoCo — Phases 1–9

This package adds a separate experiment entry point. The conventional outer
optimizer implementations in `heloco.py`, `async_diloco.py`, and
`parameter_server.py` are preserved. Launchers and default configurations live
in `run_script/`; relative data and output paths resolve from the repository root.

Implemented: configuration validation, routing to the original launcher for
HeLoCo/DiLoCo/MLA, deterministic fragments, dense fragment state, a dense
learner adapter, a round-robin quorum planner, and an in-process HeLoCo syncer
with tensor correction, per-tensor momentum, and look-ahead dispatch.
The CPU process demo also uses real localhost TCP and timed quorum/grace control.
Phase 7 adds dense one-GPU learners for the 15M Llama recipe. Phase 8 evaluates
the saved global model on held-out C4 batches. Phase 9 adds a distinct Nesterov
outer optimizer for decoupled DiLoCo and a matched-config preparation command.
The HeLoCo GPU path and held-out evaluator have been exercised on the user's
server. The new DiLoCo mode still needs its GPU integration run. General
production/FSDP support remains pending.

## Current audit fixes

The phase sections below retain development history; these settings describe
current behavior and supersede earlier token-weight and fixed-grace defaults.

- Failed broadcasts no longer gate subsequent outer updates. The syncer retains
  one latest dispatch per fragment; newer revisions supersede missed older ones.
  Retries never repeat an outer optimizer step. Snapshots still require a healthy
  quorum on the current baseline revision.
- During training, a TCP disconnect or learner exit marks only that endpoint
  unavailable. Remaining learners continue while at least `min_quorum` endpoints
  remain available. Final drain, pause and stop check surviving learners.
  Below quorum the coordinator fails explicitly. Initial startup is still strict:
  every configured learner must complete the initial handshake.
- Degraded runs save a global checkpoint and identify failed learners in
  `summary.json`. The matched-comparison launcher rejects them because the
  original all-learner token budget was not completed.
- `RemoteLearner.reconnect(transport, metadata)` and
  `DecoupledSyncer.reconnect_learner(endpoint)` support explicit reattachment and
  catch-up to the latest dispatches, including HeLoCo lookahead. The caller must
  authenticate the connection. The GPU launcher does not automatically restart
  trainers or accept reattachments; lost inner optimizer/data-loader state and
  durable distributed checkpoint recovery remain unimplemented.

Both YAML entry points expose the same decoupled settings:

```yaml
decoupled:
  weighting: tokens_squared_per_step  # normalized tokens^2 / local_steps
  merge: paper_rda                    # weighted_average or rda are optional
  adaptive_grace: true
  timing_ema_alpha: 0.2
  heloco:
    correction_order: merge_then_correct  # optional correct_then_merge
```

`rda` averages each parameter tensor's radii and unit directions separately.
`paper_rda` uses arithmetic averaging for recognized token-embedding names
(`embedding`, `tok_embeddings`, `embed_tokens`) and RDA for other tensors. Check
these names when adapting to a different model. A numerically canceled mean
unit direction produces zero rather than a noise-amplified update. The supplied YAML uses paper_rda; choose weighted_average to reproduce the
previous arithmetic-averaging runs.

In `merge_then_correct`, one merged gradient is corrected against the old
momentum. In `correct_then_merge`, each contribution is corrected against that
same old momentum, then merged. Both modes advance outer momentum exactly once
per quorum update and apply `rho` exactly once. The nonlinear modes can differ;
neither is claimed to improve convergence without experiments. Disabling
correction/lookahead still leaves HeLoCo's EMA momentum convention, so use
HeLoCo ablations to isolate correction effects from the DiLoCo sum buffer.

Adaptive grace uses `factor * max(0, tau * step_time - quorum_time - sync_time)`.
Step durations are estimated from wall time and monotonic lifetime step reports
per learner, including deliberate pacing; the kth-fastest observed pace estimates
quorum compute time. Capture-quorum latency and outer-update-to-queue-ACK latency
are tracked with EMAs. No measured compute pace means zero added grace. The
current capture latency is also used to prevent its EMA understating the cost.
This is an estimate of available slack, not a guarantee under sudden network or
compute slowdowns. `adaptive_grace: false` restores the fixed grace policy.

The YAML selects the actual scheduler used by all decoupled entry points:

```yaml
decoupled:
  scheduler: paper_offsets  # or round_robin
  sync_period: null        # H; defaults to P=num_fragments
  fragment_offsets: null  # t_p in fragment order; defaults to floor(p*H/P)
  overlap_steps: 2         # tau: communication/slack budget
  min_local_steps: null    # paper_offsets defaults to 1; round_robin to tau
```

`paper_offsets` selects a fragment when `t mod H = t_p`, starting at t=1.
Offsets must be distinct integers in [0,H), and H must be at least P. For
P=2, H=6 and offsets [0,2], updates target fragment 1 at t=2, fragment 0 at
 t=6, fragment 1 at t=8, and fragment 0 at t=12. `global_step` records this
schedule slot; `sync_step` separately counts committed updates. Missing quorum
and failed captures retry the same slot. Slots are paced using observed quorum
compute time; sync_interval supplies the bootstrap estimate. Attempts are paced
from their start rather than adding a new interval after every commit.

`round_robin` preserves the previous committed-update order and interval pacing.
Set sync_period and fragment_offsets to null in that mode; incompatible settings
and unknown scheduler names fail validation rather than being silently ignored.
Neither mode artificially delays received updates by tau local steps: learners
apply them at training boundaries.

This implements the paper's offset selection, not its entire concurrent runtime.
There is still one active fragment capture, an estimated wall-clock schedule,
contiguous fragmentation, and fixed learner budgets rather than a shared global
T stopping rule. Concurrent overlapping fragment captures and a broadcast global
training clock remain necessary for full Algorithm 1/2 fidelity. Decoupled HeLoCo
is an extension of the paper's DiLoCo outer optimizer. No fragment revisions are
skipped on capture timeout or quorum loss.

For the first GPU check, run one method before running the five-method comparison:

```bash
python run_script/run_decoupled_heloco.py --check-config
python run_script/run_decoupled_heloco.py --check-training
python run_script/run_decoupled_heloco.py --method decoupled_diloco
python run_script/run_method_comparison.py --check-config
```

These checks need the existing project environment and GPU allocation for actual
training. CPU tests cover hard learner crashes over real TCP, multiple missed
fragment cycles and catch-up, weighting, RDA, correction ordering, and timing.
They do not establish CUDA compatibility, convergence, or a performance gain.

From the repository root, using your project environment:

```bash
python run_script/run_decoupled_heloco.py --check-config
python run_script/run_decoupled_heloco.py --dry-run
python run_script/run_decoupled_heloco.py --method heloco --dry-run
python run_script/run_decoupled_heloco.py --smoke-test
```

Validation and routing previews need only Python and PyYAML. They do not query
GPUs, download assets, import torch/torchft, or instantiate a model. A valid
configuration is not a hardware-readiness or convergence check. This dry run
shows delegation and configuration, not full role commands or a computed token
budget. Actual baseline training uses the unchanged launcher's normal preflight.

To run an existing method with these separate settings:

```bash
python run_script/run_decoupled_heloco.py --method heloco
python run_script/run_decoupled_heloco.py --method diloco
python run_script/run_decoupled_heloco.py --method mla
```

Use `heloco/bin/python` in those commands if that is your project environment.
Choose the algorithm through the YAML's `method` or `--method`. The `run`
section accepts the existing launcher's option names, except `methods`,
`outer_method`, `config_file`, and `dry_run`, which the wrapper owns. Relative
assets and output paths are resolved from the repository root during execution.
The new file does not inherit experiment settings from `heloco.yaml`. To compare
against a previous run, copy that run's settings explicitly into `run`.

`run.num_fragments` configures the old push-based mechanism. Keep it at 1 for
the new mode; `decoupled.num_fragments` configures the new pull-based lifecycle.
The decoupled section is validated but has no effect on legacy runs. Phase 1
does not add quantization, transport, correction, scheduling, or a training hook.

Fragment coverage, snapshots, and revision handling are checked before adding
network traffic. The later learner adapter must respect optimizer-step boundaries and
island-wide FSDP coordination, not only protect a rank-local tensor with a lock.

One fragment with all workers is a useful check, but is not automatically
equivalent to normal asynchronous HeLoCo: merge-then-correct changes both update
order and the meaning of arrival weight `rho`. Fix work budgets, update order,
and normalization when comparing those methods.

Run the focused checks:

```bash
PYTHONPATH=.:run_script:tests/decoupled_heloco python -m unittest discover -s tests/decoupled_heloco -v
```

## Phase 2: fragment layout and state

`fragment_manager.py` now creates deterministic contiguous fragments without
splitting tensors. It preserves canonical parameter order, includes frozen
parameters, and uses the model's normal de-duplication of tied parameters.
An ordered name/shape fingerprint detects different layouts; local BF16 and
global FP32 copies intentionally have the same fingerprint. The manager holds
only metadata, so it can also describe a model built on the meta device.

`state.py` tracks each fragment's CPU FP32 baseline, local optimizer steps,
actual island tokens, highest received revision, and applied revision. Its
snapshots contain independent baseline/current copies and expose
`baseline - current` pseudo-gradients. Reading a snapshot does not reset work;
the baseline advances when a newer global replacement is actually applied.
This is an explicit initial lifecycle choice, not a claim that the referenced
Decoupled DiLoCo repository uses identical pull/offload semantics.

Incoming updates only enter a queue. Older/duplicate revisions are ignored;
the newest pending revision is applied at a training-thread step boundary.
Only that fragment's weights, baseline, and counters change. Baselines reflect
the local dtype rounding, so receiving FP32 weights into a BF16 model does not
create a false nonzero pseudo-gradient before further training.

The state primitives currently support dense, unsharded tensors. They do not
attach themselves to the existing trainer, perform FSDP collectives, reset
local optimizer state, preserve work overwritten by a replacement, or advance
baselines at pull time. The upcoming learner/syncer must coordinate snapshots
and apply boundaries across island ranks and deduplicate pull requests.

Run the same unittest command above using your existing PyTorch environment.
The new tests include actual CPU SGD steps, snapshot isolation, revision
reordering, fragment-specific resets, and BF16 baseline consistency. No GPUs,
torchft, or torchtitan are required for these tests. The existing launcher
continues to reject actual decoupled training until integration is implemented.

## Phase 3: learner adapter and round-robin scheduler

`learner.py` connects explicit successful optimizer steps to all fragment work
counters. Network/control threads can queue replacements and request snapshots;
only the owning training thread may copy live parameters at an idle boundary.
The intended trainer integration pattern is:

```python
with learner.training_step(tokens=actual_island_tokens):
    optimizer.zero_grad()
    loss = compute_loss(batch)
    loss.backward()
    optimizer.step()
```

The context assumes exactly one successful optimizer update. For AMP-skipped
steps, use `begin_step()` followed by `end_step(0, completed=False)` explicitly.
Failed contexts are not counted. The adapter does not execute the optimizer,
alter its moment buffers, or infer tokens from microbatch shapes.

Snapshot requests return futures without reading model weights in the receiver
thread. A capture happens only once the requested baseline revision and minimum
local-step count are available at a safe training boundary. If a newer baseline
has already been applied, the future fails with `StaleSnapshotRequest` instead
of silently returning a different update. Replacements are applied before new
captures at each boundary.

Use monotonically increasing integer-string request IDs, e.g. `"0"`, `"1"`.
Retrying a retained ID with identical arguments returns the same future and
capture. Treat snapshot tensors as read-only, and call `release_snapshot(id)`
after consumption/serialization. Released IDs cannot be replayed. The default
one-request limit bounds the learner's snapshot cache to one fragment; external
holders of futures/snapshots must also release their references. The upcoming
syncer still needs its own deduplication of consumed updates.

`scheduler.py` plans fragment `sync_step % num_fragments`. It includes every
ready worker in deterministic learner-ID order, requiring at least `min_quorum`.
A worker is ready only after `overlap_steps`, with nonzero token work, at the
requested global fragment revision, and with no unapplied newer replacement.
Planning without quorum leaves the fragment/clock unchanged; `commit(plan)`
advances only after the syncer commits an outer update. Metadata may change
between planning and capture, so snapshot requests recheck the same revision.

Neither component polls a network, implements a grace window, performs a global
outer update, or attaches itself to TorchTitan. Training does not wait for a
server response, but fragment copies still incur overhead at boundaries. This
is a dense single-process adapter; FSDP needs a coordinated island-wide adapter.

The focused unittest command above now checks 34 cases, including real SGD
steps, receiver-thread requests during backward, queued updates, snapshot
retries/cancellation, skipped steps, heterogeneous readiness, and commit order.
## Phase 4: in-process syncer and runnable CPU demo

`syncer.py` adds the complete dense fragment lifecycle. It plans a quorum,
requests safe-boundary captures, polls completed futures without waiting,
computes each `baseline - current`, merges the captured quorum, applies one
outer fragment step, increments that fragment's revision, and queues its new
weights for every registered learner. Learners apply those queued weights at
their own boundaries. The scheduler advances only on a committed outer update.

Historically, this phase used simple `FragmentSGD` in `optimizer.py` to isolate control-flow
correctness. It does **not** yet perform HeLoCo directional correction,
momentum, arrival scaling, or look-ahead dispatch. Merge weights are the
captured token counts normalized over contributors. This experimental baseline
is neither RDA nor the referenced implementation's `tokens**2 / local_steps`
weighting; the final aggregation policy remains a research choice.

Only one pull attempt is active at a time. Failed/canceled captures do not count
toward quorum. A lost quorum releases requests without changing global weights.
The updated dispatch is allocated before the outer state changes. After commit,
an outbox retains failed deliveries: `retry_broadcast()` retries queuing the
same revision, without repeating its outer update. A duplicate poll cannot
commit it twice. This is in-process retry handling, not durable crash recovery.

The global model and learners must start from identical weights. The syncer
checks layouts, owns its learners' request sequence, and has a single control
thread. The in-process interface still has no network, grace window, checkpoint
handshake, FSDP collectives, quantization, or production trainer integration.

Run the demo from the repository root in your existing PyTorch environment:

```bash
python run_script/run_decoupled_heloco.py --smoke-test
# For a larger quorum/overlap that needs more simulated work:
python run_script/run_decoupled_heloco.py --smoke-test --smoke-ticks 60
```

The demo reads the new YAML's island count, fragment/quorum/overlap settings,
seed, and outer learning rate. It always uses a tiny CPU token model and a
synthetic next-token task. It does not run `run.module`, the real dataset,
the GPUs, or the production token budget. Worker paces are deterministic tick
intervals `[1, 2, ..., islands]`, not measured wall-clock speedups. The output
shows fragment ID, revision, participating learners, local work, merge weights,
final revisions, and toy loss. Success requires every fragment to synchronize
and every learner to apply its final revision; insufficient ticks return 2.

Phase 4 introduced 46 tests. Those checks cover known SGD outer
updates, token weighting, nonparticipant broadcasts, independent revisions,
quorum cancellation, nonfinite captures, broadcast retry, and the CPU demo.

## Phase 5: merge-then-correct HeLoCo outer update

The default syncer and CPU demo now use `FragmentHeLoCo`. After the token-weighted
quorum merge, `correction.py` applies the existing HeLoCo tensor-block equations
once, using each tensor's own previous outer momentum. Aligned blocks pass
through, anti-aligned blocks shrink, and weakly aligned blocks rotate toward
momentum while preserving their magnitude. Missing or zero momentum passes
through before scaling by `rho`.

For the corrected merged gradient `G`, the selected fragment receives:

```text
m_next = mu * m + (1 - mu) * G
theta_next = theta - eta * (G + mu * m_next)
dispatch = theta_next - eta * mu * m_next
```

Global weights and dispatched look-ahead weights are separate copies. Learners
store the weights actually applied as their next baseline, including local
dtype rounding. Other fragments' weights and momentum do not change. Momentum
is FP32, allocated lazily for each participating tensor. All tensor results and
outgoing weights are checked and allocated before any global state commits.
Broadcast retries resend that dispatch without repeating the momentum update.

The standalone correction avoids importing the existing server stack on CPU.
Tests execute the two pure definitions directly from the unchanged `heloco.py`
and compare correction outputs, outer weights, and momentum against them. This
verifies the equations; token-weighted multiworker merge-then-correct is still
a different algorithm from normal HeLoCo's per-arrival correction and update.

`run.outer_lr` and `run.outer_momentum` set `eta` and `mu`. The new YAML section
`decoupled.heloco` contains the correction controls. Its `rho` defaults to 1.0
because one merged quorum produces one outer update. Keep legacy `run.rho`
null in the new mode; a normal-method override still uses it as usual. New-mode
correction is tensorwise after merging, so leave `run.correction_workers: all`
and `run.correction_scope: tensorwise`. Use `correction_enabled: false` to disable
directional correction while retaining `rho` and the momentum update. Use
`lookahead: false` to dispatch global weights directly.

```bash
# Default: tensor correction, outer momentum, and look-ahead
python run_script/run_decoupled_heloco.py --smoke-test
# Earlier simple-SGD lifecycle baseline
python run_script/run_decoupled_heloco.py --smoke-test --smoke-outer sgd
```

The HeLoCo demo also reads the outer momentum and new correction settings.
It still uses the tiny CPU synthetic task described above; it does not exercise
networking, timing/grace windows, TorchTitan, FSDP, or a production workload.
Production execution remains blocked. The focused suite now has 61 tests.
Next step: transport and coordinated trainer/FSDP integration.

## Phase 6: separate CPU learners over localhost TCP

Run from the repository root with the project environment active:

```bash
python run_script/run_decoupled_heloco.py --process-smoke-test
# To exercise more learners (must be at least decoupled.min_quorum):
python run_script/run_decoupled_heloco.py --process-smoke-test --process-learners 4
```

The launching process is the syncer. Two independently spawned learner processes
train the same tiny token model by default. The syncer listens on an automatically
chosen loopback port, checks a generated per-run handshake token and learner
identity, and explicitly sends identical initial weights before starting any
training. This demo's learner count overrides `run.islands` only for the demo.
It requires no extra networking package, GPU, torchft, or torchtitan.

`transport.py` uses framed TCP messages with a protocol version, basic mappings,
and independent CPU tensors. Deserialization explicitly uses `weights_only=True`.
Frames are capped at 32 MiB and inbound/outbound queues are bounded. Background
threads perform socket I/O and serialization; queue saturation fails explicitly.
The learner control thread queues pulls/replacements and reads metadata. Only
the training thread captures live weights or applies a replacement, at an idle
optimizer-step boundary. Learners keep training during capture/merge/grace, so
local work after a retained capture can be overwritten by replacement, as in the
earlier lifecycle; this phase does not add delta-preserving rebasing.

`RemoteLearner` supplies the same endpoint interface as the local adapter. A pull
returns a future without blocking. Released/duplicate responses cannot enter a
later cycle. Broadcasting retains the syncer's outbox until each learner reports
that it has queued that revision. A queue acknowledgement and actual application
are distinct: final application is verified through subsequent metadata reports.

`timing.py` gives the previously reserved timing settings these explicit semantics:

- The first cycle starts after `decoupled.sync_interval` seconds. Later cycles
  start no earlier than one interval after commit or capture timeout.
- Ready learners must meet `overlap_steps`, have token work, and have applied the
  requested fragment revision. Insufficient readiness keeps the same fragment.
- A requested capture quorum must arrive within one interval. Otherwise the
  attempt is released without a global update and can retry after one interval.
- Once a valid capture quorum arrives, a grace window of
  bounded by measured slack lets additional ready learners join. Set
  `adaptive_grace: false` to use the historical
  `sync_interval * grace_window_factor` behavior.
  Its expiry merges the valid captured contributors and releases unfinished
  requests. A zero factor commits immediately once a capture quorum arrives.

These are the prototype's scheduling choices, not an equivalence claim about
another Decoupled DiLoCo implementation. Training and network timing can change
the contributing work and toy loss even with the same seed.

The default two fragment cycles exercise momentum on every fragment. Set
`--process-cycles 1` for one cycle, or increase `--process-timeout 120` if startup
and the configured intervals need more time. The default total deadline is 60
seconds, including process startup. At completion, all learners must acknowledge
the final applied revisions, pause at a boundary, and stop cleanly. Startup,
connection, or deadline failures end the demo and clean up its children.

Output includes distinct process IDs, elapsed times, fragment revisions,
contributors, actual local work, merge weights, and toy loss. The printed wire
byte counts include framing, initialization, metadata/control traffic, and
tensors. They are transport diagnostics for this tiny model, not production
communication or performance measurements.

The focused suite now has 82 tests, including actual spawned processes and TCP
transfers, grace-window joins, zero grace, capture timeout/retry, queue backpressure,
stale responses, acknowledgement/application separation, and failure cleanup.
At Phase 6, production execution was still blocked. Multi-node deployment, reconnection,
durable recovery, chunked large-model transfer, quantization, TorchTitan, and
FSDP are not integrated. Next step: connect the adapter to real GPU training.

## Phase 7: first real GPU training integration

This adds a **GPU integration candidate** for the existing `llama3_15m` model
and C4 recipe. It has been checked against the repository's pinned TorchTitan
source and exercised with real CPU trainer processes. **CUDA forward/backward
and the installed TorchTitan environment still need validation on your GPU
server.** Passing CPU tests alone does not establish GPU correctness or
production readiness.

From the repository root, after applying the Phase 7 patch:

```bash
PYTHONPATH=.:run_script:tests/decoupled_heloco python -m unittest discover -s tests/decoupled_heloco -v
python run_script/run_decoupled_heloco.py --check-training
# Run this after the preflight passes:
python run_script/run_decoupled_heloco.py --training-timeout 1800
```

`--check-config` and `--dry-run` remain available without querying GPUs.
`--check-training` checks the allocated CUDA devices, imports the installed
training packages, builds the CPU global model, validates its fragment wire
sizes, and checks that all tokenizer IDs fit within the model's 2048 slots.
The tokenizer may use fewer slots; added/special tokens are included in the
check, and the actual count and maximum ID are printed. It launches no learners
and takes no optimizer steps. C4 access/download, dataset iteration and actual
GPU kernels are checked by the training run. Use the existing project `heloco`
environment with its pinned training dependencies; this patch adds no package
or installation requirement beyond those training dependencies.

The first supported scope is intentionally small:

- `module: models.llama3_small`, `config: llama3_15m`, from scratch.
- One dense GPU per island, on one node, with IID C4 shards. `c4_test` is also
  accepted if the pinned recipe's local test dataset is available.
- FP32 parameters and training, the preset's document-aware FlexAttention,
  ordinary AdamW and the preset LR schedule. FlexAttention still compiles its
  own kernel internally, so its first step can be slow. Model-wide compile,
  activation checkpointing and CUDA graphs are disabled for this check.
- No TorchFT manager, lighthouse or legacy parameter-server optimizer. The
  base TorchTitan `Trainer.train_step` performs forward/backward, clipping,
  AdamW and LR scheduling. The new adapter brackets each successful step.

The pinned Llama parallelization function always applies FSDP, even on one
device. The new recipe uses a dense parallelization function instead, plus the
partial-DTensor backend with all degrees equal to one. The adapter rejects
DTensor parameters. Multi-GPU islands/FSDP, larger models, language-specific
non-IID data, quantization and arbitrary `run.extra` overrides remain pending
and fail explicitly before launch. Existing model/config/trainer files and
the original launcher are unchanged.

The launcher assigns one visible CUDA device to each independent rank-0
process group. `run.gpus` contains **logical indices within the current
allocation**. For example, with `CUDA_VISIBLE_DEVICES=3,5`, use `[0,1]` for
two learners; the child masks become `3` and `5`. Match `run.islands`,
`run.gpus`, and the length of `run.island_slowness_factors` to your allocation;
keep `decoupled.min_quorum <= run.islands`. The default YAML requires four
visible GPUs. A slowness factor above one adds measured step-time pacing
outside the optimizer step. It is a simulation control, not faster compute.

Each learner's dataloader is built with `dp_world_size=run.islands` and
`dp_rank=learner_id`, even though its training process group has size one.
This avoids training every learner on the same shard after disabling TorchFT.
Weights come from one seeded CPU global model, explicitly transferred one
fragment at a time and acknowledged before training. Snapshot messages still
have the existing 32 MiB cap; preflight checks two independent FP32 copies per
fragment. Increase `decoupled.num_fragments` if the frame-size check fails.
General tensor chunking has not been implemented.

`training_adapter.py` counts valid labels (`label != -100`) from every batch
actually consumed by one optimizer step, including gradient accumulation.
Failed steps contribute no completed work. The plain pinned AdamW path has
no GradScaler skipped updates. Replacements and captures happen on the owner
thread at optimizer-step boundaries; tensor copies at those boundaries incur
CPU/GPU transfer costs. Serialization/socket I/O remain on separate threads.
Optimizer moments are retained across parameter replacements.

Each learner runs exactly `run.steps` successful optimizer steps. If
`tokens_per_parameter` is set, the budget resolves to
`ceil(tokens_per_parameter * parameter_count / (islands * batch * seq_len))`.
After its budget ends, a learner stops taking optimizer steps but continues
servicing captures and replacements at idle boundaries. Once all budgets
end, the syncer drains available quorums in round-robin order and stops at
the first fragment without a ready quorum; it does not skip that fragment.
Sub-quorum residual work can therefore remain unsynchronized. Final revision
counts may differ between fragments. All committed revisions must be applied
on every learner before clean shutdown. A run with any fragment still at
revision zero fails and asks for a larger step budget.

Each run has its own timestamped folder under `run.log_dir`, printed on launch:

- `learner_N/trainer.log`: startup, TorchTitan losses and error tracebacks.
- `learner_N/steps.csv`: local loss, gradient norm, tokens and applied revisions.
- `syncs.csv`: committed fragments, contributors, work counts and token weights.
- `summary.json`: success/failure, final revisions, learner work, capture timeouts
  and wire bytes including initialization and control traffic.
- `global_model.pt`: finite final **global** FP32 parameters on successful runs.
  These are the global optimizer weights, not dispatched look-ahead weights.
  This export omits outer/inner optimizer and dataloader state and cannot resume
  a distributed run.

Startup is bounded by `run.ps_timeout` (600 seconds by default). The total
process run, including startup, training and final drain, is bounded by
`--training-timeout` (1800 seconds by default). Failures return exit code 2,
write diagnostics, and stop the launched learners. Ctrl-C also cleans them up.

The focused suite now contains 94 tests, including accumulated token counts,
data shard identities, GPU allocation mapping, fragment initialization, and
real fixed-budget CPU processes exercising the new worker session and GPU
coordinator protocol. The next validation is the 30-step GPU run above. A
successful run verifies integration; convergence comparisons, performance,
FSDP, multi-node operation and durable recovery are later work.

### Phase 7 tokenizer validation fix

The original preflight incorrectly required a tokenizer vocabulary of exactly
2048 entries. The model has 2048 available slots; the pinned debug tokenizer
uses 2020 entries and is compatible. Preflight now validates the complete
vocabulary's token-ID range, BOS/EOS IDs and a sample encoding instead of
requiring an exact count. IDs outside `0..2047` still fail before launch, with
the actual vocabulary count and maximum ID included in the error. The model,
tokenizer files and training algorithm are unchanged. The focused suite now
has 100 tests, including smaller vocabularies, added special tokens, sparse
high IDs and genuinely incompatible tokenizers.

### Phase 8: shared global-model validation

`evaluate_decoupled_heloco.py` evaluates the reconstructed initial model and
one or more completed dense 15M global exports. It uses one visible GPU;
it starts no learners/syncer, constructs no optimizer, and takes no training
steps. It always reads **C4's English validation split**, not a learner's
training shard. The pinned text loader creates already next-token-aligned
labels and packed-document positions; the evaluator preserves those positions
and builds the model's document-aware FlexAttention masks.

Apply Phase 8 after Phases 1–7 and the Phase 7 tokenizer fix, then check:

```bash
git apply --check decoupled-heloco-phase8.patch
git apply decoupled-heloco-phase8.patch
PYTHONPATH=.:run_script:tests/decoupled_heloco python -m unittest discover -s tests/decoupled_heloco -v
```

Evaluate the two completed 500-step runs without retraining:

```bash
python run_script/evaluate_decoupled_heloco.py \
  --run-dirs \
    outputs/decoupled_heloco/20261007T043248-c283ae \
    outputs/decoupled_heloco/20261007T045422-aa1740 \
  --batches 100 \
  --device cuda:0
```

The GPU index is logical within your current `CUDA_VISIBLE_DEVICES`; this
evaluation does not require all four training GPUs. The first use of the
validation split may download data. Each model receives exactly the same
100 frozen CPU batches (51,200 valid targets for batch 1 / sequence 512).
The CLI permits 1–10,000 batches. Runs evaluated together must match their
model preset, tokenizer assets, initialization seed, batch size and sequence
length. Fragment counts and training speeds can differ.

The metrics are summed over valid target tokens, excluding `-100` labels:

- `loss`: token-weighted mean cross-entropy, in natural-log units.
- `perplexity`: `exp(loss)`, rather than an average of batch perplexities.
- `next_token_accuracy`: correct top-1 predictions divided by valid targets
  (a fraction in CSV/JSON; printed as a percentage).

Each evaluation writes a new directory under
`outputs/decoupled_heloco_evaluation/` and prints its location:

- `evaluation.json`: metrics, model/run identity, exact batch and tokenizer
  fingerprints, and per-learner/per-fragment unmerged work at shutdown.
- `evaluation.csv`: one row for the initial model and each global checkpoint.
- `validation_batches.pt`: the fixed inputs, aligned targets, document positions
  and provenance used for this evaluation.

To reuse that exact held-out set later, supply the cache's printed path:

```bash
python run_script/evaluate_decoupled_heloco.py \
  --run-dirs outputs/decoupled_heloco/20261007T045422-aa1740 \
  --batches 100 --device cuda:0 \
  --validation-cache outputs/decoupled_heloco_evaluation/YOUR_EVALUATION/validation_batches.pt
```

The evaluator refuses a cache with different tokenizer, split, tensor contents
or batch settings. `--output-dir` can choose a new output directory and refuses
an existing one, preventing accidental overwrites. Training checkpoints and
source logs are read-only.

The initial reference uses the same CPU initializer as the syncer. Future
training runs record an initial-parameter fingerprint, which validation checks.
Earlier exports lack that fingerprint: their reference is reconstructed from
the saved seed and depends on preserving the original model code and tokenizer
assets. Evaluation reports whether the fingerprint was actually verified.

This phase accepts `decoupled_heloco_global_v1` exports from successful runs;
Phase 9 extends that schema to tagged decoupled DiLoCo exports as well.
Conversions/exports for the legacy DiLoCo, HeLoCo and MLA workflows remain
subsequent work. The current
fixed-budget stopping rule is unchanged: sub-quorum tail work may remain
unmerged, and the evaluator reports that work without summing overlapping
fragment counters or claiming every local step reached the global model.

The focused suite now contains 112 tests. CPU tests verify token weighting,
target alignment, accuracy, masks, mode restoration, nonfinite rejection,
checkpoint isolation and batch-cache reuse. A coordinator fixture exercises
the CSV/JSON outputs using real CPU metrics with substituted runtime builders;
it does not constitute CUDA or real TorchTitan validation. The command above
is the next integration check on the GPU server. A lower held-out global loss
supports learning in this run; convergence and method comparisons still need
longer experiments and a consistent stopping/budget policy.

### Phase 9: matched decoupled DiLoCo pilot

Select `method: decoupled_diloco` or `--method decoupled_diloco` on the same
launcher. The inner AdamW recipe, CPU initializer, IID shard assignment,
FP32/FlexAttention execution, fragment layout, token-weighted quorum merge,
interval/grace timing, replacement boundaries, and final drain are shared
with decoupled HeLoCo. Only the outer optimizer changes:

```text
buffer_next = momentum * buffer + merged_pseudo_gradient
global_next = global - outer_lr * (merged_pseudo_gradient + momentum * buffer_next)
dispatch = global_next
```

The initial momentum buffer is the first gradient, matching PyTorch SGD with
`dampening=0`, `weight_decay=0`, and `nesterov=True`. At zero momentum this
reduces to SGD. Momentum is updated only for tensors in the committed fragment.
Weights, momentum, and outgoing copies are validated before any state changes.
DiLoCo performs no HeLoCo return correction and sends plain global weights;
`decoupled.heloco.*` applies only to HeLoCo. Disabling HeLoCo correction alone
remains a HeLoCo ablation: its EMA buffer and optional extra look-ahead differ
from the standard sum buffer used here. The new optimizer matches PyTorch's
Nesterov optimizer exactly in CPU tests across successive and interleaved
fragment updates. The existing legacy `diloco` route remains its original
asynchronous Delayed-Nesterov implementation.

The reference [Decoupled DiLoCo paper](https://arxiv.org/html/2604.21428v1)
describes fragment-wise Nesterov outer optimization and a weight proportional
to `tokens * (tokens / local_steps)`. Both decoupled methods now default to this paper weighting; set
`weighting: tokens` for the historical policy. With equal valid tokens per
local step, the normalized weights coincide. Adaptive grace is enabled by
default. A single CPU syncer, one-node TCP transport, round-robin fragment
updates, and fixed local budgets still define this prototype's scope.

Apply Phase 9 after Phase 8 and its tokenizer-fixture test fix:

```bash
git apply --check decoupled-heloco-phase9.patch
git apply decoupled-heloco-phase9.patch
PYTHONPATH=.:run_script:tests/decoupled_heloco python -m unittest discover -s tests/decoupled_heloco -v
```

Prepare a fresh config from an **intact completed run**. This reads its saved
`experiment.yaml`, `run-options.json`, and `summary.json`, checks their recipe
and step-budget consistency, and requires the global export to exist. It
copies the actual resolved budget, seed, topology, paces, tokenizer, model,
and all decoupled controls. It sets the new method and a separate output root.
The preparation command constructs no model or optimizer and needs no GPU.
It refuses to overwrite an existing config or write inside the reference run.

```bash
python run_script/prepare_decoupled_comparison.py \
  --reference-run outputs/decoupled_heloco/20261007T054425-2a0714 \
  --output-config decoupled_diloco.yaml

python run_script/run_decoupled_heloco.py --config-file decoupled_diloco.yaml --check-config
python run_script/run_decoupled_heloco.py --config-file decoupled_diloco.yaml --training-timeout 1800
```

The CPU simulation and separate-process smoke commands also select DiLoCo
from the method by default. `--smoke-outer` remains an explicit simulation-only
override. The GPU run prints its new timestamped directory under
`outputs/decoupled_diloco/`. Use that directory in a joint evaluation with the
existing HeLoCo run and reuse the exact frozen cache from its evaluation:

```bash
# Set this to the actual Logs: path printed by the completed DiLoCo run.
diloco_run=outputs/decoupled_diloco/YOUR_COMPLETED_RUN
python run_script/evaluate_decoupled_heloco.py \
  --run-dirs outputs/decoupled_heloco/20261007T054425-2a0714 "$diloco_run" \
  --batches 100 --device cuda:0 \
  --validation-cache outputs/decoupled_heloco_evaluation/20261007T054938-a92d7c/validation_batches.pt
```

New summaries record the method, outer momentum convention, and stopping
policy. Both decoupled modes use the same fixed local budgets and drain to the
first unavailable round-robin quorum. Unmerged tails remain explicitly reported
per fragment; final contributions can differ between runs. Matched processed
tokens alone do not imply matched incorporated work, outer-update count, or
wall time. Retain the original run and cache when comparing the methods.

The historical checkpoint format string remains `decoupled_heloco_global_v1`
for compatibility. New exports carry an explicit method tag, checked against
the saved experiment before loading. Older untagged exports are accepted as
HeLoCo only. One frozen evaluation stream can now contain both decoupled
methods. The initial fingerprint checks and read-only checkpoint/cache policy
are retained.

Phase 9 adds 13 tests, bringing the focused suite to 125: reference Nesterov
math, scalar expected updates, fragment isolation, failure atomicity, config
preparation/no-overwrite, method routing, real TCP process smoke, and real
CPU fixed-budget coordinator/export checks. The shared evaluator fixture now
evaluates both decoupled methods on one frozen stream and reuses its cache.
Actual CUDA DiLoCo training and held-out quality comparison are the next
integration checks on the server. Compatible exports and aligned numerical
recipes for the three legacy methods are the next comparison milestone.

### Phase 10: three original methods under the matched dense recipe

`run_matched_baselines.py` runs `heloco`, `diloco`, or `mla` using the same
dense FP32 Llama 15M, document-aware FlexAttention, plain AdamW, LR schedule,
CPU seed initializer, tokenizer, IID shard assignment, local batch/sequence
shape, step budget, and learner paces as the completed decoupled pilots.
It directly uses the repository's **original** `AsyncDiLoCo` HTTP client,
server classes, and outer optimizers. Existing tracked source files and the
original launcher remain unchanged. `run_decoupled_heloco.py --method diloco`
still delegates to that original launcher; use the new launcher for this
matched comparison.

The original launcher overrides asynchronous DiLoCo's outer LR to 0.07 and
defaults to BF16 wire transfers. The comparison deliberately uses the saved
common outer LR (0.7 for the pilot), and explicit full FP32 uploads/downloads
for all three baselines. This compares the algorithms at the matched recipe;
it does not reproduce the original launcher's LR override or provide a tuned
best-result comparison.

| Method | Original update/dispatch preserved in this launcher |
| --- | --- |
| `diloco` | `DelayedNesterovOptimizer`; milestone period = number of learners, as set by `run_heloco.ps_cmd`; plain global dispatch |
| `heloco` | `HeLoCoOptimizer` and tensorwise correction per arrival; arrival rho defaults to `1/sqrt(learners)` with the original launcher's six-decimal rounding; lookahead dispatch |
| `mla` | `MLAOptimizer` EMA update, without directional correction; plain global dispatch |

All three use whole-model exchanges (`run.num_fragments: 1`) after
`run.sync_steps` local optimizer steps. Each arrival commits an outer update;
there is no quorum averaging, grace batching, or DyLU window adjustment.
Fixed local budgets must be divisible by the window length. With four
learners, 500 steps each, and a 10-step window, expect 50 pushes per learner
and **200 whole-model server revisions**, rather than the decoupled runs'
fragment revision counts. The original asynchronous DiLoCo uses delayed
Nesterov; the decoupled DiLoCo prototype uses standard per-fragment Nesterov.
Keep those labels explicit when reporting results.

The coordinator waits until every learner has pulled and fingerprinted the
same revision-0 parameters before starting training. It removes the original
client's automatic optimizer hook and calls its original `_sync` at the same
local optimizer boundaries, so a rejected push or transport failure fails
this comparison rather than silently dropping a window. Inner optimizer
state survives replacements. Artificial slowdown is applied once per local
step, proportional to measured `train_step` time; blocking HTTP time is
excluded. After every learner finishes all windows, the coordinator checks
the exact expected revision count, freezes finalization, and requires every
learner to pull and fingerprint the final dispatch before clean exit.

The export is the **server's global parameters**, not HeLoCo's lookahead
dispatch. It uses `matched_global_v1`, with an explicit method tag, canonical
named parameters, one whole-model layout signature, and final revision.
It is for evaluation, not resumable training. Every run includes
`experiment.yaml`, `run-options.json`, `summary.json`, `global_model.pt`,
per-learner `steps.csv`, `windows.csv`, `trainer.log`, and the original
client's communication CSVs. Summary communication counters are protocol
bytes including initial/final pulls, excluding HTTP headers and heartbeats;
they are not directly equivalent to the decoupled TCP wire counters.

Apply after Phase 9:

```bash
git apply --check decoupled-heloco-phase10.patch
git apply decoupled-heloco-phase10.patch
PYTHONPATH=.:run_script:tests/decoupled_heloco python -m unittest discover -s tests/decoupled_heloco -v
```

Prepare the three configs from the intact, completed 500-step HeLoCo run.
The generator copies actual resolved recipe settings and local budgets,
including the legacy window H, and refuses to overwrite existing configs.
It does not modify `decoupled_heloco.yaml`, the reference run, or its cache.

```bash
for method in diloco heloco mla; do
  python run_script/prepare_decoupled_comparison.py \
    --reference-run outputs/decoupled_heloco/20261007T054425-2a0714 \
    --method "$method" --output-config "matched_${method}.yaml" || break
done

python run_script/run_matched_baselines.py --config-file matched_diloco.yaml --check-config
python run_script/run_matched_baselines.py --config-file matched_diloco.yaml --training-timeout 1800
python run_script/run_matched_baselines.py --config-file matched_heloco.yaml --training-timeout 1800
python run_script/run_matched_baselines.py --config-file matched_mla.yaml --training-timeout 1800
```

Run these sequentially in the same four-GPU allocation. Check the first
DiLoCo run before proceeding to the other two. Each prints its own `Logs:`
directory under `outputs/<method>/`. `--check-training` performs only CUDA,
dependency, tokenizer, and budget preflight; it launches no learner and does
not test dataset iteration or CUDA forward/backward.

The shared evaluator now accepts all five completed exports, while rejecting
unconverted checkpoints from the original workflow. Retain the existing
decoupled runs and frozen held-out cache. Set these variables to the exact
three new completed `Logs:` paths, then evaluate:

```bash
diloco_run=outputs/diloco/YOUR_COMPLETED_RUN
heloco_run=outputs/heloco/YOUR_COMPLETED_RUN
mla_run=outputs/mla/YOUR_COMPLETED_RUN
python run_script/evaluate_decoupled_heloco.py \
  --run-dirs \
    outputs/decoupled_heloco/20261007T054425-2a0714 \
    outputs/decoupled_diloco/20261007T060953-c3c852 \
    "$diloco_run" "$heloco_run" "$mla_run" \
  --batches 100 --device cuda:0 \
  --validation-cache outputs/decoupled_heloco_evaluation/20261007T054938-a92d7c/validation_batches.pt
```

Compare global held-out cross-entropy, perplexity, and next-token top-1
accuracy. The initial-model fingerprint and validation-batch hash must match.
Matched processed tokens do not equal matched incorporated work: these
baselines commit all complete windows, while decoupled runs retain their
reported per-fragment sub-quorum tails. Outer-update counts, communication
rules, and momentum conventions intentionally differ. A single seed and
500-step pilot checks the pipeline; multiple seeds, longer budgets, and
declared tuning/budget policies are needed for a performance claim.

The focused CPU suite has 132 tests. New checks cover original optimizer
class selection and first HTTP updates, FP32 wire selection, retained AdamW
moments, delayed-start initialization barriers, separate learner processes
for all three methods, exact window/revision accounting, final dispatch and
global export checks, startup failure cleanup, config generation and routing,
and all five exports sharing one frozen evaluation stream. Build-host HTTP
tests use the pure-Python HTTP/helper subset of the project's pinned TorchFT
source; no native TorchFT process groups or CUDA kernels are exercised there.
Real CUDA baseline training remains the integration check on the GPU server.

### One YAML and one command for the five-method comparison

Edit `run_script/method_comparison.yaml`, which starts with:

```yaml
method_run: [decoupled_heloco, decoupled_diloco, heloco, diloco, mla]
```

The supplied config uses four learners, 40 steps each, seed 42, and paces
`[1, 2, 3, 1]`. `run` contains the common recipe/budget and the original
methods' whole-model window. `decoupled` contains the two decoupled methods'
fragment/quorum controls. `evaluation` controls the final held-out evaluation.
Remove methods from `method_run` to run a subset. Use the exact underscore
names shown above. The single-method YAML and launchers remain available.

```bash
python run_script/run_method_comparison.py --check-config
python run_script/run_method_comparison.py --training-timeout 1800
```

The launcher validates ALL selected methods before training starts, runs each
sequentially on the same GPU allocation in a fresh process, records its exact
run directory, checks its budget and initialization fingerprint, and requires
successful completion before launching the next method. Each method uses the
same inner recipe and its existing matched outer path from Phase 10. The
deadline is per method, not for the whole five-method experiment.
`--dry-run` prints routing without importing Torch or launching processes.

With `evaluation.enabled: true` (default), the launcher then evaluates the
initial model and all global checkpoints on ONE frozen held-out C4 stream.
`evaluation.validation_cache: null` creates a new shared cache. Set it to the
path of a compatible `validation_batches.pt` to reuse the earlier stream.
An explicitly requested missing cache is rejected before training starts.
Set `evaluation.enabled: false` to train now and evaluate later using the
saved run directories and `evaluate_decoupled_heloco.py`.

The printed `Comparison output:` directory contains:

- `comparison.yaml`: the exact source config.
- `configs/<method>.yaml`: generated single-method configs.
- `<method>/`: each completed training run, including its global checkpoint.
- `<method>.log`: streamed launcher output for each method.
- `evaluation/`: the shared evaluation JSON/CSV and new frozen cache, if created.
- `comparison.csv`: initial/global loss, perplexity, next-token accuracy,
  processed tokens, local steps, outer-update count, and training time.
- `comparison.json`: the full combined evaluation and each training summary,
  including optimizer settings, stopping policy, and residual fragment work.

Existing runs and configs are never overwritten. A failed method or evaluation
stops the comparison and records the error in `comparison.json`, retaining
completed runs and logs. Process deadlines and interruption stop the launched
coordinator's entire fresh process group so its learners cannot continue into
the next method. This launcher makes the matched comparison convenient; it
preserves the algorithm/budget distinctions documented in Phase 10.

The combined installable update works on either the Phase 9 or Phase 10 code.
Extract its ZIP in the repository root and run `install_five_method_update.py`.
The installer checks both patches before choosing one, never forces an apply,
and changes only our added experiment files. It preserves the original
tracked code and your existing `decoupled_heloco.yaml`.

Eight new tests bring the focused suite to 140. They verify pure validation,
invalid later-method budgets before any launch, explicit run directories and
no overwrite, output streaming/deadlines/failure handling, and a full
five-method CPU training/evaluation sequence using real learner subprocesses,
TCP/HTTP transfers, optimizer steps, exports, and one frozen validation stream.
TorchTitan/CUDA builders in that final test are explicit CPU fixtures; GPU
execution still needs verification in the allocated server job.

## Measured convergence and coordinator resources

The supplied comparison YAML enables `monitoring`. `global_every_updates` controls
full global weight exports after that many committed outer updates; these are the
optimizer's global weights, not HeLoCo lookahead dispatches. `learner_every_steps`
controls each learner's own exports after completed optimizer steps. Both include
the actual shared initialization at zero and the final training snapshot. Exports
are immutable, evaluation-only and not resumable training checkpoints.

After training, the shared evaluator evaluates every export on the exact same
frozen held-out C4 batches. It verifies every step-zero parameter fingerprint and
reuses the identical measured initial loss. `evaluation/trajectory.csv` retains
loss, next-token accuracy, perplexity, step, training time, and processed tokens.
`convergence_plot.png` shows global held-out loss versus total observed tokens;
`global_loss_vs_time.png` shows it versus training elapsed time.
`learner_convergence.png` has one panel per method, showing all learners' held-out
losses. Local training losses remain available separately in learner steps.csv.
Neither x-axis calls a local step an outer round. Different methods need not have
different curves. Global token coordinates are asynchronous observations of
cumulative learner work, not a simultaneous barrier or merged-token accounting.

`central_memory.csv` samples Linux process RSS and process CPU time. It excludes
child learners and includes Python, threads, model/optimizer, buffers and monitoring
allocations. The sampled maximum is not a guaranteed capture of sub-sample peaks.
`central_processing.csv` measures outer apply (baseline), merge/correction/update
(decoupled), and dispatch snapshot construction or enqueue. Thread CPU and wall
seconds are separate; wall can include lock contention. Network waiting, complete
HTTP serialization, and end-to-end communication latency are not represented by
these compute phases. Protocol-byte totals retain each method's original accounting
scope, so they are not interchangeable with physical network-interface traffic.
`memory_footprint.csv` summarizes measured resources, training time, protocol bytes,
and global snapshot overhead. No theoretical memory or missing-value zero is used.

Evaluation runs afterward, but snapshot copies/writes happen during training and
can affect scheduling, memory and runtime; recorded time is instrumented runtime,
not uninstrumented throughput. For larger experiments increase snapshot intervals
and run an additional monitoring-disabled throughput comparison. At FP32, every
13.6M-parameter snapshot is about 52 MiB; storage scales with snapshots and learners.
Monitoring is opt-in for API-created configs and enabled in supplied YAMLs. Set
`monitoring.enabled: false` to disable snapshots and central resource recording.
Older runs cannot retrospectively produce these measurements.


### Completion endpoints and remaining paper-fidelity dependencies

Forced trajectory exports bypass the duplicate-update guard. They use a separate
`NNNNNNNN-final.pt` file, so the last update observation and the final completed
token/time observation both survive. The weights can be equal in these records:
local work after the last merge does not imply another global update. Evaluation
checks final tokens against each completed run/learner summary. Older unmarked
exports are labeled `legacy_unverified`; failed learners without a final export
are labeled `interrupted_learner`. Neither is presented as a verified endpoint.
Resource CSVs include completed observed tokens, and the resource plot flags
unequal totals. Equal totals alone do not match synchronization frequency, outer
work, hardware, or monitoring overhead.

The dependency order for further implementation is:
1. Implemented below: carry the scalar syncer clock in messages and make worker
   stopping/final drain follow it while retaining local-budget compatibility.
2. Permit concurrent transfers for different fragments, preserving per-fragment
   revision order, capture ownership, retries, and final checkpoint consistency.
3. Add CPU syncer replicas and their fragment all-reduce. The paper describes
   replicas maintaining global model and optimizer state; simply partitioning
   tensors among processes would be a different architecture.
4. Sample each syncer replica and report both peak per-replica RSS and aggregate
   RSS, including all-reduce bytes/time and snapshot overhead. Validate resource
   comparisons at explicit completed token budgets.

Shared-clock stopping and bounded concurrent captures are added in the following
incremental sections. The runtime still has one central syncer; endpoint repair
alone never changed the architecture or claimed a memory/runtime advantage.


### Shared syncer-clock stopping (incremental implementation)

The YAML selects the stopping rule independently of fragment order:
```yaml
decoupled:
  scheduler: paper_offsets
  stopping: syncer_steps       # or local_steps (existing behavior)
  syncer_steps: 100            # T; null in local_steps mode
  clock_lr_schedule: constant  # or local_horizon
```

T counts syncer schedule slots, not local optimizer steps or complete fragment
cycles. An update carries its committed slot in the same frame as its weights.
Learners apply both at a safe optimizer boundary and report `syncer_step` in
metadata and step CSVs. The clock never regresses. A terminal idle slot is sent
only after all committed fragment revisions have been applied by healthy peers.
A learner in an optimizer step completes that step, then stops starting local
work at the target; idle boundaries remain active for final delivery and shutdown.
Captures after T are prohibited. With sparse offsets, T need not update a
fragment; the final idle clock still reaches every healthy learner.

In clock mode `run.steps` no longer caps local work. The default clock LR policy
is constant (zero warmup and minimum multiplier 1), preventing the old local
horizon from silently zeroing learning rates. Select `local_horizon` to retain
the preset schedule with `run.steps` as its horizon. Local-budget mode retains
the original LR schedule and token budgets. The normal training timeout remains
the failure bound; no extra local cap is silently substituted for the clock.

Five-method comparisons retain `stopping: local_steps`. Clock stopping is
accepted only for decoupled-only comparisons; such runs can consume different
token totals and are not presented as matched-token results. Summaries record
the target and observed syncer clock, and `steps_per_learner` is null in clock
mode, with actual local steps/tokens retained per learner.

This clock increment adds the shared scalar syncer clock and stopping path.
Concurrent captures are added below; syncer-shard all-reduce, vector-clock
checkpoint recovery, and automatic trainer restart remain unimplemented.


### Bounded concurrent fragment captures

```yaml
decoupled:
  max_inflight_captures: 2  # 1 retains serial captures; valid range 1..num_fragments
```

Both `paper_offsets` and `round_robin` use the selected capture limit. Scheduling
reservations advance separately from committed outer steps, so different
fragments can be requested and transferred before the previous outer update
finishes. Only one capture for each fragment may be retained. Completed captures
waiting for an older slot count against the limit. The learner cache uses the
same limit; initialization and the existing per-frame size cap are unchanged.

Outer commits remain in schedule order. A younger completed capture waits
without expiring while an older capture is pending; it cannot advance the shared
clock or cause early learner stopping. Each capture has an independent capture
timeout and grace window. A failed/timed-out slot releases only its requests,
retains its schedule position, and retries with fresh request IDs. Younger
snapshots remain cached. Each cohort is sealed at its own grace deadline,
excluding unfinished requests and learners that become ready after the window,
even while the capture waits behind an older slot. Late contributors within an
open window use fresh per-learner IDs, avoiding
watermark rejection when that learner already captured a younger slot.

Reattachment removes the old endpoint's captures from every reserved slot before
it can contribute again. All-slot cancellation resets only reservations, without
changing model weights, momentum, revisions, or the committed clock. Individual
pipeline cancellation must retain its slot. A shared-clock target also limits
reservations; no capture is admitted beyond T. Final export/shutdown waits for
all reserved slots through T to commit and for healthy learners to apply the
committed revisions. Local-budget draining remains supported.

`capture_events.csv` records begin, quorum-ready, prepared, retry, and commit events with
schedule slots, fragments, elapsed seconds, and retained capture counts. The run
summary includes `max_inflight_captures` and measured `peak_inflight_captures`.
These are actual runtime observations; transfer bytes remain the measured
protocol counters. A single TCP writer per learner still serializes packets on
that connection. Concurrent capture lifetimes do not imply multiple physical
network links or parallel outer optimizer kernels. More retained snapshots can
increase coordinator/learner memory; measured RSS, rather than a memory savings
assumption, remains the comparison source.

Tests deliberately delay one fragment's TCP snapshot while another is delivered,
verify two retained captures for both methods, verify ordered clocks and sparse
stopping, compare retry results against one numerical outer update, and exercise
local-budget drain with a hard learner-process crash. CPU fixture delays are
test-only; the GPU runtime injects no synthetic delay.

The supplied YAML keeps limit 1 for compatibility. Set it to 2 in the selected
single-method or comparison YAML to enable this pipeline. Syncer replicas and
fragment all-reduce are the next architecture step; the central syncer is still
unsharded.


## Incremental syncer sharding (after concurrent captures)

Apply `decoupled-syncer-sharding.patch` after the final-endpoints, shared-clock,
and concurrent-capture patches, in that order. The default `syncer_shards: 1`
keeps the existing central optimizer. To enable persistent CPU replicas:

```yaml
decoupled:
  syncer_shards: 3  # normally one replica per learner
  syncer_timeout: 60.0
  max_inflight_captures: 2
```

Replicas are independent spawned CPU processes on the coordinator's host.
Each owns the full global FP32 model and outer momentum. The coordinator owns
capture admission, normalized contribution weights, and ordered commits;
it routes each captured contribution to `learner_id % syncer_shards`.
Every replica produces local weighted fragment statistics, then participates
in a reduce-to-replica-zero/broadcast all-reduce through direct IPC pipes.
The coordinator does not perform the numerical reduction or outer update.
Weighted averaging reduces weighted deltas; RDA reduces weighted directions
and radii before normalizing the combined direction. `paper_rda` preserves
embedding averaging, and both HeLoCo correction orders retain their meaning.
All replicas execute the same outer update and retain identical state, within
floating-point tolerance of the central reduction. Only the selected fragment
is reduced; full-model exports are separate, optional monitoring/checkpoint work.
Each replica uses one Torch compute thread.

Unavailable learners contribute zero; their replicas remain alive and participate
in every collective. A failed or stalled **syncer replica** aborts the run,
cleans up all replica processes, and cannot advance the coordinator revision
or clock for the failed update. This is fail-stop behavior, not distributed
rollback or automatic replica recovery. Checkpoints remain non-resumable.
Concurrent captures retain their capacity limit and commit order; outer
collectives execute sequentially in that order.

This is a same-host implementation of the paper's replica/fragment-all-reduce
architecture. Cross-host syncer placement and a network collective backend are
not implemented. Learner TCP ingress and dispatch still pass through the
coordinator, so this patch does not establish decentralized network bandwidth
or guaranteed speedup. The coordinator also retains initialization-model
references held by its caller and pending captured fragments; replica state
and IPC serialization add memory. It does not promise reduced aggregate RSS.

With monitoring enabled, `syncer_memory.csv` records coordinator and replica
RSS/CPU plus aggregate RSS. Replicas sample their own Linux RSS and process CPU
every 0.1 seconds; aggregate rows combine the coordinator sample with latest
replica samples, so replica observations can be up to 0.1 seconds older. Aggregate rows
are omitted if any replica observation is missing. RSS sums can double-count
shared pages. Existing `central_memory.csv` remains coordinator-only.
`syncer_replica_processing.csv` records per-update/per-rank compute CPU and
wall time, including collective waiting. `collective_tensor_bytes` describes
logical tensor payload per rank (including control scalars), not actual wire
bytes or serialization overhead. The resource comparison plots aggregate RSS
and coordinator-thread-plus-replica processing CPU; it skips sharded runs lacking an
aggregate measurement. Coordinator processing wall time already includes the
replica round trip and is not summed again with replica waiting time.

Validation covers actual spawned replicas, all merge modes, both HeLoCo
correction orders, momentum across interleaved fragments, empty replica
contributions, killed/stopped replicas, and TCP shared-clock/concurrent-capture
runs with a learner crash. GPU execution and cross-host scaling require
separate validation on the training cluster.


## Learner shutdown deadline fix (after syncer sharding)

Apply `decoupled-learner-shutdown.patch` after the syncer-sharding patch.
No YAML changes are required. A stopped protocol ACK confirms completion of
training, final boundary handling, and the worker's final trajectory export;
it does not confirm that trainer/CUDA/distributed cleanup and process exit
have finished. Both decoupled and original-method coordinators previously
allowed at most five seconds for process exit after protocol/file completion.
They now poll all eligible learners under the existing shared
`--training-timeout` deadline. A nonzero exit fails promptly with learner ID,
PID, exit code and log path. Slow teardown reports pending learner IDs every
ten seconds; reaching the overall deadline still fails and triggers the
existing process cleanup. No completion is declared while a required learner
process is still running. Training, merge, stopping, and budget semantics are
unchanged. CPU regressions exercise six-second post-ACK teardown with real
sharded/concurrent-capture sessions for both decoupled methods, deadline expiry,
and a failed learner alongside a still-running process. GPU teardown latency
and any actual teardown deadlock require verification on the GPU node.

### TCP congestion and progress reports

Ordinary unsent progress reports use one replaceable queue slot. Reports contain
cumulative local steps/tokens and complete fragment metadata; replacing a report
does not discard training or snapshot counters. Intermediate report arrival times
can differ, so asynchronous cohort selection can differ under congestion.
Snapshots, updates, releases, ACKs, errors and lifecycle messages retain FIFO
delivery and use bounded capacity waits. Socket failures and expired deadlines
still fail; degraded runs remain excluded from matched-budget comparisons.
The GPU run deadline bounds required-message enqueue waits and learner shutdown.
The coordinator pumps learner messages on its owning thread while waiting for
replica IPC; this does not start another synchronization or reorder commits.

Each GPU learner and the coordinator save `transport_diagnostics.json`, including
queue peak, replaced progress reports, congestion retries, bytes and cumulative
serialization/socket-send time. Queues retain at most the configured required
frame capacity plus one unsent progress report and one writer frame. Reader
congestion retains one decoded frame while waiting for incoming queue capacity.
No YAML changes are required.
