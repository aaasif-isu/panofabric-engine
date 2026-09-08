# Roadmap: Converting `run_heloco.py` from Window-Synchronous to Fully Asynchronous

## Executive Summary

Your current setup uses **window-synchronous training** (all workers must complete their H-step window before any sync). Converting to **fully asynchronous** means:

- Workers train **completely independently** (no barriers)
- Each worker pushes pseudo-gradients **whenever it finishes a window** — not waiting for others
- Server accepts arrivals **on-the-fly** and applies HeLoCo corrections + look-ahead immediately
- Slower workers don't stall faster ones

**Good news:** panoengine already has `AsyncDiLoCoServer` and `HeLoCoServer` — the core async infrastructure is in place. The work is mostly **integration into torchtitan's training loop**.

---

## Current Architecture (Window-Synchronous)

```
┌─────────────────────────────────────────┐
│  run_heloco.py (launcher)               │
│  - Spawns N islands as separate processes
│  - Each runs: torchrun -m torchtitan.train
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  torchtitan.train                       │
│  - Fault tolerance manager (torchft)
│  - semi_sync_method="heloco"
│  - sync_steps=10 (barrier every 10 steps)
└─────────────────────────────────────────┘
                    ↓
        ┌───────────────────────┐
        │  All islands push     │  ← All must finish window
        │  simultaneously       │
        └───────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  panoengine.decentralized.parameter_server
│  - outer_method=heloco
│  - Receives batch push/pulls
│  - Applies HeLoCo corrections
│  - Sends updated weights back
└─────────────────────────────────────────┘
```

**Key constraint:** Synchronization barrier at every `sync_steps`.

---

## Target Architecture (Fully Asynchronous)

```
Island-0 (fast):  Step 1..10 → push(t=1.2s, staleness=0)
Island-1 (slow):  Step 1..3  → push(t=2.5s, staleness=?)
Island-2 (med):   Step 1..7  → push(t=1.8s, staleness=1)

All pushes go to SAME server, NO barriers.
Server applies HeLoCo immediately, weights only stale by 1-2 steps.
```

**Key difference:** Arrivals are **independent**, server is **always available**.

---

## Three Implementation Paths

### **Path A: Use Existing AsyncDiLoCo** ✅ **RECOMMENDED**

Use `panoengine.decentralized.AsyncDiLoCo` (worker) + `HeLoCoServer` (server).

**Pros:** Tested code, low risk
**Cons:** Need new trainer entry point (bypass torchtitan's FT manager)
**Effort:** 1–2 weeks

---

### **Path B: Patch torchtitan's FT Manager**

Add `semi_sync_method=async_heloco` to torchft.

**Pros:** Unified interface
**Cons:** Upstream coupling, harder to maintain
**Effort:** 3–4 weeks

---

### **Path C: Hybrid — Staleness-Aware Within Window**

Keep barriers, but track worker progress and weight slower workers less.

**Pros:** Minimal code changes
**Cons:** Not fully async, limited benefit
**Effort:** 3–5 days

---

## Phase-by-Phase Roadmap for Path A

### **Phase 1: API Understanding** (2–3 days)

1. Study `AsyncDiLoCo` class: push/pull protocol, staleness tracking
2. Trace wire format: what bytes are sent, how model_id works
3. Document: 1-page architecture flow with staleness metadata

### **Phase 2: Find Integration Points** (2–3 days)

1. Locate torchtitan's FT manager (look for `semi_sync_method` in torchft fork)
2. Identify where `sync_steps` barrier is enforced
3. Determine if trainer can call `AsyncDiLoCo.push()` directly
4. Document: current FT flow + proposed async hooks

### **Phase 3: Standalone Async Training** (3–5 days)

1. Create `/panoengine/decentralized/async_trainer_standalone.py`
   - GPT-2 + C4 language modeling
   - Calls `AsyncDiLoCo.push()` after every N steps
   - No barriers, no torchtitan dependency
2. Start parameter server, run workers on threads
3. Verify: convergence, staleness tracking
4. Plot: arrival times vs. staleness

### **Phase 4: torchtitan Integration (Option 4b)** (1–2 weeks)

**Option 4b: New Async Entry Point** (cleaner, less risky)

1. Create `/panoengine/decentralized/async_trainer.py`
   - Full training loop (model, data, optimizer from torchtitan)
   - Uses `AsyncDiLoCo.push()/pull()`
   - Reads same config (heloco.yaml)
2. Adapt `run_heloco.py`:
   - Add flag: `--coordination-method sync|async` (default sync)
   - When async: spawn `-m panoengine.decentralized.async_trainer`
   - Skip `--fault_tolerance` flags for async
3. Validate: compare sync vs. async convergence

### **Phase 5: System Heterogeneity** (1–2 weeks)

1. Add `--island-straggler-factors [1.0, 0.8, 1.2]`
2. Insert synthetic delays in training loop
3. Measure staleness distribution, final loss
4. Plot: heterogeneity impact on convergence

---

## Effort & Timeline

| Phase | Task | Days | Cumulative |
|-------|------|------|-----------|
| 1 | Learn AsyncDiLoCo API | 2 | 2 |
| 2 | Find torchtitan hooks | 2 | 4 |
| 3 | Standalone async training | 4 | 8 |
| 4 | torchtitan integration (4b) | 10 | 18 |
| 5 | System heterogeneity | 7 | 25 |

**Total: ~3.5 weeks (5 working weeks)**

---

## Staleness Tracking (Technical Detail)

In `AsyncDiLoCoServer`, each worker has a `model_id`:

```python
class WorkerState:
    model_id: int  # When did this worker last pull?

def synchronize(self, worker):
    staleness = self.server.model_id - worker.model_id
    sync_weight = self._staleness_weight(staleness)  # Down-weight stale grads
```

**Window-sync:** staleness = 0 (all workers sync simultaneously)
**Async:** staleness = 1, 2, 5, 10+ (workers lag by multiple server steps)

---

## Push/Pull Wire Protocol

Worker side:
```python
pseudo_grad = compute_pseudo_gradient()
updated_weights = server.push(pseudo_grad, worker_id=island, model_id=my_model_id)
# Update local weights
```

Server side:
```python
def forward(self, pseudo_grad_bytes):
    pseudo_grad = deserialize(pseudo_grad_bytes)
    corrected = block_correct(pseudo_grad, momentum)    # HeLoCo
    outer_optimizer.step()                               # Apply step
    snapshot = _lookahead_snapshot()                     # Eq. 5
    return serialize(snapshot)
```

---

## Current vs. Async Training Loop

**Current (sync):**
```python
for step in range(num_steps):
    loss = forward_backward()
    # At sync_steps boundary: BLOCKS until all workers ready
    if step % sync_steps == sync_steps - 1:
        server.push_pull()  # ← BARRIER
```

**Async:**
```python
push_thread = None
for step in range(num_steps):
    # Wait for previous push to finish (non-blocking)
    if push_thread and step % async_interval == 0:
        pull_result = push_thread.join()
        update_weights(pull_result)
    
    loss = forward_backward()
    
    # Start next push in background
    if step % async_interval == 0:
        push_thread = Thread(target=push, args=(grad,))
        push_thread.start()  # ← NON-BLOCKING
```

---

## Validation Checklist

- [ ] Phase 1: AsyncDiLoCo API documented
- [ ] Phase 2: torchtitan hooks identified
- [ ] Phase 3: Standalone async converges on C4
- [ ] Phase 4: New async trainer works on 4×A100
- [ ] Comparison: sync vs. async
  - [ ] Final loss within 1% of sync
  - [ ] Staleness histogram shows range
  - [ ] Convergence speed measured
- [ ] Phase 5: Heterogeneity factors applied
- [ ] Dry-run: `--coordination-method async --dry-run`
- [ ] Real run: `--coordination-method async --steps 50 --islands 2`

---

## Key References

- AsyncDiLoCo: `panoengine/decentralized/async_diloco.py`
- HeLoCo Server: `panoengine/decentralized/heloco.py` (lines 159–250)
- Current tests: `panoengine/decentralized/tests/heloco_test.py`
- Wire protocol: `async_diloco.py` (lines 300–500, HTTP handlers)
- Staleness weighting: `heloco.py` (_staleness_weight method)
