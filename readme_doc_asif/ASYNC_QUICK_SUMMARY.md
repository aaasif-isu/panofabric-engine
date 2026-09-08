# Quick Summary: Making HeLoCo Asynchronous

## The Question
Can I make `run_heloco.py` fully asynchronous (no barriers, workers train independently)?

## The Answer
**Yes, absolutely.** Good news: 90% of the code already exists.

---

## 3 Paths

### ✅ **Path A: New Async Trainer Entry Point** (RECOMMENDED)

Bypass torchtitan's window-sync. Create new trainer with async loop.

**Code:** Create `/panoengine/decentralized/async_trainer.py`
**Effort:** 1–2 weeks
**Risk:** Low
**Why:** Clean, full control, no upstream coupling

### ⚠️ **Path B: Patch torchtitan**

Add `semi_sync_method=async_heloco` to torchft's FT manager.

**Effort:** 3–4 weeks
**Risk:** High (upstream coupling)
**Why not:** Maintenance burden

### 🟡 **Path C: Hybrid**

Keep barrier, track worker speeds, weight slower workers less.

**Effort:** 3–5 days
**Why:** Minimal changes, but not fully async

---

## What Async Looks Like

### Current (Window-Sync)
```
Trainer-0: Step 1→10 [push]
Trainer-1: Step 1→3  [WAIT]
Trainer-2: Step 1→7  [WAIT]
           
           After 10s, all push together (BARRIER)
```

### Async (After Implementation)
```
Trainer-0: Step 1→10 [push at t=1.2s, staleness=0]
Trainer-1: Step 1→3  [push at t=2.5s, staleness=1] ← No wait!
Trainer-2: Step 1→7  [push at t=1.8s, staleness=0]

Server accepts all immediately. No barrier.
```

---

## Current Code vs. Async Code

### Current (torchtitan's FT manager)
```python
def training_loop():
    for step in range(num_steps):
        loss, grad = forward_backward()
        if step % sync_steps == sync_steps - 1:
            server.push_pull()  # ← BARRIER: blocks all workers
```

### Async (new async_trainer.py)
```python
def training_loop():
    push_thread = None
    for step in range(num_steps):
        # Collect result from PREVIOUS push (non-blocking)
        if push_thread and step > 0:
            updated = push_thread.join()
            update_local_weights(updated)
        
        loss, grad = forward_backward()
        
        # START next push in background
        if step > 0 and step % async_interval == 0:
            push_thread = Thread(
                target=server.push,
                args=(grad, worker_id, model_id)
            )
            push_thread.start()  # ← Continues training while pushing
```

**Key:** No sync_steps barrier. Every worker pushes independently.

---

## 5-Week Roadmap (Path A)

| Week | Days | Phase | Goal |
|------|------|-------|------|
| 1 | 5 | Study | Learn AsyncDiLoCo API + torchtitan hooks |
| 2 | 5 | Standalone | Build & test async trainer (no torchtitan) |
| 3–4 | 10 | Integration | Full integration with heloco.yaml |
| 5 | 5 | Polish | Heterogeneity, docs, validation |

---

## Decision: Which Path?

**Choose Path A if:** You want true async, don't mind new entry point ✅
**Choose Path B if:** You must use `torchrun -m torchtitan.train` ❌
**Choose Path C if:** You want quick proof-of-concept, OK with hybrid approach 🟡

---

## What's Already There

✅ AsyncDiLoCoServer (panoengine/decentralized/async_diloco.py)
✅ HeLoCoServer (panoengine/decentralized/heloco.py)
✅ Push/pull wire protocol (HTTP, serialization)
✅ Staleness tracking & weighting
✅ Parameter server (parameter_server.py)

❌ Integration into torchtitan training loop
❌ run_heloco.py async flag
❌ System heterogeneity simulation

---

## Bottom Line

**Yes, you can make HeLoCo async. Path A is recommended. Start with Phase 1 (2 days) to learn the API. Then decide if you want to proceed.**

See `ASYNC_CONVERSION_ROADMAP.md` for full details.

