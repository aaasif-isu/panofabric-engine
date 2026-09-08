# Async Architecture Diagrams

## 1. Current vs. Target

### Current (Window-Synchronous)
```
All workers → [complete window] → [BARRIER] → [push simultaneously]
         ↓
Slow workers stall fast workers at barrier.
```

### Target (Fully Asynchronous)
```
Island-0 (fast):  → [complete window] → [push immediately at 1.2s] → [continue]
Island-1 (slow):  → [complete window] → [push immediately at 2.5s] → [continue]
Island-2 (med):   → [complete window] → [push immediately at 1.8s] → [continue]

         ↓
No barrier. Server accepts staggered arrivals. No stalls.
```

---

## 2. Training Loop Code

### Window-Sync (Current)
```python
for step in range(num_steps):
    loss = forward_backward()
    
    if step % sync_steps == sync_steps - 1:
        server.push_pull()  # ← BARRIER: all workers block here
```

### Async (New)
```python
push_thread = None
for step in range(num_steps):
    # Collect PREVIOUS push result (non-blocking)
    if push_thread and step > 0:
        updated = push_thread.join()
        update_local_weights(updated)
    
    loss = forward_backward()
    
    # START next push in background (non-blocking)
    if step > 0 and step % async_interval == 0:
        push_thread = Thread(target=server.push, args=(grad,))
        push_thread.start()  # Returns immediately
```

---

## 3. Staleness Tracking

```
Server    [model rev 0] → [rev 1] → [rev 2] → [rev 3] → [rev 4]
Fast work:    [pull 0]  [train] [push at rev 4] (staleness = 4-0 = 4)
Slow work:        [pull 0]      [train longer] [push at rev 2] (τ = 2-0 = 2)

HeLoCo: weight = 1/sqrt(1 + staleness)
  Fast: weight = 1/√5 ≈ 0.45 (down-weighted more)
  Slow: weight = 1/√3 ≈ 0.58 (down-weighted less)
```

---

## 4. Configuration Changes

**Before:**
```yaml
sync_steps: 10              # Barrier every 10 steps
```

**After:**
```yaml
coordination_method: async  # 'sync' or 'async'
async_interval: 10         # Push every 10 steps (NO barrier)
```

**Commands:**
```bash
# Sync (existing)
python run_heloco.py --steps 100

# Async (new)
python run_heloco.py --coordination-method async --steps 100
```
