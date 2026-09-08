# Quick Topology Guide - 2 Islands × 2 GPUs

## 30-Second Summary

```
YOUR SETUP:
┌────────────────────────────────────────────────┐
│        1 Machine with 4 GPUs Total             │
├─────────────────────┬──────────────────────────┤
│  ISLAND 1           │  ISLAND 2                │
│  (Worker 1)         │  (Worker 2)              │
│  GPU 0              │  GPU 2                   │
│  GPU 1              │  GPU 3                   │
├─────────────────────┼──────────────────────────┤
│  FSDP: Model        │  FSDP: Model             │
│  sharding           │  sharding                │
│  (2 GPUs each)      │  (2 GPUs each)           │
└─────────────────────┴──────────────────────────┘
          ↑                       ↑
          └───────┬───────────────┘
                  ↓
        Parameter Server
        (Sync every 20 steps)
```

---

## What Happens During Training

### **STEP 1: Initialize**
```
Model broadcast from server to both islands
```

### **STEP 2: Local Training (Steps 1-20)**
```
Island 1                          Island 2
├─ GPU0 + GPU1 train              ├─ GPU2 + GPU3 train
│ on English data                 │ on French data
│ (FSDP sync: fast)               │ (FSDP sync: fast)
└─ Compute gradients              └─ Compute gradients

⚡ FAST (local NVLink connection)
✓ Happens in parallel
✓ No network communication
```

### **STEP 3: Synchronization (After Step 20)**
```
Island 1           Island 2
Gradients ──┐      ┌── Gradients
            ↓      ↓
     Parameter Server
     (Averaging/Update)
            ↓      ↓
Weights ◄──┘      └── Weights

🔄 NETWORK COMMUNICATION (slower)
✓ Happens once per 20 steps
✓ Reduces total communication
```

### **STEP 4: Repeat (Steps 21-40)**
```
Same as Step 2 with updated weights
```

---

## GPU Assignment

```
GPU 0 ──┐
        ├─→ Island 1 (Process 1)
GPU 1 ──┘

GPU 2 ──┐
        ├─→ Island 2 (Process 2)
GPU 3 ──┘
```

---

## Data Processing (Batch Size = 8)

```
Each Step:

Island 1:
  Batch [0-3] → GPU 0
  Batch [4-7] → GPU 1
  
Island 2:
  Batch [0-3] → GPU 2 (different samples!)
  Batch [4-7] → GPU 3 (different samples!)

Note: Islands see DIFFERENT data (non_iid mode)
```

---

## Communication Overhead

```
Without Decentralized Learning (centralized):
  Every step: Send gradients to server
  100 steps = 100 communications ❌

With HeLoCo/DiLoCo (decentralized):
  Every 20 steps: Send gradients to server
  100 steps = 5 communications ✓

Result: 95% less communication! 🚀
```

---

## Evaluation Log Records

```
steps: 100
sync_steps: 20

Records per method per island:
  Step 20 ✓
  Step 40 ✓
  Step 60 ✓
  Step 80 ✓
  Step 100 ✓
  
Total: 5 records per method per island
```

With 2 islands + 3 methods (diloco, heloco, mla):
- Expected: 2 × 3 × 5 = 30 total records ✓

---

## Key Takeaways

1. **Islands = Independent Workers**
   - Train separately for 20 steps
   - Only sync at checkpoints

2. **GPUs/Island = Model Sharding (FSDP)**
   - GPU0 + GPU1 share one model
   - GPU2 + GPU3 share another model
   - Fast local sync, slow global sync

3. **Sync Steps = Checkpoints**
   - Every 20 steps: coordinate
   - Evaluation log records these points

4. **Non-IID Data = Different Languages**
   - Island 1: English
   - Island 2: French
   - Realistic federated learning scenario

---

## One More Visual

```
Timeline (simplified):

Island 1:  ┌─ Local ─┐ [Sync] ┌─ Local ─┐ [Sync] ...
GPU0,1     │ Train   │        │ Train   │
           │ (1-20)  │        │ (21-40) │
           └─────────┘        └─────────┘
                                          
Island 2:  ┌─ Local ─┐ [Sync] ┌─ Local ─┐ [Sync] ...
GPU2,3     │ Train   │        │ Train   │
           │ (1-20)  │        │ (21-40) │
           └─────────┘        └─────────┘

Both run in PARALLEL for efficiency!
```

**That's it! You now understand your distributed training setup!** 🎯
