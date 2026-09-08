# Distributed Training Topology Explained

## Your Configuration

```yaml
islands: 2              # 2 independent training replicas
gpus_per_island: 2      # Each island uses 2 GPUs
gpus: [0, 1, 2, 3]      # Total 4 GPUs available
```

---

## Hardware Setup

```
Your Machine (Single Node)
│
├─ GPU 0
├─ GPU 1
├─ GPU 2
└─ GPU 3

Total: 4 GPUs
```

---

## How They're Organized

### **Island 1 (Worker 1)**
```
┌─────────────────────────────────┐
│     ISLAND 1 (Process 1)        │
│                                 │
│  ┌──────────────────────────┐   │
│  │ GPU 0 (Rank 0)           │   │
│  │ Model Shard 1            │   │
│  └──────────────────────────┘   │
│                                 │
│  ┌──────────────────────────┐   │
│  │ GPU 1 (Rank 1)           │   │
│  │ Model Shard 2            │   │
│  └──────────────────────────┘   │
│                                 │
│  [Connected by NVLink/PCIe]     │
│  Uses FSDP (Fully Sharded        │
│   Data Parallel) to coordinate  │
└─────────────────────────────────┘
```

### **Island 2 (Worker 2)**
```
┌─────────────────────────────────┐
│     ISLAND 2 (Process 2)        │
│                                 │
│  ┌──────────────────────────┐   │
│  │ GPU 2 (Rank 0)           │   │
│  │ Model Shard 1            │   │
│  └──────────────────────────┘   │
│                                 │
│  ┌──────────────────────────┐   │
│  │ GPU 3 (Rank 1)           │   │
│  │ Model Shard 2            │   │
│  └──────────────────────────┘   │
│                                 │
│  [Connected by NVLink/PCIe]     │
│  Uses FSDP (Fully Sharded        │
│   Data Parallel) to coordinate  │
└─────────────────────────────────┘
```

### **Communication Between Islands**
```
┌─────────────────┐              ┌─────────────────┐
│   ISLAND 1      │              │   ISLAND 2      │
│                 │              │                 │
│  GPU0   GPU1    │              │  GPU2   GPU3    │
└────┬────┬───────┘              └────┬────┬───────┘
     │    │                           │    │
     └────┼──────────────────────────┘    │
          │ (Network - Ethernet/TCP)      │
          └──────────────────────────────┘
          
Data synced every sync_steps (20 steps)
```

---

## Three Levels of Parallelism

```
Level 1: INSIDE Island (GPU 0 & GPU 1)
  ↓
  FSDP: Model is split across 2 GPUs
  - GPU 0 has Shard 1 (Transformer Layers 1-8)
  - GPU 1 has Shard 2 (Transformer Layers 9-16)
  - They sync within each forward pass (fast, local)

Level 2: INSIDE Island (Training)
  ↓
  Each GPU processes part of the batch
  - GPU 0: Batch items 1-4
  - GPU 1: Batch items 5-8
  Combined: 8 total (batch_size: 8 in config)

Level 3: BETWEEN Islands (HeLoCo/DiLoCo)
  ↓
  Island 1 and Island 2 train independently
  Then sync gradients every sync_steps=20
  - Island 1 does steps 1-20
  - Island 2 does steps 1-20 in parallel
  - Then both push to parameter server
  - Server computes new weights
  - Both download updated model
  - Repeat
```

---

## Training Flow - Step by Step

### **Initialization Phase**

```
Parameter Server (Central)
  │
  ├─→ Broadcast initial model to Island 1 (GPU 0, GPU 1)
  └─→ Broadcast initial model to Island 2 (GPU 2, GPU 3)
```

### **Training Phase (Window 1: Steps 1-20)**

```
Timeline:

Island 1 (GPU 0, GPU 1)              Island 2 (GPU 2, GPU 3)
├─ Step 1: Train batch               ├─ Step 1: Train batch
│  ├─ GPU 0 computes shard 1         │  ├─ GPU 2 computes shard 1
│  └─ GPU 1 computes shard 2         │  └─ GPU 3 computes shard 2
│  └─ [FSDP sync within GPU pair]    │  └─ [FSDP sync within GPU pair]
│                                    │
├─ Step 2: Train batch               ├─ Step 2: Train batch
│  [FSDP sync]                       │  [FSDP sync]
│                                    │
├─ ...                               ├─ ...
│                                    │
├─ Step 20: Train batch              ├─ Step 20: Train batch
│  [FSDP sync]                       │  [FSDP sync]
│                                    │
└─ Push gradients to server          └─ Push gradients to server
   (HeLoCo/DiLoCo communication)        (HeLoCo/DiLoCo communication)
```

### **Synchronization Phase (After Step 20)**

```
Island 1 Gradients    Island 2 Gradients
         ↓                     ↓
         └─────────┬───────────┘
                   ↓
         Parameter Server
         (Parameter averaging or
          other aggregation)
                   ↓
         New Model Weights
         ↓                     ↓
    Island 1              Island 2
   Download         &     Download
   Updated Model         Updated Model
```

### **Training Phase (Window 2: Steps 21-40)**

```
Island 1 starts with updated model    Island 2 starts with updated model
Continue training steps 21-40         Continue training steps 21-40
(Same as Window 1)                    (Same as Window 1)
```

---

## Data Flow Example

Let's say your batch is 8 samples:

```
Batch 8 (batch_size from config)
├─ GPU 0 (Island 1): Processes samples [0-3]
│  └─ Model Shard 1 (Layers 1-8)
│
├─ GPU 1 (Island 1): Processes samples [4-7]
│  └─ Model Shard 2 (Layers 9-16)
│
├─ GPU 2 (Island 2): Processes samples [0-3] (DIFFERENT data!)
│  └─ Model Shard 1
│
├─ GPU 3 (Island 2): Processes samples [4-7] (DIFFERENT data!)
│  └─ Model Shard 2
│
└─ Every sync_steps: Aggregate & update all models
```

---

## Key Concepts

### **FSDP (Fully Sharded Data Parallel)**
- Model sharded across GPUs within Island
- Each GPU holds part of parameters
- Fast communication (NVLink)
- Local synchronization

### **HeLoCo / DiLoCo (Decentralized Learning)**
- Independent training across Islands
- Sync only every sync_steps
- Reduced communication overhead
- Parameter server aggregates

### **Your Synchronization Points**
```
With steps=100, sync_steps=20:

Island 1: [1-20] SYNC [21-40] SYNC [41-60] SYNC [61-80] SYNC [81-100]
Island 2: [1-20] SYNC [21-40] SYNC [41-60] SYNC [61-80] SYNC [81-100]

Evaluation log: steps 20, 40, 60, 80, 100 (5 checkpoints)
```

---

## Summary

| Item | Value | Explanation |
|------|-------|-------------|
| **Islands** | 2 | Two independent workers |
| **GPUs/Island** | 2 | FSDP model sharding |
| **Total GPUs** | 4 | GPU 0, 1, 2, 3 |
| **Batch Size** | 8 | 4 per GPU × 2 GPUs/island |
| **Steps** | 100 | Total training iterations |
| **Sync Steps** | 20 | Checkpoint every 20 steps |
| **Eval Records** | 5 per method | At steps 20, 40, 60, 80, 100 |

**Does this explain it clearly?** 🎯
