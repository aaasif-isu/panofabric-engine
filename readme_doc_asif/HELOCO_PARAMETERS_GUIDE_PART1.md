# HeLoCo Parameters Guide - Part 1: Overview & Timing

## Overview

HeLoCo (Heterogeneity-aware Low-Communication training) implements **tensorwise block-level directional correction** on the server side. This guide explains **when** correction, rotation, and shrinking happen, and what parameters control them.

---

## When Does Correction Happen?

### **Timing: After Each Window (sync_steps)**

**HeLoCo's correction cycle:**

1. **During Window (steps 1 to sync_steps)**
   - Each island trains independently on local data for `sync_steps` iterations
   - Workers compute local gradients (no correction yet)

2. **At Window Boundary (end of sync_steps)**
   - Workers push their **pseudo-gradients** to the parameter server
   - The pseudo-gradient Δ is the difference between current and initial model

3. **On Parameter Server (CORRECTION HAPPENS HERE)**
   - Server applies **block correction** (Algorithm 2) to the incoming pseudo-gradient
   - Three possible outcomes for each tensor block:
     - ✅ **Pass through** (well-aligned)
     - 🔄 **Rotate** (weakly-aligned)
     - 📉 **Shrink** (anti-aligned)

4. **Server Updates Model**
   - Corrected pseudo-gradient is used in momentum-lookahead update
   - Server sends back the look-ahead position θ̄ to workers

5. **Next Window Starts**
   - Workers receive the updated model and repeat

---

## Configuration File: heloco.yaml

Here are the key HeLoCo parameters in `heloco.yaml`:

### **Window and Fragment Control**

```yaml
# ================================================================ DECENTRALIZED TRAINING
sync_steps: 20          # Window H: inner steps between pushes to parameter server
num_fragments: 1        # fragment-wise sync; must divide sync_steps (1 = whole model)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `sync_steps` | 20 | **How many training steps happen locally before syncing to server.** This is the "window" size H. Smaller = more frequent communication, larger = more stale gradients. |
| `num_fragments` | 1 | If > 1, the model is split into P fragments; each fragment syncs independently. 1 = whole model syncs at once. |

---

### **Server-Side Outer Optimizer**

```yaml
# Server-side outer optimizer (HeLoCo, DiLoCo, MLA)
outer_lr: 0.7           # outer learning rate (HeLoCoOptimizer / MLAOptimizer)
outer_momentum: 0.9     # momentum
rho: null               # HeLoCo arrival weight; null -> auto (1/sqrt(islands))
should_quantize: false  # int8 pseudo-gradient upload (reduces comm)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `outer_lr` | 0.7 | Learning rate η for the outer optimizer. Used in momentum-lookahead: θ̄ = θ − η·μ·m |
| `outer_momentum` | 0.9 | Momentum coefficient μ. Higher = more momentum. Range: [0, 1) |
| `rho` | auto | **Arrival weight ρ** applied after block correction. Default: 1/√K (K = num islands). |
| `should_quantize` | false | Quantize pseudo-gradients to int8 for reduced communication overhead. |

---

### **Async Coordination**

```yaml
coordination_method: sync    # or: async
async_interval: 1            # check staleness every N window pulses (async only)
max_wait_time: 0.0           # max seconds to wait for server response; 0.0 = no timeout
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `coordination_method` | sync | **sync**: Workers wait at window boundary (torchft barrier). **async**: Workers never block; server processes updates as they arrive. |
| `async_interval` | 1 | In async mode, check staleness every N pulses. |
| `max_wait_time` | 0.0 | Max seconds to wait for server. 0.0 = no timeout. |

