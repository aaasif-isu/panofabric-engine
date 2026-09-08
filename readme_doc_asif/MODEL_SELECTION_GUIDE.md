# Model Selection Guide - Simple Explanation

## The 3 YAML Parameters for Models

```yaml
module: llama3_small          # ← Model FAMILY
config: llama3_15m            # ← Specific SIZE within family
hf_assets: ./assets/tokenizer/debug   # ← Tokenizer/weights to use
```

---

## What Each Parameter Means

### **1. `module`: Model Family**

This is the **MODEL FAMILY** name. Think of it as the model category.

**Available:**
- `llama3_small` - Small Llama model (15M params)
- `llama3` - Larger Llama models (8B, 70B, 405B)
- `qwen3` - Qwen model family
- `gpt_oss` - GPT models
- etc.

**Rule:** The module must exist in `./models/` directory!

```bash
ls ./models/
# Shows: llama3_small/, llama3/, qwen3/, gpt_oss/, etc.
```

---

### **2. `config`: Specific Model Size**

This is the **SIZE/VARIANT** within the family. It must be a function name in that module's `config_registry.py`.

**Examples for `llama3_small`:**
- `llama3_15m` - 15 million params ✅ SMALL

**Examples for `llama3`:**
- `llama3_8b` - 8 billion params (large)
- `llama3_70b` - 70 billion params (very large)
- `llama3_405b` - 405 billion params (huge, won't fit)

**Rule:** The config must exist in `./models/{module}/config_registry.py`!

```bash
# Check available configs:
grep "^def " ./models/llama3_small/config_registry.py
# Output: def llama3_15m(), def llama3_30m(), etc.

grep "^def " ./models/llama3/config_registry.py
# Output: def llama3_8b(), def llama3_70b(), def llama3_405b(), etc.
```

---

### **3. `hf_assets`: Tokenizer/Weights Path**

This is the **TOKENIZER VOCABULARY** and optional pre-trained weights.

**For `llama3_small` (15M):**
```yaml
hf_assets: ./assets/tokenizer/debug   # Debug tokenizer (2048 vocab)
```

**For `llama3` (8B, 70B, 405B):**
```yaml
hf_assets: ./assets/hf/meta-llama     # HuggingFace Llama tokenizer
```

**For `qwen3` (600M, 1.7B):**
```yaml
hf_assets: ./assets/hf/Qwen3-0.6B     # Qwen tokenizer
```

**Rule:** The vocab size in config must match the tokenizer!

```bash
# Check what's available:
ls ./assets/tokenizer/
# Output: debug/

ls ./assets/hf/
# Output: Qwen3-0.6B/, meta-llama/, etc.
```

---

## Common Setups (Copy-Paste Ready)

### **Setup 1: Fastest (15M params) - RECOMMENDED FOR TESTING**

```yaml
module: llama3_small
config: llama3_15m
hf_assets: ./assets/tokenizer/debug
```

**Pros:** Fast, fits on 4 GPUs easily  
**Cons:** Very small model  
**Use for:** Quick testing ✅

---

### **Setup 2: Medium (8B params) - If you want bigger**

```yaml
module: llama3
config: llama3_8b
hf_assets: ./assets/hf/meta-llama
```

**Pros:** Real Llama model  
**Cons:** Needs more VRAM, slower  
**Note:** May OOM on 4 GPUs with batch=8!

---

### **Setup 3: Small Qwen (600M params)**

```yaml
module: qwen3
config: qwen3_0_6b
hf_assets: ./assets/hf/Qwen3-0.6B
```

**Pros:** Small but newer model  
**Cons:** Compatibility issues (as you saw)  
**Status:** ⚠️ May have torchtitan version conflicts

---

## How to Find Available Options

### **Step 1: List available modules**
```bash
ls ./models/
# Shows: llama3_small/, llama3/, qwen3/, etc.
```

### **Step 2: List available configs in a module**
```bash
grep "^def " ./models/llama3_small/config_registry.py
# Shows: def llama3_15m(), def llama3_30m(), etc.
```

### **Step 3: List available tokenizers**
```bash
ls ./assets/tokenizer/
ls ./assets/hf/
# Shows: debug/, Qwen3-0.6B/, meta-llama/, etc.
```

### **Step 4: Check tokenizer vocab size in config**
```bash
grep "vocab_size\|2048\|128" ./models/llama3_small/config_registry.py
# Should match the tokenizer vocab!
```

---

## Troubleshooting

### **Error: "Cannot import module 'llama3_small'"**
❌ Module name doesn't exist  
✅ Check: `ls ./models/`

### **Error: "Config function 'llama3_base' not found"**
❌ Config name is wrong for this module  
✅ Check: `grep "^def " ./models/llama3/config_registry.py`

### **Error: "vocab_size mismatch"**
❌ Tokenizer vocab doesn't match config  
✅ Check: Config vocab size = Tokenizer vocab size

### **Error: "Cannot import module 'qwen3_0_6b'"**
❌ Wrong! The module is just `qwen3`, not `qwen3_0_6b`  
✅ Use: `module: qwen3` + `config: qwen3_0_6b`

---

## Quick Reference Table

| Want | module | config | hf_assets | Size | Speed |
|------|--------|--------|-----------|------|-------|
| **Fastest** | llama3_small | llama3_15m | ./assets/tokenizer/debug | 15M | ⚡⚡⚡ |
| **Medium** | llama3 | llama3_8b | ./assets/hf/meta-llama | 8B | ⚡ |
| **Small Qwen** | qwen3 | qwen3_0_6b | ./assets/hf/Qwen3-0.6B | 600M | ⚡⚡ |
| **Very Large** | llama3 | llama3_70b | ./assets/hf/meta-llama | 70B | 🐌 |

---

## YAML Configuration Template

```yaml
# ================================================================ MODEL & RECIPE
# Step 1: Choose module (model family)
#   Options: llama3_small, llama3, qwen3, gpt_oss, ...
module: llama3_small

# Step 2: Choose config (size within family)
#   Run: grep "^def " ./models/{module}/config_registry.py
config: llama3_15m

# Step 3: Choose assets (tokenizer)
#   Options: ./assets/tokenizer/debug, ./assets/hf/Qwen3-0.6B, ./assets/hf/meta-llama
hf_assets: ./assets/tokenizer/debug
```

---

## Your Current Setup

✅ **FIXED!** Now uses:
```yaml
module: llama3_small          # Small model family
config: llama3_15m            # 15M params size
hf_assets: ./assets/tokenizer/debug   # Debug tokenizer
```

**This will work!** Just run:
```bash
python run_heloco.py
```

🚀 **That's it!**
