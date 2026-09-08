# Non-IID Data Distribution Guide

## Overview

The launcher now supports three data distribution modes:
- **IID** (default): All islands get the same language (C4 English) - identical and independent distribution
- **Non-IID**: Each island gets a different language from C4 - heterogeneous distribution
- **Both**: Run both IID and non-IID sequentially for comparison

This allows you to test and compare method performance under both IID and non-IID data scenarios.

---

## Configuration

### Via YAML (`heloco.yaml`)

```yaml
data_distribution: iid              # or: non_iid, both
languages: [english, french]        # per-island languages (only for non_iid/both)
```

### Via Command Line

```bash
# IID mode (default)
heloco/bin/python run_heloco.py --data-distribution iid

# Non-IID mode with 2 islands
heloco/bin/python run_heloco.py --data-distribution non_iid --islands 2 --languages english french

# Non-IID mode with 3 islands
heloco/bin/python run_heloco.py --data-distribution non_iid --islands 3 --languages english french german

# Both modes sequentially
heloco/bin/python run_heloco.py --data-distribution both --languages english french
```

---

## Available Languages (C4 Dataset)

C4 supports the following languages:

```
english, french, german, spanish, italian, portuguese,
romanian, dutch, greek, czech, polish, hungarian,
croatian, swedish, finnish, danish, bulgarian, lithuanian,
slovene, slovak, irish, maltese
```

---

## Usage Examples

### Example 1: Compare IID vs Non-IID

YAML Configuration:
```yaml
data_distribution: both
languages: [english, french]
islands: 2
steps: 100
methods: [heloco, diloco]
```

Command:
```bash
heloco/bin/python run_heloco.py
```

**Output:**
- First run: IID mode (both islands see English from C4)
- Second run: Non-IID mode (island 0 → English, island 1 → French)
- Comparison plot shows:
  - `heloco_iid` vs `diloco_iid`
  - `heloco_non_iid` vs `diloco_non_iid`

### Example 2: Non-IID with Multiple Languages

```bash
heloco/bin/python run_heloco.py \
  --data-distribution non_iid \
  --islands 4 \
  --gpus-per-island 2 \
  --languages english french german spanish \
  --steps 100
```

**Output:**
- Island 0: C4 English
- Island 1: C4 French
- Island 2: C4 German
- Island 3: C4 Spanish
- All islands see the same number of tokens for fair comparison

### Example 3: Only Non-IID (Skip IID)

```bash
heloco/bin/python run_heloco.py \
  --data-distribution non_iid \
  --languages english french
```

---

## How It Works

### Environment Variable Mechanism

For each island in non-IID mode, the launcher sets the `ISLAND_LANGUAGE` environment variable:

```
Island 0: ISLAND_LANGUAGE=english
Island 1: ISLAND_LANGUAGE=french
Island 2: ISLAND_LANGUAGE=german
...
```

**For trainer developers:** The trainer should read `ISLAND_LANGUAGE` from the environment and load the corresponding C4 language instead of defaulting to English:

```python
# Example in trainer code (pseudo-code)
language = os.environ.get("ISLAND_LANGUAGE", "english")
dataset = load_dataset(f"allenai/c4", name=language, split="train", streaming=True)
```

### Token Balancing

Each island processes data from its assigned language such that all islands see approximately the same number of tokens for a fair comparison.

---

## Output Structure

### IID Mode
```
outputs/heloco_run/
├── method-heloco/
│   └── metrics/
├── method-diloco/
│   └── metrics/
└── comparison_loss.png (heloco vs diloco, both IID)
```

### Non-IID Mode
```
outputs/heloco_run/
├── method-heloco_non_iid/
│   └── metrics/
├── method-diloco_non_iid/
│   └── metrics/
└── comparison_loss.png (heloco vs diloco, both non-IID)
```

### Both Modes
```
outputs/heloco_run/
├── method-heloco_iid/
│   └── metrics/
├── method-diloco_iid/
│   └── metrics/
├── method-heloco_non_iid/
│   └── metrics/
├── method-diloco_non_iid/
│   └── metrics/
└── comparison_loss.png (all 4 combinations)
```

---

## Validation & Constraints

1. **Language count must match island count:**
   ```
   ✅ --islands 2 --languages english french        (OK)
   ❌ --islands 3 --languages english french        (ERROR)
   ```

2. **Languages only used in non-IID/both modes:**
   ```
   ✅ --data-distribution iid (languages ignored)
   ```

3. **Non-IID/both modes require languages:**
   ```bash
   # This will error:
   heloco/bin/python run_heloco.py --data-distribution non_iid
   
   # Must provide languages:
   heloco/bin/python run_heloco.py --data-distribution non_iid --languages english french
   ```

---

## Dry-Run Mode

Check your configuration without actually running:

```bash
heloco/bin/python run_heloco.py --dry-run --data-distribution non_iid --languages english french
```

Output shows:
- Configuration loaded
- Data distribution mode
- Languages per island
- Commands that would be executed

---

## Notes

- **IID vs Non-IID trade-off:** IID provides identical data distribution (controlled experiment), while non-IID tests method robustness to heterogeneous data
- **Fair comparison:** Token counts are balanced across islands regardless of language
- **Sequential execution:** "Both" mode runs IID first, then non-IID sequentially (not in parallel)
- **No code changes needed:** Trainer only needs to read `ISLAND_LANGUAGE` env var if set; defaults to English if not set

---

## Troubleshooting

**Error: "Number of languages must match number of islands"**
- Solution: Provide exactly one language per island

**Error: "languages required for non_iid mode"**
- Solution: Specify `--languages` when using `--data-distribution non_iid`

**Non-IID data not loading:**
- Ensure trainer code reads `ISLAND_LANGUAGE` environment variable
- Check that C4 has the specified language available
- Verify network connectivity to HuggingFace Hub
