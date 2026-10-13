#!/usr/bin/env python
"""Generate a balanced multilingual validation cache."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from panoengine.decentralized.patch_dataset import patch_torchtitan_c4
patch_torchtitan_c4()

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--batches", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    import torch
    from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
    from panoengine.decentralized.decoupled_heloco.config import load_config
    from panoengine.decentralized.decoupled_heloco.gpu_recipe import build_recipe
    from panoengine.decentralized.decoupled_heloco.evaluation import freeze_batches, validation_cache

    folder = args.run_dir.resolve()
    config = load_config(folder / "experiment.yaml")
    options = SimpleNamespace(**json.loads((folder / "run-options.json").read_text()))

    recipe = build_recipe(options, REPO_ROOT, folder)
    tokenizer = recipe.tokenizer.build(tokenizer_path=recipe.hf_assets_path)
    vocab_size = recipe.model_spec.model.vocab_size

    all_batches = []
    num_langs = len(args.languages)
    base_batches = args.batches // num_langs
    remainder = args.batches % num_langs

    lang_counts = {}

    for i, lang in enumerate(args.languages):
        lang_batches = base_batches + (1 if i < remainder else 0)
        lang_counts[lang] = lang_batches
        if lang_batches == 0:
            continue
        os.environ["ISLAND_LANGUAGE"] = lang
        loader_config = HuggingFaceTextDataLoader.Config(dataset="c4_validation", infinite=False, num_workers=0)
        loader = loader_config.build(dp_world_size=1, dp_rank=0, tokenizer=tokenizer,
                                     seq_len=options.seq_len, local_batch_size=options.batch)
        frozen = freeze_batches(loader, lang_batches, seq_len=options.seq_len, batch_size=options.batch, vocab_size=vocab_size)
        all_batches.extend(frozen)

    from panoengine.decentralized.decoupled_heloco.global_validation import tokenizer_fingerprint
    tokenizer_sha = tokenizer_fingerprint(tokenizer, Path(options.hf_assets))
    cache_metadata = {"dataset": "c4_validation", "split": "validation", "seq_len": options.seq_len,
                      "batch_size": options.batch, "batch_count": args.batches, "tokenizer_sha256": tokenizer_sha}
    
    cache = validation_cache(all_batches, cache_metadata)
    torch.save(cache, args.output)
    print(json.dumps(lang_counts))

if __name__ == "__main__":
    main()
