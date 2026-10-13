#!/usr/bin/env python
"""Evaluate local model checkpoint on held-out batches."""

import argparse
import os
import sys
from pathlib import Path
import pickle
import json
from types import SimpleNamespace
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from panoengine.decentralized.patch_dataset import patch_torchtitan_c4
patch_torchtitan_c4()

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--learner-id", type=int, required=True)
    parser.add_argument("--validation-cache", type=Path, default=None)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    os.environ["ISLAND_LANGUAGE"] = args.language

    from torchtitan.distributed.utils import set_spmd_backend
    from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
    from panoengine.decentralized.decoupled_heloco.config import load_config
    from panoengine.decentralized.decoupled_heloco.gpu_recipe import build_initial_model, build_recipe, validate_tokenizer
    from panoengine.decentralized.decoupled_heloco.evaluation import evaluate_model, freeze_batches

    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision("highest")
    set_spmd_backend("partial_dtensor")

    folder = args.run_dir.resolve()
    config = load_config(folder / "experiment.yaml")
    options = SimpleNamespace(**json.loads((folder / "run-options.json").read_text()))
    
    recipe = build_recipe(options, REPO_ROOT, folder)
    tokenizer = recipe.tokenizer.build(tokenizer_path=recipe.hf_assets_path)
    vocab_size = recipe.model_spec.model.vocab_size
    validate_tokenizer(tokenizer, vocab_size)

    model = build_initial_model(recipe, options.seed)
    
    # Load local model
    local_checkpoint = folder / f"learner_{args.learner_id}" / "local_model.pt"
    if not local_checkpoint.is_file():
        raise RuntimeError(f"Local checkpoint not found: {local_checkpoint}")
    
    payload = torch.load(local_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload)
    model.to(args.device)

    # Build loader
    if getattr(args, "validation_cache", None) is not None and args.validation_cache.is_file():
        cache = torch.load(args.validation_cache, map_location="cpu", weights_only=True)
        from panoengine.decentralized.decoupled_heloco.evaluation import load_validation_cache
        frozen = load_validation_cache(cache, cache["metadata"], vocab_size=vocab_size)
    else:
        loader_config = HuggingFaceTextDataLoader.Config(dataset="c4_validation", infinite=False, num_workers=0)
        loader = loader_config.build(dp_world_size=1, dp_rank=0, tokenizer=tokenizer,
                                     seq_len=options.seq_len, local_batch_size=options.batch)
        frozen = freeze_batches(loader, args.batches, seq_len=options.seq_len, batch_size=options.batch, vocab_size=vocab_size)
    
    metrics = evaluate_model(model, frozen, device=torch.device(args.device), vocab_size=vocab_size)
    loss = metrics["loss"]
    ppl = metrics["perplexity"]
    acc = metrics["next_token_accuracy"]
    
    print(json.dumps({"loss": loss, "perplexity": ppl, "next_token_accuracy": acc}))

if __name__ == "__main__":
    main()