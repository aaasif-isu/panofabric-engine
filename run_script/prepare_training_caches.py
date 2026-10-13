#!/usr/bin/env python
"""Generate bounded deterministic training caches per learner."""

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from panoengine.decentralized.patch_dataset import patch_torchtitan_c4
patch_torchtitan_c4()

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--batches", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    import torch
    from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
    from panoengine.decentralized.decoupled_heloco.config import load_config
    from panoengine.decentralized.decoupled_heloco.gpu_recipe import build_recipe
    
    config = load_config(args.config_file)
    from run_script.run_decoupled_heloco import _load_legacy_launcher, _parse_legacy_options
    legacy = _load_legacy_launcher()
    import tempfile
    import yaml
    with tempfile.TemporaryDirectory() as directory:
        effective = Path(directory) / "legacy.yaml"
        effective.write_text(yaml.safe_dump(config.legacy_options(), sort_keys=False), encoding="utf-8")
        options = _parse_legacy_options(legacy, effective)

    # We need a dummy folder for build_recipe logging path
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recipe = build_recipe(options, REPO_ROOT, args.output_dir)
    tokenizer = recipe.tokenizer.build(tokenizer_path=recipe.hf_assets_path)
    
    for learner_id in range(options.islands):
        dist_mode = getattr(options, "data_distribution", "iid")
        languages = getattr(options, "languages", [])
        is_non_iid = dist_mode in ("non_iid", "both") and languages
        if is_non_iid:
            lang = languages[learner_id % len(languages)]
            os.environ["ISLAND_LANGUAGE"] = lang
            dp_world_size = 1
            dp_rank = 0
        else:
            os.environ["ISLAND_LANGUAGE"] = "en"
            dp_world_size = options.islands
            dp_rank = learner_id

        loader_config = HuggingFaceTextDataLoader.Config(dataset="c4", infinite=False, num_workers=0)
        loader = loader_config.build(dp_world_size=dp_world_size, dp_rank=dp_rank, tokenizer=tokenizer,
                                     seq_len=options.seq_len, local_batch_size=options.batch)
        iterator = iter(loader)
        batches = []
        for i in range(args.batches):
            try:
                inputs, labels = next(iterator)
                batches.append(({key: value.clone() for key, value in inputs.items()}, labels.clone()))
            except StopIteration:
                break
        torch.save(batches, args.output_dir / f"train_cache_learner_{learner_id}.pt")
        print(f"Saved {len(batches)} batches for learner {learner_id}")

if __name__ == "__main__":
    main()
