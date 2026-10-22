import sys
import os
from pathlib import Path

print("Starting test...")
from panoengine.decentralized.decoupled_heloco.gpu_recipe import build_recipe
from panoengine.decentralized.decoupled_heloco.config import load_config
from types import SimpleNamespace

print("Loaded imports")
config = load_config(Path("run_script/decoupled_heloco.yaml"))
print("Config loaded")

options = SimpleNamespace(
    module="models.qwen3",
    config="qwen3_0_6b",
    batch=1,
    seq_len=512,
    steps=10,
    hf_assets="assets/hf/Qwen3-0.6B",
    dataset="c4",
    seed=42,
    gpus_per_island=1,
    data_distribution="iid",
    islands=4,
)

recipe = build_recipe(options, Path("."), Path("/tmp/dump"))
print("Recipe built:", recipe.model_spec.name, recipe.model_spec.flavor)
