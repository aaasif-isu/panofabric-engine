"""Explicit dense, from-scratch 15M recipe for the first GPU integration."""

from dataclasses import fields
import math
from pathlib import Path
import sys

from .config import ConfigError, DECOUPLED_METHODS
from .training_adapter import IslandDataLoaderConfig


def build_initial_model(recipe, seed):
    """Reproduce the syncer's CPU initialization without changing caller RNG."""
    import torch

    if type(seed) is not int:
        raise ConfigError("initial model seed must be an integer")
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            recipe.model_spec.model.update_from_config(config=recipe)
            with torch.device("meta"):
                model = recipe.model_spec.model.build()
            model.to_empty(device="cpu")
            model.init_weights(buffer_device=None)
    finally:
        torch.set_default_dtype(dtype)
    return model


def validate_tokenizer(tokenizer, model_vocab_size):
    """Check the full token-ID range, including added/special tokens.

    Model vocab_size is an embedding/output capacity, not a requirement that
    the tokenizer use every slot. Count alone also misses sparse high IDs.
    """
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, dict) or not vocab:
        raise ConfigError("tokenizer has an empty or invalid vocabulary")
    ids = list(vocab.values())
    if any(type(token_id) is not int or token_id < 0 for token_id in ids):
        raise ConfigError("tokenizer vocabulary contains invalid token IDs")
    largest = max(ids)
    if largest >= model_vocab_size:
        raise ConfigError(
            f"tokenizer is incompatible: vocabulary={len(vocab)}, max_token_id={largest}; "
            f"model token IDs must be in 0..{model_vocab_size - 1}. "
            "Set run.hf_assets to a compatible debug tokenizer directory. No training launched."
        )
    for name in ("bos_id", "eos_id"):
        token_id = getattr(tokenizer, name, None)
        if token_id is not None and (type(token_id) is not int or not 0 <= token_id < model_vocab_size):
            raise ConfigError(f"tokenizer {name}={token_id} exceeds the model token-ID range 0..{model_vocab_size - 1}")
    sample = tokenizer.encode("A short tokenizer compatibility check.", add_bos=True, add_eos=True)
    if not sample or any(type(token_id) is not int or not 0 <= token_id < model_vocab_size for token_id in sample):
        raise ConfigError(f"tokenizer emitted invalid IDs; model token IDs must be in 0..{model_vocab_size - 1}")
    return len(vocab), largest


def validate_training_options(config, options):
    if config.method not in DECOUPLED_METHODS:
        raise ConfigError("GPU training requires a decoupled method")
    validate_dense_options(options)


def validate_dense_options(options):
    """Recipe restrictions shared by decoupled and matched baseline runs."""
    if options.gpus_per_island != 1:
        raise ConfigError("Phase 7 supports one dense GPU per island; FSDP integration is pending")
    if (options.module, options.config) != ("models.llama3_small", "llama3_15m"):
        raise ConfigError("Phase 7 supports models.llama3_small / llama3_15m only")
    if options.data_distribution != "iid" or options.languages:
        raise ConfigError("Phase 7 supports IID data shards; language-specific shards are pending")
    if options.dataset not in {"c4", "c4_test"}:
        raise ConfigError("Phase 7 supports c4 or the local c4_test dataset")
    if options.should_quantize or options.correction_heatmap or options.extra:
        raise ConfigError("Phase 7 requires should_quantize=false, correction_heatmap=false, and extra=[]")
    if options.host not in {"127.0.0.1", "localhost"}:
        raise ConfigError("Phase 7 uses localhost TCP; multi-node launch is pending")
    if not math.isfinite(options.ps_timeout) or options.ps_timeout <= 0:
        raise ConfigError("run.ps_timeout must be finite and positive")
    if len(options.island_slowness_factors) not in {1, options.islands}:
        raise ConfigError("provide one slowness factor or one factor per island")
    if any(factor < 1 for factor in options.island_slowness_factors):
        raise ConfigError("Phase 7 slowness factors must be at least 1")


def dense_parallelize(model, **_kwargs):
    # The pinned parallelize_llama always invokes fully_shard, including on
    # one device. The existing dense adapter rejects its DTensor parameters.
    # This path deliberately bypasses FSDP, AC and compile for the first run.
    return model


def build_recipe(options, repo_root, dump_folder, *, learner_id=None):
    from torchtitan.config import ConfigManager
    from torchtitan.components.checkpointer import CheckpointManager
    from torchtitan.components.optimizer import OptimizersContainer
    from torchtitan.trainer import Trainer
    from models.llama3_small import model_registry

    arguments = [
        "--module", options.module, "--config", options.config,
        "--fault_tolerance.no_enable",
        "--training.local_batch_size", str(options.batch),
        "--training.seq_len", str(options.seq_len),
        "--training.steps", str(options.steps),
        "--hf_assets_path", str((Path(repo_root) / options.hf_assets).resolve()),
        "--dump_folder", str(dump_folder),
        "--dataloader.dataset", options.dataset,
        "--debug.seed", str(options.seed),
    ]
    previous = sys.argv
    try:
        # The repository's adamw factory reads sys.argv, not parse_args(args).
        sys.argv = ["decoupled-gpu", *arguments]
        recipe = ConfigManager().parse_args(arguments)
    finally:
        sys.argv = previous
    if type(recipe.optimizer) is not OptimizersContainer.Config:
        raise ConfigError("recipe did not select the plain optimizer with TorchFT disabled")
    # Construct base Trainer explicitly: the selected registry normally owns
    # a FaultTolerantTrainer.Config, whose build() would enter the old FT plane.
    cfg = Trainer.Config(**{field.name: getattr(recipe, field.name) for field in fields(Trainer.Config)})
    # Keep the preset's document-aware FlexAttention. The pinned language
    # model APIs explicitly reject SDPA because it loses document boundaries.
    cfg.model_spec = model_registry("15M", attn_backend="flex")
    cfg.model_spec.parallelize_fn = dense_parallelize
    cfg.training.disable_cuda_graphs = True
    cfg.training.dtype = "float32"
    cfg.training.mixed_precision_param = "float32"
    cfg.training.global_batch_size = options.batch
    cfg.training.enable_cpu_offload = False
    cfg.parallelism.data_parallel_replicate_degree = 1
    cfg.parallelism.data_parallel_shard_degree = 1
    cfg.parallelism.spmd_backend = "partial_dtensor"
    cfg.debug.spmd_typechecking = False
    cfg.compile.enable = False
    cfg.activation_checkpoint = None
    cfg.checkpoint = CheckpointManager.Config(enable=False)
    cfg.validator.enable = False
    cfg.profiler.enable_profiling = False
    cfg.metrics.log_freq = 1
    cfg.metrics.enable_tensorboard = False
    cfg.metrics.enable_wandb = False
    cfg.dataloader.num_workers = 0
    if learner_id is not None:
        cfg.dataloader = IslandDataLoaderConfig(cfg.dataloader, options.islands, learner_id)
    return cfg
