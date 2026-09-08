# Copyright (c) Panocular AI.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Presets for the small Llama3 recipes.

Selected by torchtitan's ConfigManager: ``--module models.llama3_small --config
llama3_15m``. Same shape as ``models.llama3.config_registry``: every preset is a
``FaultTolerantTrainer.Config`` with the shared ``semi_sync()`` block, so the
decentralized strategy (diloco / heloco / local_sgd) is picked at launch.
"""

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import CommConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from torchtitan.experiments.torchft.trainer import FaultTolerantTrainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.tools.profiler import Profiler

from panoengine.train.strategies import adamw, semi_sync

from . import model_registry


def llama3_15m() -> FaultTolerantTrainer.Config:
    """~15M-parameter Llama3 (dim 256, 16 layers, vocab 2048).

    The vocab matches torchtitan's 2048-token debug tokenizer, so point
    ``--hf_assets_path`` at a directory holding that tokenizer (e.g.
    ``assets/tokenizer/debug``). A real 150k-vocab tokenizer would make the
    embedding + head alone ~78M parameters.
    """
    return FaultTolerantTrainer.Config(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="./assets/tokenizer/debug",
        dump_folder="./outputs",
        profiler=Profiler.Config(
            enable_profiling=False,
            save_traces_folder="profile_trace",
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=1,
            enable_tensorboard=False,
            save_tb_folder="tb",
            enable_wandb=False,
        ),
        model_spec=model_registry("15M"),
        optimizer=adamw(lr=1e-3),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=20,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=512,
            max_norm=1.0,
            steps=1000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=TorchFTCheckpointManager.Config(
            enable=False,
            enable_ft_dataloader_checkpoints=False,
            folder="checkpoint",
            interval=500,
            last_save_model_only=True,
            export_dtype="float32",
        ),
        activation_checkpoint=SelectiveAC.Config(),
        comm=CommConfig(train_timeout_seconds=60),
        fault_tolerance=semi_sync(),
        validator=Validator.Config(
            enable=False,
        ),
    )


def _small_llama3_preset(
    flavor: str,
    *,
    local_batch_size: int,
    seq_len: int,
    lr: float,
) -> FaultTolerantTrainer.Config:
    """Shared preset body for the 500M/1B/4B from-scratch small-Llama3 flavors.

    Same vocab-2048 debug tokenizer as ``llama3_15m`` -- no HF checkpoint
    download needed, since these are algorithm experiments (DiLoCo/HeLoCo/MLA
    behaviour), not real-weight fine-tuning.
    """
    return FaultTolerantTrainer.Config(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="./assets/tokenizer/debug",
        dump_folder="./outputs",
        profiler=Profiler.Config(
            enable_profiling=False,
            save_traces_folder="profile_trace",
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=1,
            enable_tensorboard=False,
            save_tb_folder="tb",
            enable_wandb=False,
        ),
        model_spec=model_registry(flavor),
        optimizer=adamw(lr=lr),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=20,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=local_batch_size,
            seq_len=seq_len,
            max_norm=1.0,
            steps=1000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=TorchFTCheckpointManager.Config(
            enable=False,
            enable_ft_dataloader_checkpoints=False,
            folder="checkpoint",
            interval=500,
            last_save_model_only=True,
            export_dtype="float32",
        ),
        activation_checkpoint=SelectiveAC.Config(),
        comm=CommConfig(train_timeout_seconds=60),
        fault_tolerance=semi_sync(),
        validator=Validator.Config(
            enable=False,
        ),
    )


def llama3_500m() -> FaultTolerantTrainer.Config:
    """~500M-parameter Llama3 (dim 1280, 24 layers, vocab 2048, debug tokenizer)."""
    return _small_llama3_preset("500M", local_batch_size=8, seq_len=512, lr=6e-4)


def llama3_1b() -> FaultTolerantTrainer.Config:
    """~1B-parameter Llama3 (dim 1536, 36 layers, vocab 2048, debug tokenizer)."""
    return _small_llama3_preset("1B", local_batch_size=4, seq_len=512, lr=4e-4)


def llama3_4b() -> FaultTolerantTrainer.Config:
    """~4B-parameter Llama3 (dim 2560, 50 layers, vocab 2048, debug tokenizer)."""
    return _small_llama3_preset("4B", local_batch_size=2, seq_len=512, lr=2e-4)
