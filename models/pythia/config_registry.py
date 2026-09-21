# Copyright (c) Panocular AI.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Presets for the four required Pythia sizes (genuine GPT-NeoX, see
``models/pythia/__init__.py``).

Selected by torchtitan's ConfigManager: ``--module models.pythia --config
pythia_14m`` (and 70m/410m/1b). Same shape as every other recipe's
config_registry: a bare ``FaultTolerantTrainer.Config`` with the shared
``semi_sync()`` block, so the decentralized strategy (diloco / heloco /
local_sgd) is picked at launch, not here.

``hf_assets_path`` follows the repo's ``assets/hf/<Model>`` convention (see
``models/qwen3/config_registry.py``) -- point it at a LOCAL directory holding
Pythia's own GPT-NeoX-BPE tokenizer (``tokenizer.json`` + ``tokenizer_config.json``,
vocab_size=50304; e.g. fetched once with ``hf download EleutherAI/pythia-70m
tokenizer.json tokenizer_config.json --local-dir assets/hf/pythia``, run
manually -- this module does not fetch anything itself). All four flavors
share the same tokenizer (``./assets/hf/pythia``, see ``_HF_ASSETS_PATH``
below), since Pythia's GPT-NeoX-BPE vocab is identical across sizes.
``model_registry()`` validates that tokenizer's vocab against the flavor's
vocab_size before the model is built (see ``_validate_tokenizer_vocab``);
bring-up without a tokenizer on disk yet (e.g. parameter counting) skips the
check.
"""

from dataclasses import dataclass

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from torchtitan.experiments.torchft.trainer import FaultTolerantTrainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.tools.profiler import Profiler

from panoengine.train.strategies import adamw, semi_sync

from . import model_registry


@dataclass(kw_only=True, slots=True)
class PythiaFTConfig(FaultTolerantTrainer.Config):
    hf_model: str = ""
    """HF repo id read by the transformers_modeling_backend's
    ``update_from_config`` purely to resolve model_type="gpt_neox" ->
    GPTNeoXForCausalLM via AutoConfig; every architecture dim is overridden
    by the flavor's explicit TitanModelConfig (see models/pythia/__init__.py),
    and no weights are fetched (meta-device build)."""


#: Shared tokenizer directory for every Pythia flavor (all four share one
#: GPT-NeoX-BPE tokenizer, vocab_size=50304). Not fetched by this repo -- see
#: the module docstring for the one-time `hf download` command.
_HF_ASSETS_PATH = "./assets/hf/pythia"

#: Local, checked-in GPT-NeoX config.json stub (assets/hf/pythia/config.json)
#: used ONLY so update_from_config's AutoConfig.from_pretrained() call
#: resolves architectures=["GPTNeoXForCausalLM"] from disk -- a LOCAL path,
#: so no network fetch happens. Every architecture-critical dim in it is
#: overridden by the flavor's explicit TitanModelConfig (see
#: models/pythia/__init__.py), so the stub's own dims are never read.
# _HF_MODEL_ARCH_DONOR = _HF_ASSETS_PATH

_HF_MODEL_ARCH_DONOR = {
    "14m": "./assets/hf/pythia/14m",
    "70m": "./assets/hf/pythia/70m",
    "410m": "./assets/hf/pythia/410m",
    "1b": "./assets/hf/pythia/1b",
}


def _pythia_preset(
    flavor: str,
    *,
    local_batch_size: int,
    seq_len: int,
    lr: float,
    warmup_steps: int,
) -> PythiaFTConfig:
    """Shared preset body -- only size/batch/lr/warmup differ per flavor."""
    return PythiaFTConfig(
        #hf_model=_HF_MODEL_ARCH_DONOR,
        hf_model=_HF_MODEL_ARCH_DONOR[flavor],
        loss=CrossEntropyLoss.Config(),
        hf_assets_path=_HF_ASSETS_PATH,
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
        model_spec=model_registry(flavor, hf_assets_path=_HF_ASSETS_PATH),
        optimizer=adamw(lr=lr),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=warmup_steps,
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
        fault_tolerance=semi_sync(num_fragments=1),  # no fragment_fn: whole-model DiLoCo
        validator=Validator.Config(
            enable=False,
        ),
    )


def pythia_14m() -> FaultTolerantTrainer.Config:
    """Pythia-14M: dim 128, 6 layers, 4 heads, GPT-NeoX-BPE vocab 50304."""
    return _pythia_preset("14m", local_batch_size=16, seq_len=512, lr=1e-3, warmup_steps=20)


def pythia_70m() -> FaultTolerantTrainer.Config:
    """Pythia-70M: dim 512, 6 layers, 8 heads, GPT-NeoX-BPE vocab 50304."""
    return _pythia_preset("70m", local_batch_size=8, seq_len=512, lr=1e-3, warmup_steps=20)


def pythia_410m() -> FaultTolerantTrainer.Config:
    """Pythia-410M: dim 1024, 24 layers, 16 heads, GPT-NeoX-BPE vocab 50304."""
    return _pythia_preset("410m", local_batch_size=4, seq_len=512, lr=3e-4, warmup_steps=50)


def pythia_1b() -> FaultTolerantTrainer.Config:
    """Pythia-1B: dim 2048, 16 layers, 8 heads, GPT-NeoX-BPE vocab 50304."""
    return _pythia_preset("1b", local_batch_size=2, seq_len=512, lr=2e-4, warmup_steps=100)
