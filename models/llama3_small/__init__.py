# Copyright (c) Panocular AI.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Small Llama3-architecture recipes (tens of millions of parameters).

Mirrors ``models.llama3`` exactly -- same parallelize / pipeline / fragment /
state-dict-adapter functions from torchtitan -- but builds its own
``Llama3Model.Config`` sizes instead of the upstream ``llama3_configs`` table,
which only offers ``debugmodel`` and >= 1B. Meant for algorithm experiments
(DiLoCo / HeLoCo behaviour) where a real-sized model is not the point.

Flavors are registered in ``SMALL_CONFIGS``; add one by adding a builder there.
"""

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.experiments.torchft.config.job_config import FaultTolerantModelSpec
from torchtitan.experiments.torchft.diloco import fragment_llm
from torchtitan.models.common import (
    ComplexRoPE,
    compute_ffn_hidden_dim,
    Embedding,
    Linear,
    RMSNorm,
)
from torchtitan.models.common.config_utils import make_ffn_config, make_gqa_config
from torchtitan.models.common.config_utils import get_attention_config
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.llama3 import Llama3StateDictAdapter, parallelize_llama
from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock

# Same init policy as torchtitan/models/llama3/__init__.py.
_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def build_small_llama3(
    *,
    dim: int,
    n_layers: int,
    n_heads: int,
    vocab_size: int,
    n_kv_heads: int | None = None,
    max_seq_len: int = 8192,
    rope_theta: float = 500000.0,
    attn_backend: str = "flex",
) -> Llama3Model.Config:
    """A Llama3 model config of arbitrary size.

    Parameter count ~= 2 * vocab_size * dim (embeddings + untied head)
                     + n_layers * (4 * dim^2 + 3 * dim * ffn_hidden_dim).
    """
    hidden_dim = compute_ffn_hidden_dim(dim, multiple_of=256)
    rope = ComplexRoPE.Config(
        dim=dim // n_heads,
        max_seq_len=max_seq_len,
        theta=rope_theta,
        scaling="llama",
    )
    inner_attention = get_attention_config(attn_backend)
    layers = [
        Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
            attention=make_gqa_config(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                wqkv_param_init=_LINEAR_INIT,
                wo_param_init=_depth_init(layer_id),
                inner_attention=inner_attention,
                fuse_qkv=True,
                rope=rope,
            ),
            feed_forward=make_ffn_config(
                dim=dim,
                hidden_dim=hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            ),
        )
        for layer_id in range(n_layers)
    ]
    return Llama3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim, out_features=vocab_size, param_init=_output_linear_init(dim)
        ),
        layers=layers,
    )


def _15m(attn_backend: str) -> Llama3Model.Config:
    # 16 layers x (4*256^2 + 3*256*768) = 13.6M, + 2 * 2048 * 256 = 1.0M  -> ~14.7M.
    # vocab_size=2048 matches torchtitan's debug tokenizer (tests/assets/tokenizer).
    return build_small_llama3(
        dim=256, n_layers=16, n_heads=8, n_kv_heads=4, vocab_size=2048,
        attn_backend=attn_backend,
    )


def _500m(attn_backend: str) -> Llama3Model.Config:
    # 24 layers x (4*1280^2 + 3*1280*3328) ~= 464M, + 2*2048*1280 ~= 5.2M -> ~469M.
    return build_small_llama3(
        dim=1280, n_layers=24, n_heads=16, n_kv_heads=8, vocab_size=2048,
        attn_backend=attn_backend,
    )


def _1b(attn_backend: str) -> Llama3Model.Config:
    # 36 layers x (4*1536^2 + 3*1536*4096) ~= 1019M, + 2*2048*1536 ~= 6.3M -> ~1.02B.
    return build_small_llama3(
        dim=1536, n_layers=36, n_heads=24, n_kv_heads=12, vocab_size=2048,
        attn_backend=attn_backend,
    )


def _4b(attn_backend: str) -> Llama3Model.Config:
    # 50 layers x (4*2560^2 + 3*2560*6912) ~= 3965M, + 2*2048*2560 ~= 10.5M -> ~3.98B.
    return build_small_llama3(
        dim=2560, n_layers=50, n_heads=20, n_kv_heads=10, vocab_size=2048,
        attn_backend=attn_backend,
    )


#: flavor name -> builder(attn_backend) -> Llama3Model.Config
#: All flavors use the 2048-vocab debug tokenizer (assets/tokenizer/debug) since
#: these are from-scratch algorithm experiments, not loading real pretrained weights.
SMALL_CONFIGS: dict[str, Callable[[str], Llama3Model.Config]] = {
    "15M": _15m,
    "500M": _500m,
    "1B": _1b,
    "4B": _4b,
}


def model_registry(flavor: str, attn_backend: str = "flex") -> FaultTolerantModelSpec:
    if flavor not in SMALL_CONFIGS:
        raise ValueError(f"unknown flavor {flavor!r}; choose from {sorted(SMALL_CONFIGS)}")
    return FaultTolerantModelSpec(
        name="torchft/llama3_small",
        flavor=flavor,
        model=SMALL_CONFIGS[flavor](attn_backend),
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
        fragment_fn=fragment_llm,
    )
