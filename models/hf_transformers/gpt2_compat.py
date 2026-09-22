"""Compatibility shim for GPT-2-family models."""

import torch.nn as nn


class _GPT2LayersView(nn.Module):
    """TorchTitan-facing view over GPT-2 transformer blocks."""

    def __init__(self, blocks):
        super().__init__()

        # Keep a reference to the original GPT-2 ModuleList without
        # registering the ModuleList container itself.
        object.__setattr__(self, "_blocks", blocks)

        # Register the SAME block objects as children of this view.
        # This lets TorchTitan's to_empty()/module traversal reach them.
        for i, block in enumerate(blocks):
            self.add_module(str(i), block)

    def __iter__(self):
        return iter(self._blocks)

    def __len__(self):
        return len(self._blocks)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            idx = int(idx)
        return self._blocks[idx]

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

    def keys(self):
        return self._modules.keys()


_GPT2_NOOP_ROTARY = nn.Identity()


def enable_gpt2_compat():
    from transformers.models.gpt2.modeling_gpt2 import (
        GPT2LMHeadModel,
        GPT2Model,
    )

    if getattr(GPT2LMHeadModel, "_panofabric_gpt2_compat", False):
        return

    # GPT2LMHeadModel.transformer -> TorchTitan expects .model
    GPT2LMHeadModel.model = property(
        lambda self: self.transformer
    )

    # GPT2Model.h -> TorchTitan expects .layers
    GPT2Model.layers = property(
        lambda self: _GPT2LayersView(self.h)
    )

    # GPT2Model.wte -> TorchTitan expects .embed_tokens
    GPT2Model.embed_tokens = property(
        lambda self: self.wte
    )

    # GPT2Model.ln_f -> TorchTitan expects .norm
    GPT2Model.norm = property(
        lambda self: self.ln_f
    )

    # GPT-2 uses learned positional embeddings, not RoPE.
    # This exists only because TorchTitan expects a rotary_emb child.
    GPT2Model.rotary_emb = property(
        lambda self: self.wpe
    )

    GPT2LMHeadModel._panofabric_gpt2_compat = True
