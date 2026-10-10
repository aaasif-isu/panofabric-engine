# Copyright (c) Panocular AI. All rights reserved.
# Licensed under the BSD-style license in the repository's LICENSE file.
"""Standalone tensor correction matching heloco.py's block_correct.

The existing module imports async server/torchft dependencies. Keeping this
pure-PyTorch math here permits CPU checks without that server stack. Reference
equivalence tests load the exact existing function, without editing its file.
"""

from typing import Dict, Optional

import torch


@torch.profiler.record_function("decoupled_heloco.block_correct")
def block_correct(
    pseudo_grads: Dict[str, torch.Tensor],
    momentum_buffers: Dict[str, Optional[torch.Tensor]],
    rho: float = 1.0,
    c_ok: float = 0.2,
    k_s: float = 0.5,
    k_d: float = 1.0,
    kappa: float = 3.0,
    beta_max: float = 0.5,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    corrected: Dict[str, torch.Tensor] = {}
    for name, delta in pseudo_grads.items():
        m = momentum_buffers.get(name)
        if m is None:
            corrected[name] = (rho * delta).to(delta.dtype)
            continue
        delta_f = delta.float()
        m_f = m.float()
        norm_d = delta_f.norm()
        norm_m = m_f.norm()
        safe_d = norm_d.clamp(min=eps)
        safe_m = norm_m.clamp(min=eps)
        cos_b = torch.dot(delta_f.flatten(), m_f.flatten()) / (safe_d * safe_m)
        conf_b = norm_d / (norm_d + kappa * norm_m + eps)
        u_hat = delta_f / safe_d
        v_hat = m_f / safe_m
        beta_b = torch.clamp(k_s * (-cos_b) * conf_b, max=beta_max)
        block_anti = delta_f - beta_b * cos_b * norm_d * v_hat
        lambda_b = torch.clamp(k_d * (1.0 - cos_b) * conf_b, max=1.0)
        u_mix = (1.0 - lambda_b) * u_hat + lambda_b * v_hat
        norm_mix = u_mix.norm()
        block_weak = torch.where(
            norm_mix > eps,
            norm_d * u_mix / norm_mix.clamp(min=eps),
            delta_f,
        )
        degen = (norm_d < eps) | (norm_m < eps)
        corrected_block = torch.where(
            degen | (cos_b >= c_ok), delta_f,
            torch.where(cos_b < 0.0, block_anti, block_weak),
        )
        corrected[name] = (rho * corrected_block).to(delta.dtype)
    return corrected
