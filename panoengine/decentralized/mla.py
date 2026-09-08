# Copyright (c) Panocular AI. All rights reserved.
# Licensed under the BSD-style license; see LICENSE file.
"""
MLA — Momentum Look-Ahead Outer Optimizer
==========================================
Base MLA outer optimizer for Async DiLoCo.

Simple momentum update on server side:
    m      <- γ·m + (1−γ)·Δ
    θ      <- θ − lr·(γ·m_new + Δ)

NOTE: This is the *server update rule only*. Look-ahead worker dispatch
(initializing workers at θ − lr·γ·m) is a HeLoCo component, not MLA.

Reference: MomentumLookAhead from heloco_stable_v2.ipynb
"""

import logging
from typing import Any, Optional

import torch
from torch import nn, optim

from panoengine.decentralized.async_diloco import AsyncDiLoCoServer

logger = logging.getLogger(__name__)


class MLAOptimizer(optim.Optimizer):
    """Base MLA outer optimizer for Async DiLoCo.
    
    Expects p.grad = pseudo-gradient Δ = θ_start − θ_final.
    
    Update rule:
        m  <- γ·m + (1−γ)·Δ
        θ  <- θ − lr·(γ·m_new + Δ)
    """

    def __init__(
        self,
        params: Any,
        lr: float = 0.1,
        momentum: float = 0.9,
    ) -> None:
        """Initialize MLA optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            momentum: Momentum coefficient γ (0 <= gamma < 1)
        """
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Invalid momentum: {momentum}")
        
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:
        """Apply one MLA server update.
        
        Assumes p.grad contains the incoming pseudo-gradient Δ.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            gamma = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                delta = p.grad
                state = self.state[p]
                
                # Initialize momentum buffer
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                
                m = state["m"]
                
                # Update momentum: m = γ·m + (1−γ)·Δ
                m.mul_(gamma).add_(delta, alpha=(1.0 - gamma))
                
                # Update params: θ = θ − lr·(γ·m_new + Δ)
                p.add_(m, alpha=-(lr * gamma))
                p.add_(delta, alpha=-lr)

        return loss


class MLAServer(AsyncDiLoCoServer):
    """Parameter server for MLA (Momentum Look-Ahead) distributed training.

    Extends AsyncDiLoCoServer with the MLA outer optimizer, which applies
    simple momentum look-ahead on the server side during parameter updates.

    The wire protocol is identical to AsyncDiLoCoServer's base class, so no
    changes on the worker side are required (MLAWorker = AsyncDiLoCo).

    MLA uses only two hyperparameters:
      - lr (learning rate)
      - momentum (momentum coefficient γ, where 0 ≤ γ < 1)

    Unlike HeLoCo, MLA does not perform look-ahead initialization of workers
    or block correction; it applies momentum look-ahead only at the server
    parameter update step.
    """

    def __init__(
        self,
        model: nn.Module,
        outer_optimizer: MLAOptimizer,
        **kwargs: Any,
    ) -> None:
        """Initialize MLAServer.

        Args:
            model: Global (outer) model on CPU.
            outer_optimizer: MLAOptimizer bound to model.parameters().
            **kwargs: All AsyncDiLoCoServer options (ports, hosts, auth,
                DyLU, quantization, grace period, checkpointing, …).
        """
        super().__init__(model=model, outer_optimizer=outer_optimizer, **kwargs)
