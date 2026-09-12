# Copyright (c) Panocular AI
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Bayes — Bayesian Uncertainty-Weighted Distributed Optimizer
=============================================================
A Kalman-flavored outer optimizer for async/decentralized training that
replaces HeLoCo's hard threshold-based block correction with a principled,
element-wise uncertainty-weighted blending.

Motivation
----------
HeLoCo's block_correct() makes a discrete pass/shrink/rotate decision per
tensor block using five hand-tuned constants (c_ok, k_s, k_d, kappa,
beta_max).  This is fundamentally coarse:

* One cosine-similarity scalar per tensor block discards the rich
  element-level signal — two regions of the same weight matrix may need
  opposite corrections.
* The pass/rotate/shrink boundaries create discontinuities where tiny
  changes in cos_b produce qualitatively different behavior.
* The confidence formula conf = ‖Δ‖/(‖Δ‖ + κ‖m‖ + ε) assumes magnitude
  encodes reliability, which is not principled.

Bayes replaces all of this with a Kalman-filter-inspired update that is
continuous, element-wise, data-driven, and has interpretable hyper-
parameters.

Algorithm
---------
State per parameter (all float32):
    m  — momentum / filtered-gradient estimate
    P  — prior-uncertainty EMA (running  ‖m‖²)
    R  — measurement-noise EMA (running  ‖Δ − m‖²)

For each incoming pseudo-gradient Δ:

    1.  Kalman gain:            K = P / (P + R + ε)       ∈ [0, 1]
    2.  Innovation (blend):     G = m + K·(Δ − m)
    3.  Prior update:          P ← β_p·P + (1−β_p)·m²
        Noise update:          R ← β_r·R + (1−β_r)·(Δ − m)²
    4.  Outer momentum:        m ← μ·m + (1−μ)·G
        Outer step:            θ ← θ − lr·(G + μ·m)

Why this is better
------------------

* **Element-wise**: every scalar weight gets its own gain K.  Two
  directions in the same tensor block can receive different corrections.
* **Continuous**: no thresholds, no pass/rotate/shrink boundaries —
  K schedules smoothly between "trust momentum" and "trust the update".
* **Data-driven**: R is measured from the actual recent deviation of
  updates from the momentum trajectory, not guessed from magnitude.
* **Only 2 new hyperparameters**: beta_p (P decay) and beta_r (R decay)
  both have clear semantics as EMA smoothing factors, replacing 5 opaque
  HeLoCo constants.
* **Natural heterogeneity handling**: a noisy island (high-variance
  updates) drives R up → K down → its deviations are damped toward
  momentum.  A consistent island (low-variance updates) drives R down →
  K up → its genuinely new information passes through.

BayesServer extends AsyncDiLoCoServer (the same base as MLAServer).  The
wire protocol is unchanged — workers are standard AsyncDiLoCo/HeLoCo
workers.  All correction logic lives inside BayesOptimizer.step().

Reference
---------
    HeLoCo paper: https://arxiv.org/pdf/2606.00271
"""

import logging
from typing import Any, Optional

import torch
from torch import nn, optim

from panoengine.decentralized.async_diloco import AsyncDiLoCoServer

logger = logging.getLogger(__name__)


class BayesOptimizer(optim.Optimizer):
    """Kalman-filtered outer optimizer for decentralized training.

    Maintains per-parameter uncertainty estimates and blends each incoming
    pseudo-gradient toward momentum proportionally to relative confidence.

    State buffers (all float32 to avoid precision loss during accumulation):
        ``m`` — momentum / filtered-gradient estimate
        ``P`` — prior uncertainty EMA  (≈ running ⟨‖m‖²⟩)
        ``R`` — measurement-noise EMA  (≈ running ⟨‖Δ − m‖²⟩)

    Args:
        params:    Model parameters.
        lr:        Outer learning rate η (default 0.7).
        momentum:  Outer momentum μ ∈ [0, 1) (default 0.9).
        beta_p:    EMA decay for prior uncertainty P (default 0.99).
                   Close to 1 → P is sticky, momentum is trusted longer.
        beta_r:    EMA decay for measurement noise R (default 0.95).
                   Close to 1 → R is sticky, noise estimate adapts slowly.
        eps:       Numerical floor (default 1e-8).
    """

    def __init__(
        self,
        params: Any,
        lr: float = 0.7,
        momentum: float = 0.9,
        beta_p: float = 0.99,
        beta_r: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if not (0.0 < beta_p < 1.0):
            raise ValueError(f"beta_p must be in (0, 1), got {beta_p}")
        if not (0.0 < beta_r < 1.0):
            raise ValueError(f"beta_r must be in (0, 1), got {beta_r}")
        defaults = dict(
            lr=lr, momentum=momentum,
            beta_p=beta_p, beta_r=beta_r, eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Optional[float]:  # type: ignore[override]
        """Apply one Kalman-filtered outer step.

        Expects ``p.grad`` to contain the incoming pseudo-gradient
        ``Δ = θ_start − θ_final`` (set by the server's ``_commit_step_locked``).

        Returns:
            Loss from closure, or ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            mu: float = group["momentum"]
            beta_p: float = group["beta_p"]
            beta_r: float = group["beta_r"]
            eps: float = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # ── fetch incoming pseudo-gradient (float32 for stable math) ──
                delta = p.grad.detach().float()
                state = self.state[p]

                # ── lazily initialise state ─────────────────────────────────
                if "m" not in state:
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                    # Seed P with ‖Δ‖² so first K ≈ 1 (pass-through).
                    state["P"] = (delta * delta).clone()
                    state["R"] = torch.zeros_like(delta)

                m: torch.Tensor = state["m"]
                P: torch.Tensor = state["P"]
                R: torch.Tensor = state["R"]

                # ── 1. element-wise Kalman gain ───────────────────────────
                #  K = P / (P + R + ε)   – higher R → lower K → trust momentum
                K = P / (P + R + eps)

                # ── 2. innovation-blended pseudo-gradient ─────────────────
                #  G = m + K·(Δ − m)
                innovation = delta - m
                G = m + K * innovation

                # ── 3. update uncertainty estimates ───────────────────────
                #  (uses OLD m — true Kalman ordering: measure deviation
                #   against the prior state, not the posterior)
                P.mul_(beta_p).addcmul_(m, m, value=(1.0 - beta_p))
                R.mul_(beta_r).addcmul_(innovation, innovation,
                                        value=(1.0 - beta_r))

                # ── 4. outer momentum + parameter step (MLA backbone) ─────
                #  m_new = μ·m + (1−μ)·G
                #  θ_new = θ − lr·(G + μ·m_new)
                m.mul_(mu).add_(G, alpha=(1.0 - mu))
                p.add_(-(G + mu * m), alpha=lr)

        return loss



class BayesServer(AsyncDiLoCoServer):
    """Parameter server for Bayes uncertainty-weighted distributed training.

    Extends :class:`AsyncDiLoCoServer` with :class:`BayesOptimizer` as the
    outer optimizer.  The wire protocol is identical — workers are the
    standard AsyncDiLoCo / HeLoCo workers with no changes.

    Unlike :class:`HeLoCoServer`, this server does **not** perform separate
    block correction before committing; all correction is inside
    :meth:`BayesOptimizer.step`, called from the base-class
    ``_commit_step_locked``.

    No look-ahead worker dispatch is applied (like MLA); to ablate that
    choice, subclass and override ``_build_snapshot_locked``.

    Args:
        model:           Global (outer) model on CPU.
        outer_optimizer: :class:`BayesOptimizer` bound to
                         ``model.parameters()``.
        **kwargs:        All :class:`AsyncDiLoCoServer` options (ports,
                         hosts, DyLU, quantisation, grace period, …).
    """

    def __init__(
        self,
        model: nn.Module,
        outer_optimizer: BayesOptimizer,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, outer_optimizer=outer_optimizer, **kwargs)

