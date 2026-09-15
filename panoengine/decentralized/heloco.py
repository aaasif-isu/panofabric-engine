# Copyright (c) Panocular AI
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
HeLoCo — Heterogeneity-aware Low-Communication Training
=========================================================
Extends AsyncDiLoCo with two server-side modifications:

  1. Look-ahead worker initialization (Eq. 5):
     Workers receive θ̄ = θ − η·μ·m instead of θ, so they fine-tune
     from the predicted future outer-model position.

  2. Tensor-block directional correction (Algorithm 2):
     Each incoming pseudo-gradient block is compared against the current
     outer momentum. Aligned blocks pass through; anti-aligned blocks are
     shrunk; weakly-aligned blocks are rotated toward momentum while
     preserving the original block magnitude.

The worker class (HeLoCoWorker) is AsyncDiLoCo unchanged — both HeLoCo
modifications live entirely on the server.

Reference: HeLoCo paper https://arxiv.org/pdf/2606.00271.
"""

import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.profiler
from torch import nn, optim

from panoengine.decentralized.async_diloco import (
    AsyncDiLoCo,
    AsyncDiLoCoServer,
    DelayedNesterovOptimizer,
)

logger: logging.Logger = logging.getLogger(__name__)

_OTHER_ISLANDS_METHODS = ("heloco_uncorrected", "diloco")


def _parse_correction_workers(spec: Any) -> Optional[Set[int]]:
    """Parse ``correction_workers`` (``all`` | ``none`` | comma-separated
    island indices | a list/set of ints) into ``None`` (= all islands get
    correction) or a ``set[int]`` of selected island indices (``set()`` for
    ``none``)."""
    if spec is None:
        return None
    if isinstance(spec, (set, frozenset)):
        return set(int(x) for x in spec)
    if isinstance(spec, (list, tuple)):
        return set(int(x) for x in spec)
    text = str(spec).strip().lower()
    if text in ("", "all"):
        return None
    if text == "none":
        return set()
    return set(int(x) for x in text.split(",") if x.strip() != "")


class HeLoCoOptimizer(optim.Optimizer):
    """
    Outer optimizer implementing HeLoCo's MLA update rule (Eqs. 18-19):

      m_{t+1} = μ·m_t + (1−μ)·G_t
      θ_{t+1} = θ_t − η·(G_t + μ·m_{t+1})

    Block correction is applied by HeLoCoServer *before* setting p.grad,
    so this optimizer receives the already-corrected gradient G_t and only
    applies the plain momentum-lookahead update.

    Momentum buffers are stored as float32 regardless of parameter dtype
    to avoid precision loss during accumulation.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 0.7,
        momentum: float = 0.9,
    ) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:  # type: ignore[override]
        """
        Apply one outer step using the corrected gradient already in p.grad.

        Eqs. 18-19:
          m_{t+1} = μ·m_t + (1−μ)·G_t
          θ_{t+1} = θ_t − η·(G_t + μ·m_{t+1})
        """
        for group in self.param_groups:
            lr: float = group["lr"]
            mu: float = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                G = p.grad.detach().float()
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                m: torch.Tensor = state["m"]
                m.mul_(mu).add_(G, alpha=1.0 - mu)       # Eq. 18
                p.add_(-(G + mu * m), alpha=lr)           # Eq. 19


@torch.profiler.record_function("heloco.block_correct")
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
    """
    Algorithm 2: Tensor-block directional correction (paper Eqs. 9-15).

    Three cases per block b:
      cos_b ≥ c_ok  → pass through (Eq. 9)
      cos_b < 0     → shrink: Δ̂_b = Δ_b − β_b·cos_b·‖Δ_b‖·v̂_b, β_b = clamp(k_s·(-cos_b)·conf_b, β_max) (Eqs. 10-11)
      otherwise     → rotate: Δ̂_b = ‖Δ_b‖·ũ_mix/‖ũ_mix‖, ũ_mix=(1−λ_b)û_b+λ_b·v̂_b (Eqs. 12-14)
      conf_b = ‖Δ_b‖/(‖Δ_b‖+κ‖m_b‖+ε) (Eq. 15)

    No .item() calls — all math stays in tensor land, enabling PyTorch to fuse
    ops and avoid per-scalar Python/C++ round-trips.
    """
    corrected: Dict[str, torch.Tensor] = {}

    for name, delta in pseudo_grads.items():
        m = momentum_buffers.get(name)
        if m is None:
            corrected[name] = (rho * delta).to(delta.dtype)
            continue

        delta_f = delta.float()
        m_f     = m.float()
        norm_d  = delta_f.norm()
        norm_m  = m_f.norm()
        safe_d  = norm_d.clamp(min=eps)
        safe_m  = norm_m.clamp(min=eps)

        cos_b  = torch.dot(delta_f.flatten(), m_f.flatten()) / (safe_d * safe_m)  # Eqs. 7-8
        conf_b = norm_d / (norm_d + kappa * norm_m + eps)                          # Eq. 15

        u_hat = delta_f / safe_d
        v_hat = m_f     / safe_m

        # Anti-aligned case (Eqs. 10-11)
        beta_b     = torch.clamp(k_s * (-cos_b) * conf_b, max=beta_max)
        block_anti = delta_f - beta_b * cos_b * norm_d * v_hat

        # Weakly-aligned case (Eqs. 12-14)
        lambda_b   = torch.clamp(k_d * (1.0 - cos_b) * conf_b, max=1.0)
        u_mix      = (1.0 - lambda_b) * u_hat + lambda_b * v_hat
        norm_mix   = u_mix.norm()
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


@torch.profiler.record_function("heloco.whole_gradient_correct")
def whole_gradient_correct(
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
    """``correction_scope: whole_gradient`` variant of :func:`block_correct`.

    Instead of one cosine/confidence/correction decision PER TENSOR, the
    entire worker pseudo-gradient (and its matching momentum, zero-filled for
    any tensor whose momentum hasn't been seeded yet) is flattened into ONE
    logical vector and exactly ONE decision is made and applied consistently
    across every tensor of the update.

    Reuses :func:`block_correct`'s equations exactly (Eqs. 9-15) by calling it
    on a single synthetic ``"__whole_gradient__"`` block spanning the
    concatenation of every tensor -- no separate correction math.
    """
    names: List[str] = list(pseudo_grads.keys())
    shapes = {n: pseudo_grads[n].shape for n in names}
    numels = {n: pseudo_grads[n].numel() for n in names}
    dtypes = {n: pseudo_grads[n].dtype for n in names}

    delta_flat = torch.cat([pseudo_grads[n].reshape(-1).float() for n in names])
    if all(momentum_buffers.get(n) is None for n in names):
        m_flat: Optional[torch.Tensor] = None
    else:
        m_flat = torch.cat(
            [
                (
                    momentum_buffers[n].float()
                    if momentum_buffers.get(n) is not None
                    else torch.zeros_like(pseudo_grads[n], dtype=torch.float32)
                ).reshape(-1)
                for n in names
            ]
        )

    corrected_flat = block_correct(
        {"__whole_gradient__": delta_flat},
        {"__whole_gradient__": m_flat},
        rho=rho,
        c_ok=c_ok,
        k_s=k_s,
        k_d=k_d,
        kappa=kappa,
        beta_max=beta_max,
        eps=eps,
    )["__whole_gradient__"]

    corrected: Dict[str, torch.Tensor] = {}
    offset = 0
    for n in names:
        n_el = numels[n]
        corrected[n] = (
            corrected_flat[offset : offset + n_el].view(shapes[n]).to(dtypes[n])
        )
        offset += n_el
    return corrected


class HeLoCoServer(AsyncDiLoCoServer):
    """
    Parameter server for HeLoCo distributed training.

    Extends AsyncDiLoCoServer with:
      1. Look-ahead initialization: both pull-only and post-full-sync
         responses send θ̄ = θ − η·μ·m so workers train from the
         predicted future outer position (Algorithm 1, line 3).
      2. Block correction: before each outer step, the incoming
         pseudo-gradient is passed through block_correct() to align
         it with the outer momentum (Algorithm 2).

    Both modifications hook into the base class (:meth:`_apply_one` and
    :meth:`_build_snapshot_locked`); the wire protocol is exactly
    :meth:`AsyncDiLoCoServer.forward`, so no changes on the worker side
    are required (HeLoCoWorker = AsyncDiLoCo). The look-ahead snapshot is
    computed at most once per revision via the base snapshot cache.

    DyLU is inherited and continues to work when dylu_H > 0.
    """

    def __init__(
        self,
        model: nn.Module,
        outer_optimizer: HeLoCoOptimizer,
        rho: float = 1.0,
        c_ok: float = 0.2,
        k_s: float = 0.5,
        k_d: float = 1.0,
        kappa: float = 3.0,
        beta_max: float = 0.5,
        eps: float = 1e-8,
        correction_workers: Any = "all",
        correction_scope: str = "tensorwise",
        correction_heatmap: bool = False,
        correction_heatmap_dir: Optional[str] = None,
        other_islands_method: str = "heloco_uncorrected",
        diloco_lr: Optional[float] = None,
        diloco_momentum: Optional[float] = None,
        diloco_nesterov_period: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model: Global (outer) model on CPU.
            outer_optimizer: HeLoCoOptimizer bound to model.parameters().
            rho: Arrival weight ρ applied after block correction.
                 Paper recommends 1/√K for K concurrent workers.
            c_ok: Alignment threshold (default 0.2).
            k_s: Anti-aligned shrinkage strength (default 0.5).
            k_d: Weakly-aligned rotation strength (default 1.0).
            kappa: Confidence factor momentum scale κ (default 3.0).
            beta_max: Shrinkage coefficient cap (default 0.5).
            eps: Numerical floor (default 1e-8).
            correction_workers: Which islands get HeLoCo correction --
                ``"all"`` (default, matches prior behavior), ``"none"``, a
                comma-separated string of island indices (e.g. ``"0,2"``),
                or a list/set of ints. Islands NOT selected still train and
                push/pull normally; what happens to their pseudo-gradient is
                governed by ``other_islands_method``.
            correction_scope: ``"tensorwise"`` (default) makes one
                cosine/confidence/correction decision per tensor (Algorithm 2
                exactly); ``"whole_gradient"`` flattens the whole update into
                one vector and makes a single decision applied to all of it.
            correction_heatmap: If True (and ``correction_scope ==
                "tensorwise"``), record one row per corrected worker/update/
                tensor (island id, step/exchange index, tensor name, cosine
                similarity, correction type, correction magnitude, normalized
                correction magnitude) for regenerating a heatmap later. With
                ``correction_scope == "whole_gradient"``, one whole-update
                row is recorded per push instead.
            correction_heatmap_dir: Directory the heatmap CSV is written
                under (``correction_heatmap.csv``). Defaults to
                ``$PANOFABRIC_COMM_LOG_DIR/../correction_heatmap`` (mirrors
                the comm-metrics CSV convention) when unset.
            other_islands_method: What non-selected (``correction_workers``)
                islands get instead of HeLoCo directional correction --
                ``"heloco_uncorrected"`` (default, matches prior behavior):
                the raw pseudo-gradient (rho-scaled only) is committed
                through :class:`HeLoCoOptimizer` -- these islands still use
                the HeLoCo MLA update and look-ahead dispatch, only the
                Algorithm-2 directional correction is skipped (an ablation
                of the correction step only).
                ``"diloco"``: non-selected islands are true async DiLoCo --
                committed through a SEPARATE :class:`DelayedNesterovOptimizer`
                (its own independent momentum state) and sent the plain
                global-model snapshot on pull, never the HeLoCo look-ahead
                one. Selected islands are unaffected either way (always full
                HeLoCo: correction + HeLoCoOptimizer/MLA + look-ahead).
            diloco_lr: Outer lr for the true-DiLoCo optimizer used by
                non-selected islands when ``other_islands_method ==
                "diloco"``. Defaults to the HeLoCo optimizer's own lr.
            diloco_momentum: Momentum for the true-DiLoCo optimizer.
                Defaults to the HeLoCo optimizer's own momentum.
            diloco_nesterov_period: ``DelayedNesterovOptimizer``'s
                ``nesterov_period`` (pushes between momentum corrections;
                should be >= the number of non-selected islands). Only used
                when ``other_islands_method == "diloco"``.
            **kwargs: All :class:`AsyncDiLoCoServer` options (ports, hosts,
                auth, DyLU, quantization, grace period, checkpointing, …).
        """
        # Set HeLoCo attrs before super().__init__ launches the server thread
        self._rho = rho
        self._c_ok = c_ok
        self._k_s = k_s
        self._k_d = k_d
        self._kappa = kappa
        self._beta_max = beta_max
        self._eps = eps
        # correction_workers: None means "all" (every island corrected);
        # a set (possibly empty, for "none") is the explicit island allow-list.
        self._correction_workers: Optional[Set[int]] = _parse_correction_workers(
            correction_workers
        )
        if correction_scope not in ("tensorwise", "whole_gradient"):
            raise ValueError(
                f"correction_scope must be 'tensorwise' or 'whole_gradient', "
                f"got {correction_scope!r}"
            )
        self._correction_scope = correction_scope
        self._correction_heatmap = bool(correction_heatmap)
        self._correction_heatmap_dir = correction_heatmap_dir or os.environ.get(
            "PANOFABRIC_COMM_LOG_DIR", ""
        )
        self._heatmap_exchange_idx: int = 0

        if other_islands_method not in _OTHER_ISLANDS_METHODS:
            raise ValueError(
                f"other_islands_method must be one of "
                f"{_OTHER_ISLANDS_METHODS}, got {other_islands_method!r}"
            )
        self._other_islands_method = other_islands_method
        # A SEPARATE outer optimizer + momentum state for non-selected
        # islands when other_islands_method == "diloco". It is bound to the
        # SAME nn.Parameter objects as outer_optimizer (the one shared
        # global model), but torch.optim.Optimizer keys its state dict by
        # parameter-object identity in a dict private to the optimizer
        # INSTANCE, so HeLoCo's "m" momentum buffer and this DiLoCo
        # optimizer's "grad_buffer"/"m" buffers never collide or get
        # silently reused as each other.
        self._diloco_optimizer: Optional[DelayedNesterovOptimizer] = None
        if other_islands_method == "diloco":
            self._diloco_optimizer = DelayedNesterovOptimizer(
                model.parameters(),
                lr=(
                    diloco_lr if diloco_lr is not None
                    else outer_optimizer.defaults["lr"]
                ),
                momentum=(
                    diloco_momentum if diloco_momentum is not None
                    else outer_optimizer.defaults["momentum"]
                ),
                nesterov_period=diloco_nesterov_period,
            )
        super().__init__(model=model, outer_optimizer=outer_optimizer, **kwargs)

    def _worker_gets_correction(self, island_id: int) -> bool:
        """True unless this worker's island was explicitly excluded via
        ``correction_workers``. Unknown island id (-1, no PF_ISLAND_ID set)
        always gets correction -- the pre-existing default behavior."""
        if self._correction_workers is None or island_id < 0:
            return True
        return island_id in self._correction_workers

    def _worker_is_diloco(self, island_id: int) -> bool:
        """True iff this push/pull should use true DiLoCo semantics --
        only possible when ``other_islands_method == "diloco"`` AND this
        island is NOT one of the HeLoCo-selected ``correction_workers``.
        An unknown island id (-1) is always treated as HeLoCo-selected
        (matches :meth:`_worker_gets_correction`'s default)."""
        return (
            self._other_islands_method == "diloco"
            and self._diloco_optimizer is not None
            and not self._worker_gets_correction(island_id)
        )

    def _log_heatmap_rows(
        self,
        island_id: int,
        rows: List[Dict[str, Any]],
    ) -> None:
        """Append correction-heatmap rows to
        ``<correction_heatmap_dir>/correction_heatmap.csv``.

        Reuses the same CSV-append convention as
        :meth:`AsyncDiLoCo._emit_comm_metrics` (write header once, then
        append one row per record) instead of building a second logging
        mechanism.
        """
        if not rows:
            return
        import csv as _csv

        out_dir = self._correction_heatmap_dir or "."
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "correction_heatmap.csv")
        header_needed = not os.path.exists(path)
        fieldnames = [
            "island_id",
            "exchange",
            "tensor",
            "cosine_similarity",
            "correction_type",
            "correction_magnitude",
            "normalized_correction_magnitude",
        ]
        with open(path, "a", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            if header_needed:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @torch.profiler.record_function("heloco.lookahead_snapshot")
    def _lookahead_snapshot(self, names: List[str]) -> Dict[str, torch.Tensor]:
        """Compute θ̄ = θ − η·μ·m (Eq. 5) for the parameters in ``names`` —
        one fragment's slice, or all of them. Must hold self._lock."""
        param_to_hyper: Dict[int, Tuple[float, float]] = {}
        for group in self._outer_optimizer.param_groups:
            lr: float = group["lr"]
            mu: float = group["momentum"]
            for p in group["params"]:
                param_to_hyper[id(p)] = (lr, mu)

        snapshot: Dict[str, torch.Tensor] = {}
        # Group by lr*mu so _foreach_sub can process all params in one C++ call.
        scale_groups: Dict[float, List[Tuple[str, torch.nn.Parameter, torch.Tensor]]] = defaultdict(list)

        for name in names:
            p = self._params_by_name[name]
            state = self._outer_optimizer.state.get(p)
            m = state["m"] if (state and "m" in state) else None
            lr, mu = param_to_hyper.get(id(p), (0.0, 0.0))
            if m is not None and mu > 0.0:
                scale_groups[lr * mu].append((name, p, m))
            else:
                snapshot[name] = p.data.clone().detach()  # momentum not yet seeded — send raw θ

        for scale, items in scale_groups.items():
            ps_f = [p.data.float() for _, p, _ in items]
            ms   = [m for _, _, m in items]
            for (name, p, _), la in zip(items, torch._foreach_sub(ps_f, ms, alpha=scale)):
                snapshot[name] = la.to(p.dtype).detach()

        return snapshot

    def _snapshot_variant(self, island_id: int) -> str:
        # Two distinct snapshot CONTENTS can exist at the same revision when
        # other_islands_method == "diloco": the HeLoCo look-ahead one and
        # the plain-params one. Keep them as separate cache entries.
        return "diloco" if self._worker_is_diloco(island_id) else "heloco"

    def _build_snapshot_locked(
        self, names: List[str], island_id: int = -1
    ) -> Dict[str, torch.Tensor]:
        # HeLoCo-selected islands (and, for other_islands_method ==
        # "heloco_uncorrected", every island) receive the look-ahead
        # position θ̄, never raw θ. A true-DiLoCo island
        # (other_islands_method == "diloco") receives the plain global-model
        # snapshot instead -- normal DiLoCo dispatch, no look-ahead shift.
        if self._worker_is_diloco(island_id):
            return {name: self._params_by_name[name].data for name in names}
        return self._lookahead_snapshot(names)

    @torch.profiler.record_function("heloco.apply")
    def _apply_one(
        self,
        pseudo_grads: Dict[str, torch.Tensor],
        fragment: int = 0,
        island_id: int = -1,
    ) -> None:
        """Block-correct one worker's pseudo-gradient, then commit the outer step.

        Clone momentum (brief lock) → block_correct (no lock) → commit (lock).
        Sequential grace-batch workers therefore each correct against the
        momentum updated by the previous worker's step (paper Algorithm 2
        ordering).

        Momentum is cloned only for the parameters this push covers
        (``pseudo_grads`` keys — one fragment's slice under fragment-wise
        sync, which also shrinks the per-push clone to model/P). Block
        correction is per-parameter, so a fragment push corrects and commits
        bitwise the same values a whole-model push of the same deltas would.

        ``island_id`` (the pushing worker's island index, -1 if unknown)
        gates whether HeLoCo correction runs at all (``correction_workers``):
        a non-selected island's pseudo-gradient is either committed unchanged
        through HeLoCoOptimizer (``other_islands_method ==
        "heloco_uncorrected"``, ``rho``-scaled only) or handed to a
        SEPARATE true-DiLoCo optimizer (``other_islands_method ==
        "diloco"``).
        """
        if self._worker_is_diloco(island_id):
            # True async DiLoCo for this push: no block correction, no
            # HeLoCo momentum/heatmap involvement at all -- committed
            # through the independent DelayedNesterovOptimizer instance so
            # its momentum state never touches HeLoCoOptimizer's.
            with self._lock:
                self._commit_step_locked(
                    pseudo_grads, fragment, optimizer=self._diloco_optimizer
                )
            return

        with self._lock:
            mom_bufs: Dict[str, Optional[torch.Tensor]] = {}
            for name in pseudo_grads:
                p = self._params_by_name[name]
                state = self._outer_optimizer.state.get(p)
                m = state["m"] if (state and "m" in state) else None
                mom_bufs[name] = m.clone() if m is not None else None

        gets_correction = self._worker_gets_correction(island_id)
        if not gets_correction:
            # Bypass HeLoCo correction entirely: commit the raw pseudo-
            # gradient (rho-scaled, matching block_correct's degenerate
            # "no momentum yet" branch) -- plain async DiLoCo behavior.
            corrected = {
                name: (self._rho * delta).to(delta.dtype)
                for name, delta in pseudo_grads.items()
            }
        elif self._correction_scope == "whole_gradient":
            corrected = whole_gradient_correct(
                pseudo_grads,
                mom_bufs,
                rho=self._rho,
                c_ok=self._c_ok,
                k_s=self._k_s,
                k_d=self._k_d,
                kappa=self._kappa,
                beta_max=self._beta_max,
                eps=self._eps,
            )
        else:
            corrected = block_correct(
                pseudo_grads,
                mom_bufs,
                rho=self._rho,
                c_ok=self._c_ok,
                k_s=self._k_s,
                k_d=self._k_d,
                kappa=self._kappa,
                beta_max=self._beta_max,
                eps=self._eps,
            )

        if self._correction_heatmap and gets_correction:
            try:
                self._record_heatmap(pseudo_grads, mom_bufs, corrected, island_id)
            except Exception:
                logger.exception("correction heatmap logging failed")

        with self._lock:
            self._commit_step_locked(corrected, fragment)

    def _record_heatmap(
        self,
        pseudo_grads: Dict[str, torch.Tensor],
        mom_bufs: Dict[str, Optional[torch.Tensor]],
        corrected: Dict[str, torch.Tensor],
        island_id: int,
    ) -> None:
        """Build and append correction-heatmap rows for one applied push.

        ``tensorwise``: one row per tensor with that tensor's own cosine
        similarity (Eqs. 7-8) and correction type, recomputed the same way
        :func:`block_correct` derives them internally (no new math).
        ``whole_gradient``: one row for the whole update instead, matching
        the single flattened decision :func:`whole_gradient_correct` made.
        """
        self._heatmap_exchange_idx += 1
        exchange = self._heatmap_exchange_idx
        rows: List[Dict[str, Any]] = []

        if self._correction_scope == "whole_gradient":
            names = list(pseudo_grads.keys())
            delta_flat = torch.cat(
                [pseudo_grads[n].reshape(-1).float() for n in names]
            )
            m_flat = torch.cat(
                [
                    (
                        mom_bufs[n].float()
                        if mom_bufs.get(n) is not None
                        else torch.zeros_like(pseudo_grads[n], dtype=torch.float32)
                    ).reshape(-1)
                    for n in names
                ]
            )
            corrected_flat = torch.cat(
                [corrected[n].reshape(-1).float() for n in names]
            )
            row = self._heatmap_row(
                island_id, exchange, "__whole_gradient__",
                delta_flat, m_flat, corrected_flat,
            )
            if row is not None:
                rows.append(row)
        else:
            for name, delta in pseudo_grads.items():
                m = mom_bufs.get(name)
                m_f = m.float() if m is not None else None
                row = self._heatmap_row(
                    island_id, exchange, name,
                    delta.float(), m_f, corrected[name].float(),
                )
                if row is not None:
                    rows.append(row)

        self._log_heatmap_rows(island_id, rows)

    def _heatmap_row(
        self,
        island_id: int,
        exchange: int,
        tensor_name: str,
        delta_f: torch.Tensor,
        m_f: Optional[torch.Tensor],
        corrected_f: torch.Tensor,
    ) -> Optional[Dict[str, Any]]:
        """One heatmap row for a single (tensor or whole-gradient) block.

        cosine/correction-type derivation mirrors :func:`block_correct`
        exactly (Eqs. 7-9); no new correction math is introduced here --
        this only re-derives the SAME scalars block_correct already computed
        internally, for logging.
        """
        norm_d = delta_f.norm().item()
        if m_f is None or norm_d < self._eps:
            cos_b = float("nan")
            corr_type = "pass"
        else:
            norm_m = m_f.norm().item()
            if norm_m < self._eps:
                cos_b = float("nan")
                corr_type = "pass"
            else:
                cos_b = (
                    torch.dot(delta_f.flatten(), m_f.flatten()).item()
                    / (norm_d * norm_m)
                )
                if cos_b >= self._c_ok:
                    corr_type = "pass"
                elif cos_b < 0.0:
                    corr_type = "shrink"
                else:
                    corr_type = "rotate"

        magnitude = (corrected_f - delta_f).norm().item()
        normalized_magnitude = magnitude / (norm_d + self._eps)

        return {
            "island_id": island_id,
            "exchange": exchange,
            "tensor": tensor_name,
            "cosine_similarity": cos_b,
            "correction_type": corr_type,
            "correction_magnitude": magnitude,
            "normalized_correction_magnitude": normalized_magnitude,
        }


# Workers are standard AsyncDiLoCo — all HeLoCo logic lives on the server.
HeLoCoWorker = AsyncDiLoCo
