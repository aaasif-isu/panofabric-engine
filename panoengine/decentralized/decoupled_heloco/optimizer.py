"""Fragment outer optimizers: toy SGD, standard Nesterov DiLoCo, and HeLoCo."""

from collections.abc import Mapping
import math

import torch

from .fragment_manager import FragmentManager
from .config import HeLoCoConfig
from .correction import block_correct
from .state import _cpu_copy


class FragmentSGD:
    def __init__(self, manager: FragmentManager, parameters: Mapping[str, torch.Tensor], *, lr: float):
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr < 0:
            raise ValueError("outer lr must be finite and nonnegative")
        manager.validate_model(parameters)
        self.manager = manager
        self.lr = lr
        self._parameters = _cpu_copy(parameters)

    def snapshot(self, fragment_id: int) -> dict[str, torch.Tensor]:
        return {name: tensor.clone() for name, tensor in self.manager.select(fragment_id, self._parameters).items()}

    def model_snapshot(self) -> dict[str, torch.Tensor]:
        return {name: tensor.clone() for name, tensor in self._parameters.items()}

    def step(self, fragment_id: int, gradient: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.manager.validate_update(fragment_id, gradient)
        gradient = _cpu_copy(gradient)
        if any(not bool(torch.isfinite(tensor).all()) for tensor in gradient.values()):
            raise ValueError("outer pseudo-gradient contains nonfinite values")
        staged = {name: self._parameters[name] - self.lr * tensor for name, tensor in gradient.items()}
        if any(not bool(torch.isfinite(tensor).all()) for tensor in staged.values()):
            raise ValueError("outer update would produce nonfinite parameters")
        dispatch = _cpu_copy(staged)
        # Complete validation/allocation before changing any parameter entry.
        self._parameters.update(staged)
        return dispatch


class FragmentDiLoCo(FragmentSGD):
    """Standard Nesterov-SGD on a merged fragment, with plain global dispatch.

    b_next = momentum * b + gradient
    theta_next = theta - lr * (gradient + momentum * b_next)

    The first buffer is the gradient, as in torch.optim.SGD with dampening=0
    and nesterov=True. This uses a sum buffer, whereas HeLoCo uses an EMA.
    There is no return correction or extra look-ahead of outgoing weights.
    All weights, buffers, and dispatch copies are staged before any commit.
    """

    def __init__(self, manager, parameters, *, lr, momentum=0.9):
        if isinstance(momentum, bool) or not isinstance(momentum, (int, float)) or not math.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError("outer momentum must be finite and in [0, 1)")
        super().__init__(manager, parameters, lr=lr)
        self.momentum = momentum
        self._momentum: dict[str, torch.Tensor] = {}

    def momentum_snapshot(self, fragment_id):
        return {
            name: self._momentum[name].clone() if name in self._momentum else None
            for name in self.manager.fragment(fragment_id).parameter_names
        }

    def dispatch_snapshot(self, fragment_id):
        return self.snapshot(fragment_id)

    def step(self, fragment_id, gradient):
        self.manager.validate_update(fragment_id, gradient)
        gradient = _cpu_copy(gradient)
        if any(not bool(torch.isfinite(value).all()) for value in gradient.values()):
            raise ValueError("outer pseudo-gradient contains nonfinite values")
        staged_weights, staged_momentum, dispatch = {}, {}, {}
        for name, value in gradient.items():
            direction = value
            if self.momentum:
                previous = self._momentum.get(name)
                moment = value.clone() if previous is None else previous.clone().mul_(self.momentum).add_(value)
                direction = value.add(moment, alpha=self.momentum)
                if not bool(torch.isfinite(moment).all()):
                    raise ValueError("DiLoCo update would produce nonfinite momentum")
                staged_momentum[name] = moment
            parameter = self._parameters[name].clone().add_(direction, alpha=-self.lr)
            if not bool(torch.isfinite(parameter).all()):
                raise ValueError("DiLoCo update would produce nonfinite parameters")
            staged_weights[name] = parameter
            dispatch[name] = parameter.clone()
        self._parameters.update(staged_weights)
        self._momentum.update(staged_momentum)
        return dispatch


class FragmentHeLoCo(FragmentSGD):
    """Correct a merged fragment, update its momentum, and dispatch look-ahead.

    Momentum is FP32 and per parameter tensor, allocated only when that tensor
    first participates. Every tensor result/dispatch is staged and validated
    before changing global weights or momentum. Other fragments are untouched.
    """

    def __init__(
        self, manager: FragmentManager, parameters: Mapping[str, torch.Tensor], *,
        lr: float, momentum: float = 0.9, config: HeLoCoConfig | None = None,
    ):
        if isinstance(momentum, bool) or not isinstance(momentum, (int, float)) or not math.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError("outer momentum must be finite and in [0, 1)")
        self.config = config if config is not None else HeLoCoConfig()
        self.config.validate()
        super().__init__(manager, parameters, lr=lr)
        self.momentum = momentum
        self._momentum: dict[str, torch.Tensor] = {}

    def momentum_snapshot(self, fragment_id: int) -> dict[str, torch.Tensor | None]:
        return {
            name: self._momentum[name].clone() if name in self._momentum else None
            for name in self.manager.fragment(fragment_id).parameter_names
        }

    def dispatch_snapshot(self, fragment_id: int) -> dict[str, torch.Tensor]:
        values = self.snapshot(fragment_id)
        if self.config.lookahead:
            for name, tensor in values.items():
                if name in self._momentum:
                    values[name] = tensor.sub(self._momentum[name], alpha=self.lr * self.momentum)
        return values

    def step(self, fragment_id: int, gradient: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.manager.validate_update(fragment_id, gradient)
        gradient = _cpu_copy(gradient)
        if any(not bool(torch.isfinite(tensor).all()) for tensor in gradient.values()):
            raise ValueError("outer pseudo-gradient contains nonfinite values")
        if self.config.correction_enabled:
            corrected = block_correct(
                gradient, {name: self._momentum.get(name) for name in gradient},
                rho=self.config.rho, c_ok=self.config.c_ok, k_s=self.config.k_s,
                k_d=self.config.k_d, kappa=self.config.kappa,
                beta_max=self.config.beta_max, eps=self.config.eps,
            )
        else:
            corrected = {name: self.config.rho * value for name, value in gradient.items()}
        staged_weights = {}
        staged_momentum = {}
        dispatch = {}
        for name, value in corrected.items():
            if not bool(torch.isfinite(value).all()):
                raise ValueError("corrected pseudo-gradient contains nonfinite values")
            previous = self._momentum.get(name)
            moment = torch.zeros_like(value) if previous is None else previous.clone()
            moment.mul_(self.momentum).add_(value, alpha=1.0 - self.momentum)
            parameter = self._parameters[name].clone()
            parameter.add_(-(value + self.momentum * moment), alpha=self.lr)
            outgoing = parameter.sub(moment, alpha=self.lr * self.momentum) if self.config.lookahead else parameter.clone()
            if any(not bool(torch.isfinite(tensor).all()) for tensor in (moment, parameter, outgoing)):
                raise ValueError("HeLoCo update would produce nonfinite state or dispatch")
            staged_weights[name] = parameter
            staged_momentum[name] = moment
            dispatch[name] = outgoing
        self._parameters.update(staged_weights)
        self._momentum.update(staged_momentum)
        return dispatch
