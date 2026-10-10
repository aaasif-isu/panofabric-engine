"""Paper contribution weighting and per-tensor radial-directional merging."""

import math
from itertools import chain
import torch


def contribution_weights(tokens, steps, mode="tokens_squared_per_step"):
    if not tokens or len(tokens) != len(steps):
        raise ValueError("token/step counts must have equal nonzero length")
    if any(type(t) is not int or t <= 0 for t in tokens) or any(type(s) is not int or s <= 0 for s in steps):
        raise ValueError("contribution counts must be positive integers")
    if mode == "tokens_squared_per_step":
        # Scaling before squaring avoids overflow for long-running counters.
        scale = max(tokens)
        scores = [(t / scale) ** 2 / s for t, s in zip(tokens, steps)]
    elif mode == "tokens":
        scores = [t / max(tokens) for t in tokens]
    else:
        raise ValueError("unknown contribution weighting")
    total = math.fsum(scores)
    return tuple(score / total for score in scores)


def merge_gradients(gradients, weights, mode="weighted_average"):
    gradients = iter(gradients)
    first = next(gradients, None)
    if first is None or not weights:
        raise ValueError("gradients/weights must have equal nonzero length")
    gradients = chain((first,), gradients)
    result = torch.zeros_like(first)
    if mode == "weighted_average":
        for gradient, weight in zip(gradients, weights, strict=True):
            result.add_(gradient, alpha=weight)
        return result
    if mode != "rda":
        raise ValueError("unknown merge mode")
    radius = torch.zeros((), dtype=result.dtype, device=result.device)
    for gradient, weight in zip(gradients, weights, strict=True):
        norm = gradient.norm()
        radius.add_(norm, alpha=weight)
        if bool(norm > 0):
            result.add_(gradient / norm, alpha=weight)
    direction_norm = result.norm()
    # Exactly opposing directions have no defined mean direction. Return zero
    # rather than amplifying numerical noise into a full-radius update.
    if bool(direction_norm > torch.finfo(result.dtype).eps):
        return result.mul_(radius / direction_norm)
    return torch.zeros_like(result)
