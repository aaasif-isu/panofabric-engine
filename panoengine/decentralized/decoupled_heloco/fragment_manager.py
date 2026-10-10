"""Deterministic whole-tensor fragments in canonical parameter order.

The contiguous partition rule matches async_diloco._fragment_bounds without
importing its networking or torchft dependencies. Tensors are never split.
Layouts describe global shapes; FSDP transport requires a separate adapter.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    shape: tuple[int, ...]
    numel: int


@dataclass(frozen=True)
class Fragment:
    fragment_id: int
    parameter_names: tuple[str, ...]
    numel: int


class FragmentManager:
    """Own layout metadata only, with no retained model/tensor references."""

    def __init__(self, named_parameters: Iterable[tuple[str, torch.Tensor]], num_fragments: int):
        if type(num_fragments) is not int or num_fragments < 1:
            raise ValueError("num_fragments must be a positive integer")
        specs = []
        names = set()
        tensor_ids = set()
        tensor_refs = []  # Prevent ID reuse for generators yielding temporary tensors.
        for name, tensor in named_parameters:
            if not isinstance(name, str) or not name or name in names:
                raise ValueError(f"parameter names must be unique nonempty strings; got {name!r}")
            if id(tensor) in tensor_ids:
                raise ValueError("shared parameters must have one canonical name; use model.named_parameters()")
            names.add(name)
            tensor_ids.add(id(tensor))
            tensor_refs.append(tensor)
            specs.append(ParameterSpec(name, tuple(tensor.shape), tensor.numel()))
        if not specs:
            raise ValueError("cannot fragment an empty parameter collection")
        if num_fragments > len(specs):
            raise ValueError("num_fragments exceeds the number of parameter tensors")

        total = sum(spec.numel for spec in specs)
        boundaries = []
        start = accumulated = 0
        # Same cumulative-numel boundaries as the existing PanoFabric layout.
        for index, spec in enumerate(specs):
            accumulated += spec.numel
            remaining_fragments = num_fragments - len(boundaries) - 1
            if remaining_fragments == 0:
                continue
            remaining_parameters = len(specs) - index - 1
            if (
                accumulated * num_fragments >= (len(boundaries) + 1) * total
                or remaining_parameters == remaining_fragments
            ):
                boundaries.append((start, index + 1))
                start = index + 1
        boundaries.append((start, len(specs)))

        self.parameter_specs = tuple(specs)
        self.fragments = tuple(
            Fragment(i, tuple(s.name for s in specs[a:b]), sum(s.numel for s in specs[a:b]))
            for i, (a, b) in enumerate(boundaries)
        )
        self.parameter_to_fragment = MappingProxyType({
            name: fragment.fragment_id
            for fragment in self.fragments
            for name in fragment.parameter_names
        })
        self._specs = {spec.name: spec for spec in specs}
        # Dtype/device are excluded: global FP32 and local BF16 tensors can
        # legitimately share a layout, as can CPU and GPU copies.
        serialized = json.dumps(
            {"parameters": [(s.name, s.shape) for s in specs], "num_fragments": num_fragments},
            separators=(",", ":"),
        )
        self.layout_signature = hashlib.sha256(serialized.encode()).hexdigest()

    @classmethod
    def from_model(cls, model, num_fragments: int) -> FragmentManager:
        # PyTorch's default named_parameters() de-duplicates tied parameters.
        return cls(model.named_parameters(), num_fragments)

    @property
    def num_fragments(self) -> int:
        return len(self.fragments)

    @property
    def total_numel(self) -> int:
        return sum(fragment.numel for fragment in self.fragments)

    def fragment(self, fragment_id: int) -> Fragment:
        if type(fragment_id) is not int or not 0 <= fragment_id < self.num_fragments:
            raise ValueError(f"fragment_id must be in [0, {self.num_fragments})")
        return self.fragments[fragment_id]

    def select(self, fragment_id: int, parameters: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Select and check a fragment from either a fragment or full mapping."""
        fragment = self.fragment(fragment_id)
        selected = {}
        for name in fragment.parameter_names:
            if name not in parameters:
                raise ValueError(f"missing fragment parameter {name!r}")
            tensor = parameters[name]
            if tuple(tensor.shape) != self._specs[name].shape:
                raise ValueError(f"shape mismatch for {name!r}: expected {self._specs[name].shape}")
            selected[name] = tensor
        return selected

    def validate_update(self, fragment_id: int, parameters: Mapping[str, torch.Tensor]) -> None:
        if set(parameters) != set(self.fragment(fragment_id).parameter_names):
            raise ValueError("a fragment update must contain exactly that fragment's parameter names")
        self.select(fragment_id, parameters)

    def validate_model(self, parameters: Mapping[str, torch.Tensor]) -> None:
        if set(parameters) != set(self._specs):
            raise ValueError("model parameter names do not match the fragment layout")
        for fragment in self.fragments:
            self.select(fragment.fragment_id, parameters)
