"""Configuration for the separate Decoupled HeLoCo experiment launcher."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any


LEGACY_METHODS = ("heloco", "diloco", "mla")
DECOUPLED_METHODS = ("decoupled_heloco", "decoupled_diloco")
METHODS = (*LEGACY_METHODS, *DECOUPLED_METHODS)
RESERVED_RUN_KEYS = {"methods", "outer_method", "config_file", "dry_run"}


class ConfigError(ValueError):
    """An invalid experiment setting, before any roles are launched."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ConfigError(f"{name} must be a mapping with string keys")
    return value


def _positive_int(value: Any, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ConfigError(f"{name} must be a positive integer")


def _finite_number(value: Any, name: str, *, minimum: float = 0.0) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
    ):
        raise ConfigError(f"{name} must be finite and >= {minimum}")


@dataclass(frozen=True)
class HeLoCoConfig:
    """Correction applied to one merged quorum gradient, not each arrival."""

    rho: float = 1.0
    correction_enabled: bool = True
    lookahead: bool = True
    c_ok: float = 0.2
    k_s: float = 0.5
    k_d: float = 1.0
    kappa: float = 3.0
    beta_max: float = 0.5
    eps: float = 1e-8

    def validate(self) -> None:
        for name in ("rho", "c_ok", "k_s", "k_d", "kappa", "beta_max", "eps"):
            _finite_number(getattr(self, name), f"decoupled.heloco.{name}")
        if self.c_ok > 1 or self.beta_max > 1:
            raise ConfigError("decoupled.heloco.c_ok and beta_max must be in [0, 1]")
        if self.eps == 0:
            raise ConfigError("decoupled.heloco.eps must be > 0")
        for name in ("correction_enabled", "lookahead"):
            if type(getattr(self, name)) is not bool:
                raise ConfigError(f"decoupled.heloco.{name} must be true or false")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> HeLoCoConfig:
        unknown = values.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise ConfigError(f"decoupled.heloco: unknown option(s) {sorted(unknown)}")
        result = cls(**values)
        result.validate()
        return result


@dataclass(frozen=True)
class DecoupledConfig:
    """Settings reserved for the new fragment lifecycle, independent of H."""

    num_fragments: int = 4
    min_quorum: int = 2
    overlap_steps: int = 5
    sync_interval: float = 1.0
    grace_window_factor: float = 0.8
    scheduler: str = "round_robin"
    merge: str = "weighted_average"
    heloco: HeLoCoConfig = field(default_factory=HeLoCoConfig)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> DecoupledConfig:
        unknown = values.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise ConfigError(f"decoupled: unknown option(s) {sorted(unknown)}")
        values = dict(values)
        heloco = HeLoCoConfig.from_mapping(_mapping(values.pop("heloco", {}), "decoupled.heloco"))
        result = cls(**values, heloco=heloco)
        for name in ("num_fragments", "min_quorum", "overlap_steps"):
            _positive_int(getattr(result, name), f"decoupled.{name}")
        _finite_number(result.sync_interval, "decoupled.sync_interval")
        if result.sync_interval == 0:
            raise ConfigError("decoupled.sync_interval must be > 0")
        _finite_number(result.grace_window_factor, "decoupled.grace_window_factor")
        if result.grace_window_factor > 1:
            raise ConfigError("decoupled.grace_window_factor must be in [0, 1]")
        if result.scheduler != "round_robin":
            raise ConfigError("only decoupled.scheduler: round_robin is planned for v1")
        if result.merge != "weighted_average":
            raise ConfigError("only decoupled.merge: weighted_average is planned for v1")
        return result


@dataclass(frozen=True)
class ExperimentConfig:
    method: str
    run: dict[str, Any] = field(default_factory=dict)
    decoupled: DecoupledConfig = field(default_factory=DecoupledConfig)

    @property
    def fragment_outer_method(self) -> str:
        if self.method not in DECOUPLED_METHODS:
            raise ConfigError("fragment outer optimizer requires a decoupled method")
        return self.method.removeprefix("decoupled_")

    def legacy_options(self) -> dict[str, Any]:
        """Use the old parser for run keys; never pass new sections to it.

        For decoupled mode, the corresponding legacy parser validates common settings.
        This mapping must never be used to execute decoupled training.
        """
        method = self.method if self.method in LEGACY_METHODS else self.fragment_outer_method
        return {**self.run, "methods": [method], "outer_method": method}

    def validate_run(self, args: Any) -> None:
        """Validate common settings after resolving the existing CLI defaults."""
        for name in ("islands", "gpus_per_island", "steps", "seq_len", "batch", "sync_steps", "num_fragments"):
            _positive_int(getattr(args, name), f"run.{name}")
        _finite_number(args.outer_lr, "run.outer_lr")
        _finite_number(args.outer_momentum, "run.outer_momentum")
        if args.outer_momentum >= 1:
            raise ConfigError("run.outer_momentum must be in [0, 1)")
        if args.tokens_per_parameter is not None:
            _finite_number(args.tokens_per_parameter, "run.tokens_per_parameter")
            if args.tokens_per_parameter == 0:
                raise ConfigError("run.tokens_per_parameter must be > 0 or null")
        if args.rho is not None:
            _finite_number(args.rho, "run.rho")
        for name, choices in (
            ("coordination_method", ("sync", "async")),
            ("data_distribution", ("iid", "non_iid", "both")),
            ("correction_scope", ("tensorwise", "whole_gradient")),
            ("other_islands_method", ("heloco_uncorrected", "diloco")),
        ):
            if getattr(args, name) not in choices:
                raise ConfigError(f"run.{name} must be one of {choices}")
        for name in ("should_quantize", "correction_heatmap"):
            if type(getattr(args, name)) is not bool:
                raise ConfigError(f"run.{name} must be true or false")
        if not isinstance(args.extra, list) or any(not isinstance(x, str) for x in args.extra):
            raise ConfigError("run.extra must contain strings")
        for factor in args.island_slowness_factors:
            _finite_number(factor, "run.island_slowness_factors")
            if factor == 0:
                raise ConfigError("run.island_slowness_factors must be > 0")
        if args.gpus is not None:
            try:
                gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
            except (AttributeError, ValueError) as exc:
                raise ConfigError("run.gpus must be a list or comma-separated GPU IDs") from exc
            if len(gpus) < args.islands * args.gpus_per_island:
                raise ConfigError("run.gpus lists fewer GPUs than the configured topology needs")
            if min(gpus) < 0 or len(set(gpus)) != len(gpus):
                raise ConfigError("run.gpus must contain distinct nonnegative GPU IDs")
        if self.method in DECOUPLED_METHODS:
            if self.decoupled.min_quorum > args.islands:
                raise ConfigError("decoupled.min_quorum cannot exceed run.islands")
            if args.coordination_method != "async":
                raise ConfigError(f"{self.method} requires run.coordination_method: async")
            if args.num_fragments != 1:
                raise ConfigError("use decoupled.num_fragments for the new mode; keep run.num_fragments: 1")
            if args.rho is not None:
                raise ConfigError("run.rho is a legacy arrival weight; use decoupled.heloco.rho for a merged quorum")
            if args.correction_workers != "all" or args.correction_scope != "tensorwise":
                raise ConfigError("decoupled correction is tensorwise after merging; use decoupled.heloco.correction_enabled to disable it")
        else:
            if args.sync_steps % args.num_fragments:
                raise ConfigError("run.sync_steps must be divisible by run.num_fragments for legacy methods")
            if args.num_fragments > 1 and self.method != "heloco":
                raise ConfigError("legacy fragmentation requires method: heloco")
            if args.coordination_method == "sync" and args.num_fragments != 1:
                raise ConfigError("legacy synchronous coordination requires run.num_fragments: 1")


def load_config(path: Path, method_override: str | None = None) -> ExperimentConfig:
    import yaml

    with path.open(encoding="utf-8") as stream:
        data = _mapping(yaml.safe_load(stream), "configuration")
    unknown = data.keys() - {"method", "run", "decoupled"}
    if unknown:
        raise ConfigError(f"configuration: unknown section(s) {sorted(unknown)}")
    method = method_override if method_override is not None else data.get("method")
    if method not in METHODS:
        raise ConfigError(f"method must be one of {METHODS}; got {method!r}")
    raw_run = _mapping(data.get("run", {}), "run")
    run = {key.replace("-", "_"): value for key, value in raw_run.items()}
    if len(run) != len(raw_run):
        raise ConfigError("run contains duplicate options with hyphen/underscore spellings")
    reserved = run.keys() & RESERVED_RUN_KEYS
    if reserved:
        raise ConfigError(f"run: reserved option(s) {sorted(reserved)}; select the method at the top level")
    decoupled = DecoupledConfig.from_mapping(_mapping(data.get("decoupled", {}), "decoupled"))
    return ExperimentConfig(method=method, run=run, decoupled=decoupled)
