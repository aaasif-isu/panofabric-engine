"""Explicit scope of the matched, whole-model asynchronous baselines."""

import math

from .config import ConfigError, LEGACY_METHODS

BASELINE_SCOPE = "matched_dense_single_gpu_localhost"
BASELINE_STOPPING = "fixed_local_budgets_all_whole_model_windows_committed_final_pull"
BASELINE_FORMAT = "matched_global_v1"


def validate_baseline_options(config, options):
    if config.method not in LEGACY_METHODS:
        raise ConfigError("matched baseline launcher requires heloco, diloco, or mla")
    config.validate_run(options)
    if options.coordination_method != "async" or options.num_fragments != 1:
        raise ConfigError("matched baselines require async coordination and run.num_fragments: 1")
    if options.async_interval != 1 or options.max_wait_time != 0:
        raise ConfigError("matched baselines require async_interval: 1 and max_wait_time: 0")
    if options.correction_workers != "all" or options.correction_scope != "tensorwise" or options.other_islands_method != "heloco_uncorrected":
        raise ConfigError("matched HeLoCo requires all workers and tensorwise correction")
    if any(getattr(options, name) is not None for name in ("diloco_lr", "diloco_momentum", "diloco_nesterov_period")):
        raise ConfigError("hybrid HeLoCo/DiLoCo controls are outside this comparison")
    if config.method != "heloco" and options.rho is not None:
        raise ConfigError("run.rho applies only to the HeLoCo baseline")
    if options.tokens_per_parameter is None and options.steps % options.sync_steps:
        raise ConfigError("matched baseline run.steps must be divisible by run.sync_steps; partial windows would leave unmerged work")


def baseline_metadata(config, options):
    heloco = config.method == "heloco"
    diloco = config.method == "diloco"
    # Match run_heloco.ps_cmd, including its six-decimal rho argument.
    rho = float(f"{options.rho if options.rho is not None else 1 / math.sqrt(options.islands):.6f}") if heloco else None
    return {"name": "delayed_nesterov" if diloco else config.method,
            "lr": options.outer_lr, "momentum": options.outer_momentum,
            "momentum_convention": "sum_of_period_mean_gradients" if diloco else "ema",
            "nesterov_period": options.islands if diloco else None,
            "rho": rho, "correction_enabled": heloco, "lookahead": heloco,
            "merge": "one_step_per_arrival", "window_steps": options.sync_steps,
            "num_fragments": 1, "wire_dtype": "float32", "dylu_enabled": False}
