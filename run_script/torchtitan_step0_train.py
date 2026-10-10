"""TorchTitan wrapper that logs the first pre-update loss as step 0."""

import runpy

from torchtitan.experiments.torchft import trainer as ft_trainer


TrainerClass = getattr(
    ft_trainer,
    "FaultTolerantTrainer",
    getattr(ft_trainer, "Trainer", None),
)

if TrainerClass is None:
    raise RuntimeError("Could not locate TorchFT trainer class")


_original_train_step = TrainerClass.train_step


def _train_step_with_step0(self, *args, **kwargs):
    metrics = self.metrics_processor

    original_should_log = metrics.should_log
    original_log = metrics.log

    def should_log(step):
        return step == 1 or original_should_log(step)

    def log(step, *log_args, **log_kwargs):
        log_step = 0 if step == 1 else step
        return original_log(log_step, *log_args, **log_kwargs)

    metrics.should_log = should_log
    metrics.log = log

    try:
        return _original_train_step(self, *args, **kwargs)
    finally:
        metrics.should_log = original_should_log
        metrics.log = original_log


TrainerClass.train_step = _train_step_with_step0


if __name__ == "__main__":
    runpy.run_module("torchtitan.train", run_name="__main__")