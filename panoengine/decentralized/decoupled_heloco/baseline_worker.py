"""Fixed-budget learner using the original whole-model AsyncDiLoCo client."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from .baseline_training import write_json
from .config import load_config
from .evaluation import parameter_fingerprint
from .training_adapter import _CountedBatches


def wait_for_gate(path, deadline):
    while time.monotonic() < deadline:
        if Path(path).is_file():
            return json.loads(Path(path).read_text())
        time.sleep(0.01)
    raise TimeoutError(f"learner deadline exceeded waiting for {Path(path).name}")


def inner_optimizer(trainer):
    # Pinned TorchTitan has one plain optimizer inside its container. The
    # CPU process fixture exposes that same optimizer directly.
    container = getattr(trainer, "optimizers", None)
    optimizers = getattr(container, "optimizers", None)
    if optimizers is not None:
        if len(optimizers) != 1:
            raise RuntimeError("matched baseline requires exactly one inner optimizer")
        return optimizers[0]
    optimizer = getattr(trainer, "optimizer", None)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise RuntimeError("trainer has no supported plain inner optimizer")
    return optimizer


def run_baseline_session(trainer, config, options, learner_id, run_dir, deadline):
    from panoengine.decentralized.async_diloco import AsyncDiLoCo

    run_dir = Path(run_dir)
    folder = run_dir / f"learner_{learner_id}"
    endpoint = json.loads((run_dir / "server.json").read_text())
    if len(trainer.model_parts) != 1:
        raise RuntimeError("matched baseline requires one model part")
    model = trainer.model_parts[0]
    if any(p.is_meta or p.dtype != torch.float32 or hasattr(p, "to_local") for p in model.parameters()):
        raise RuntimeError("matched baseline requires dense materialized FP32 parameters")
    optimizer = inner_optimizer(trainer)
    client = AsyncDiLoCo(endpoint["address"], model, optimizer, sync_every=options.sync_steps,
                         heartbeat_address=endpoint["heartbeat_address"], heartbeat_interval=1.0,
                         backup_device="cpu", reset_inner_state=False, should_quantize=False,
                         wire_bf16=False, num_fragments=1, min_replicas=0,
                         sync_timeout=min(options.ps_timeout, max(0.1, deadline - time.monotonic())))
    iterator = trainer.batch_generator(trainer.dataloader)
    factors = options.island_slowness_factors
    factor = factors[0] if len(factors) == 1 else factors[learner_id]
    record = {}
    original_log = trainer.metrics_processor.log

    def log_loss(step, avg_loss, max_loss, grad_norm, **kwargs):
        record.update(loss=float(avg_loss), grad_norm=float(grad_norm))
        return original_log(step, avg_loss, max_loss, grad_norm, **kwargs)

    trainer.metrics_processor.log = log_loss
    total_tokens, window_tokens, pushes = 0, 0, 0
    try:
        with client:
            # Call the original _sync explicitly at the same optimizer
            # boundaries. Its normal hook catches transport failures and
            # drops windows; a comparison must fail rather than hide those.
            for handle in client._hooks:
                handle.remove()
            client._hooks.clear()
            initial_hash = parameter_fingerprint(dict(model.named_parameters()))
            if client._baseline_revision != 0 or initial_hash != endpoint["initial_parameters_sha256"]:
                raise RuntimeError("learner initial pull differs from the shared revision-0 model")
            write_json(folder / "ready.json", {"learner_id": learner_id, "pid": os.getpid(),
                                                "revision": 0, "initial_parameters_sha256": initial_hash})
            wait_for_gate(run_dir / "start.json", deadline)
            client._window_start = time.monotonic()
            origin = time.monotonic()
            with (folder / "steps.csv").open("w", newline="") as stream, (folder / "windows.csv").open("w", newline="") as windows:
                steps = csv.DictWriter(stream, fieldnames=("step", "elapsed_s", "step_s", "tokens", "total_tokens", "loss", "grad_norm", "applied_revisions"))
                commits = csv.DictWriter(windows, fieldnames=("push", "local_step", "window_steps", "window_tokens", "base_revision", "received_revision", "exchange_s"))
                steps.writeheader()
                commits.writeheader()
                while trainer.step < options.steps:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("baseline learner exceeded its training deadline")
                    before = time.monotonic()
                    trainer.step += 1
                    record.clear()
                    batch = _CountedBatches(iterator)
                    trainer.train_step(batch)
                    duration = time.monotonic() - before
                    if batch.tokens < 1 or not record or any(not math.isfinite(v) for v in record.values()):
                        raise RuntimeError("trainer did not report finite loss/grad_norm and positive valid tokens")
                    total_tokens += batch.tokens
                    window_tokens += batch.tokens
                    client._local_step += 1
                    if client._local_step == options.sync_steps:
                        if any(not torch.isfinite(p).all() for p in model.parameters()):
                            raise RuntimeError("local model became nonfinite before its push")
                        base_revision = client._baseline_revision
                        exchange_start = time.monotonic()
                        client._sync()
                        if client._skip_speed_report or client._sync_every != options.sync_steps or client._baseline_revision <= base_revision:
                            raise RuntimeError("original server rejected the local window or changed its length")
                        if any(not torch.isfinite(p).all() for p in model.parameters()):
                            raise RuntimeError("server returned nonfinite parameters")
                        pushes += 1
                        commits.writerow({"push": pushes, "local_step": trainer.step, "window_steps": client._local_step,
                                          "window_tokens": window_tokens, "base_revision": base_revision,
                                          "received_revision": client._baseline_revision, "exchange_s": time.monotonic() - exchange_start})
                        windows.flush()
                        client._local_step = 0
                        window_tokens = 0
                        client._window_start = time.monotonic()
                    steps.writerow({"step": trainer.step, "elapsed_s": time.monotonic() - origin,
                                    "step_s": duration, "tokens": batch.tokens, "total_tokens": total_tokens,
                                    **record, "applied_revisions": json.dumps([client._baseline_revision])})
                    stream.flush()
                    # Original-client pacing is explicitly disabled by the
                    # coordinator. Delay compute time only, once per step.
                    delay = duration * (factor - 1)
                    if time.monotonic() + delay >= deadline:
                        raise TimeoutError("baseline pacing would exceed the training deadline")
                    time.sleep(delay)
            if client._local_step or window_tokens:
                raise RuntimeError("partial local window remains; the budget must be divisible by H")
            write_json(folder / "training-done.json", {"local_steps": trainer.step, "pushes": pushes,
                                                        "remaining_steps": client._local_step, "total_tokens": total_tokens})
            final = wait_for_gate(run_dir / "finalize.json", deadline)
            client._pull_global()  # Pull-only: never adds an outer update.
            final_hash = parameter_fingerprint(dict(model.named_parameters()))
            if client._baseline_revision != final["revision"] or final_hash != final["dispatch_sha256"]:
                raise RuntimeError("final pull did not adopt the frozen server dispatch")
            result = {"learner_id": learner_id, "pid": os.getpid(), "total_local_steps": trainer.step,
                      "total_tokens": total_tokens, "pushes": pushes, "final_dispatch_sha256": final_hash,
                      "protocol_sent_bytes": client._comm_bytes_up_total,
                      "protocol_received_bytes": client._comm_bytes_down_total,
                      "communication_s": client._comm_seconds_total,
                      "fragments": [{"fragment_id": 0, "local_steps": 0, "tokens": 0,
                                     "last_received_revision": final["revision"], "last_applied_revision": final["revision"]}]}
        # Publish only after the original context announces clean completion.
        write_json(folder / "result.json", result)
        return result
    finally:
        trainer.metrics_processor.log = original_log


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--learner-id", type=int, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    args = parser.parse_args(argv)
    trainer = None
    try:
        from torchtitan.tools.logging import init_logger
        from torchtitan.trainer import Trainer
        from .gpu_recipe import build_recipe

        init_logger()
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("baseline GPU learner must see exactly one CUDA device")
        config = load_config(args.run_dir / "experiment.yaml")
        options = SimpleNamespace(**json.loads((args.run_dir / "run-options.json").read_text()))
        recipe = build_recipe(options, Path.cwd(), args.run_dir / f"learner_{args.learner_id}", learner_id=args.learner_id)
        trainer = Trainer(config=recipe)
        if any(p.device.type != "cuda" for part in trainer.model_parts for p in part.parameters()):
            raise RuntimeError("baseline GPU model parameters must be on CUDA")
        run_baseline_session(trainer, config, options, args.learner_id, args.run_dir, time.monotonic() + args.timeout)
    finally:
        if trainer is not None:
            trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
