"""Single-rank GPU learner entry point; network work stays off training thread."""

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace

import torch

from .config import load_config
from .fragment_manager import FragmentManager
from .learner import DecoupledLearner, SyncerClockComplete
from .process_smoke import _serve_learner, _wait_message
from .training_adapter import train_one_step
from .transport import FramedTransport


def initialize_fragments(transport, model, count, deadline):
    """Transfer one fragment at a time, keeping the existing 32 MiB cap.

    Nothing trains or constructs its baseline until every fragment is loaded.
    Initialization ACKs bound the in-flight storage to a single fragment.
    """
    manager = FragmentManager.from_model(model, count)
    live = dict(model.named_parameters())
    for fragment_id in range(count):
        message = _wait_message(transport, deadline)
        body = message["body"]
        if message["kind"] != "initialize_fragment" or body.get("fragment_id") != fragment_id:
            raise ValueError("initialization fragments must arrive in canonical order")
        if body.get("layout_signature") != manager.layout_signature:
            raise ValueError("global/learner initialization layouts differ")
        values = body["parameters"]
        manager.validate_update(fragment_id, values)
        for value in values.values():
            if value.dtype != torch.float32 or value.device.type != "cpu" or not torch.isfinite(value).all():
                raise ValueError("initialization requires finite CPU FP32 parameters")
        with torch.no_grad():
            for name, value in values.items():
                live[name].copy_(value)
        transport.send("initialize_ack", {"fragment_id": fragment_id})
    return manager


def run_session(transport, trainer, config, options, learner_id, log_folder, deadline):
    """Selected local/syncer budget, then idle boundaries for final drain/stop.

    Used by the GPU entry and CPU integration tests. The trainer owns the
    ordinary forward/backward/optimizer implementation and its data iterator.
    """
    model = trainer.model_parts[0]
    initialize_fragments(transport, model, config.decoupled.num_fragments, deadline)
    clock_target = config.decoupled.syncer_steps if config.decoupled.stopping == "syncer_steps" else None
    learner = DecoupledLearner(model, config.decoupled.num_fragments, learner_id=learner_id,
                              stop_at_syncer_step=clock_target,
                              max_snapshot_requests=config.decoupled.max_inflight_captures)
    transport.send("ready", {"pid": os.getpid(), "metadata": asdict(learner.metadata())})
    started, pause, parked, stop, finished, done = (threading.Event() for _ in range(6))
    failures = []
    service = threading.Thread(target=_serve_learner, args=(transport, learner, started, pause, parked, stop, finished, failures), kwargs={"training_done": done, "deadline": deadline}, daemon=True)
    service.start()
    # Preserve TorchTitan's metrics while recording every local training loss.
    loss_record = {}
    original_log = trainer.metrics_processor.log

    def record_loss(step, avg_loss, max_loss, grad_norm, **kwargs):
        loss_record.update(loss=float(avg_loss), grad_norm=float(grad_norm))
        return original_log(step, avg_loss, max_loss, grad_norm, **kwargs)

    trainer.metrics_processor.log = record_loss
    from .monitoring import Trajectory
    trajectory = Trajectory(Path(log_folder), config.monitoring, kind="learner")
    trajectory.clock_provider = lambda: learner.metadata().syncer_step
    iterator = trainer.batch_generator(trainer.dataloader)
    factors = options.island_slowness_factors
    factor = factors[0] if len(factors) == 1 else factors[learner_id]
    origin = time.monotonic()
    trajectory.save(0, dict(model.named_parameters()), force=True)
    try:
        with (Path(log_folder) / "steps.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("step", "elapsed_s", "step_s", "tokens", "total_tokens", "loss", "grad_norm", "applied_revisions", "syncer_step"))
            writer.writeheader()
            while not stop.is_set():
                if not started.is_set() or pause.is_set() or done.is_set():
                    learner.boundary()
                    if pause.is_set():
                        parked.set()
                    stop.wait(0.005)
                    continue
                learner.boundary()
                if clock_target is not None and learner.metadata().syncer_step >= clock_target:
                    done.set()
                    continue
                before = time.monotonic()
                if trainer.step == 0:
                    trajectory.origin = before
                trainer.step += 1
                loss_record.clear()
                try:
                    tokens = train_one_step(trainer, iterator, learner)
                except SyncerClockComplete:
                    # The clock may arrive after the previous boundary. The
                    # adapter rejects before consuming a batch/optimizer step.
                    trainer.step -= 1
                    done.set()
                    continue
                duration = time.monotonic() - before
                metadata = learner.metadata()
                if not loss_record:
                    raise RuntimeError("trainer did not report its per-step loss")
                writer.writerow({"step": trainer.step, "elapsed_s": time.monotonic() - origin, "step_s": duration, "tokens": tokens, "total_tokens": metadata.total_tokens, **loss_record, "applied_revisions": json.dumps([f.last_applied_revision for f in metadata.fragments]), "syncer_step": metadata.syncer_step})
                stream.flush()
                local_done = clock_target is None and trainer.step >= options.steps
                trajectory.save(trainer.step, dict(model.named_parameters()), tokens=metadata.total_tokens, force=local_done)
                if local_done or (clock_target is not None and metadata.syncer_step >= clock_target):
                    done.set()
                # Pacing is outside the optimizer step; the control thread and
                # socket threads continue to receive and queue requests.
                stop.wait(duration * (factor - 1))
    finally:
        trainer.metrics_processor.log = original_log
        learner.boundary()
        if not failures:
            trajectory.save(trainer.step, dict(model.named_parameters()), tokens=learner.metadata().total_tokens, force=True, final=True)
        finished.set()
        service.join(timeout=max(.001, deadline - time.monotonic()))
        (Path(log_folder) / "transport_diagnostics.json").write_text(json.dumps(transport.diagnostics(), indent=2))
    if service.is_alive():
        raise TimeoutError("learner control thread did not stop")
    if failures:
        raise RuntimeError(str(failures[0]))
    transport.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--learner-id", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    args = parser.parse_args(argv)
    trainer = transport = None
    try:
        from torchtitan.tools.logging import init_logger
        from torchtitan.trainer import Trainer
        from .gpu_recipe import build_recipe

        init_logger()
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("GPU learner must see exactly one CUDA device")
        config = load_config(args.run_dir / "experiment.yaml")
        options = SimpleNamespace(**json.loads((args.run_dir / "run-options.json").read_text()))
        folder = args.run_dir / f"learner_{args.learner_id}"
        cfg = build_recipe(options, Path.cwd(), folder, learner_id=args.learner_id, experiment_config=config)
        trainer = Trainer(config=cfg)
        if len(trainer.model_parts) != 1 or any(p.device.type != "cuda" or hasattr(p, "to_local") for p in trainer.model_parts[0].parameters()):
            raise RuntimeError("GPU path requires one dense CUDA model part")
        transport = FramedTransport(socket.create_connection(("127.0.0.1", args.port), timeout=10.0))
        transport.send("hello", {"learner_id": args.learner_id, "token": os.environ["DECOUPLED_RUN_TOKEN"]})
        run_session(transport, trainer, config, options, args.learner_id, folder, time.monotonic() + args.timeout)
    except BaseException as exc:
        if transport is not None:
            try:
                transport.send("fatal", {"error": str(exc)})
                transport.flush(timeout=0.5)
            except Exception:
                pass
        raise
    finally:
        if transport is not None:
            transport.close()
        if trainer is not None:
            trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
