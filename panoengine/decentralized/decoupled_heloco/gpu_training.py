"""Localhost GPU training coordinator for dense 15M Llama learners."""

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
import math
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time

import torch

from .config import ConfigError
from .fragment_manager import FragmentManager
from .gpu_recipe import build_initial_model, build_recipe, validate_tokenizer, validate_training_options
from .process_smoke import _wait_message
from .state import _cpu_copy
from .syncer import DecoupledSyncer
from .timing import TimedSyncController
from .transport import FramedTransport, MAX_FRAME_BYTES, RemoteLearner, encode_message, metadata_from_wire

STOPPING_POLICY = "fixed_local_budgets_drain_to_first_unavailable_scheduled_quorum"


def outer_optimizer_metadata(config, options):
    diloco = config.fragment_outer_method == "diloco"
    return {"name": "nesterov_sgd" if diloco else "heloco",
            "lr": options.outer_lr, "momentum": options.outer_momentum,
            "momentum_convention": "sum" if diloco else "ema",
            "correction_enabled": False if diloco else config.decoupled.heloco.correction_enabled,
            "lookahead": False if diloco else config.decoupled.heloco.lookahead,
            "merge": config.decoupled.merge, "weighting": config.decoupled.weighting,
            "correction_order": config.decoupled.heloco.correction_order if not diloco else None}


def select_devices(options, visible_devices, device_count):
    """Interpret run.gpus as logical indices within the current allocation."""
    indices = list(range(options.islands)) if options.gpus is None else [int(x) for x in options.gpus.split(",") if x.strip()][:options.islands]
    if len(indices) != options.islands or len(set(indices)) != len(indices) or any(i < 0 or i >= device_count for i in indices):
        raise ConfigError(f"requested GPU indices {indices}; only {device_count} CUDA devices are visible. Edit run.islands/run.gpus to match your allocation")
    if visible_devices is None:
        return [str(i) for i in indices]
    visible = [value.strip() for value in visible_devices.split(",") if value.strip()]
    if len(visible) != device_count:
        raise ConfigError("CUDA_VISIBLE_DEVICES does not match the device count; use a flat GPU allocation for Phase 7")
    return [visible[i] for i in indices]


def validate_frame_sizes(model, count):
    manager = FragmentManager.from_model(model, count)
    parameters = dict(model.named_parameters())
    for fragment in manager.fragments:
        # Snapshot contains two independent FP32 copies. Leave space for tensor
        # names/storage metadata, then verify the actual serialized payload.
        if fragment.numel * 8 + 65536 + len(fragment.parameter_names) * 1024 > MAX_FRAME_BYTES:
            raise ConfigError("a fragment snapshot exceeds 32 MiB; increase decoupled.num_fragments (large-tensor chunking is pending)")
        baseline = _cpu_copy(manager.select(fragment.fragment_id, parameters))
        current = _cpu_copy(baseline)
        encode_message("snapshot", {"request_id": "0", "snapshot": {"fragment_id": fragment.fragment_id, "layout_signature": manager.layout_signature, "base_revision": 0, "local_steps": 1, "tokens": 1, "baseline": baseline, "current": current}})
    return manager


def preflight(config, options, repo_root):
    """Check CUDA allocation, installed recipe APIs, tokenizer and wire layout.

    It launches no learners and takes no optimizer steps. Dataset download,
    iteration, CUDA forward/backward and real timing are checked by the run.
    """
    validate_training_options(config, options)
    if not torch.cuda.is_available():
        raise ConfigError("CUDA is unavailable; run --check-training inside your allocated GPU job using the project's heloco environment")
    devices = select_devices(options, os.environ.get("CUDA_VISIBLE_DEVICES"), torch.cuda.device_count())
    assets = (Path(repo_root) / options.hf_assets).resolve()
    if not assets.is_dir():
        raise ConfigError(f"tokenizer directory does not exist: {assets}; set run.hf_assets to your existing debug tokenizer")
    cfg = build_recipe(options, repo_root, Path(repo_root) / options.log_dir, experiment_config=config)
    tokenizer = cfg.tokenizer.build(tokenizer_path=cfg.hf_assets_path)
    vocab_size, max_token_id = validate_tokenizer(tokenizer, cfg.model_spec.model.vocab_size)
    print(f"Tokenizer compatible: vocabulary={vocab_size}, max_token_id={max_token_id}, model_vocab_size={cfg.model_spec.model.vocab_size}")
    model = build_initial_model(cfg, options.seed)
    manager = validate_frame_sizes(model, config.decoupled.num_fragments)
    if options.tokens_per_parameter is not None:
        options.steps = math.ceil(options.tokens_per_parameter * manager.total_numel / (options.islands * options.batch * options.seq_len))
    if config.decoupled.stopping == "local_steps" and options.steps < config.decoupled.capture_min_steps:
        raise ConfigError("the resolved training budget is smaller than min_local_steps; increase run.steps or the token budget")
    print(f"GPU preflight passed: learners={options.islands}, parameters={manager.total_numel:,}, local_steps_setting={options.steps}, device_masks={devices}")
    print(f"CPU syncer replicas: {config.decoupled.syncer_shards}; same-host IPC fragment all-reduce; timeout={config.decoupled.syncer_timeout}s")
    print("Recipe: dense FP32 Llama 15M, FlexAttention, plain AdamW; IID data sharded across learners.")
    budget = (f"syncer clock t={config.decoupled.syncer_steps}; local run.steps is not a stopping budget"
              if config.decoupled.stopping == "syncer_steps" else "fixed local budgets and quorum-preserving drain")
    print(f"Outer optimizer: {config.fragment_outer_method}; stopping: {budget}.")
    if config.decoupled.stopping == "syncer_steps":
        print(f"Clock-mode inner LR schedule: {config.decoupled.clock_lr_schedule} (local_horizon uses run.steps).")
    print("Dataset iteration and CUDA forward/backward will be checked during training.")
    return model, devices


def delivered(syncer, peers):
    revisions = syncer.fragment_revisions
    syncer.retry_broadcast()
    return all(f.last_applied_revision == revisions[f.fragment_id] for peer in peers if peer.available for f in peer.metadata().fragments)


def drain_complete(syncer, peers):
    """Stop at the first unavailable round-robin quorum after budgets end."""
    if not all(peer.training_done for peer in peers if peer.available) or syncer.has_active_sync or not delivered(syncer, peers):
        return False
    fragment_id = syncer.scheduler.fragment_id
    return syncer.scheduler.plan(syncer.learner_metadata(), fragment_revision=syncer.fragment_revisions[fragment_id]) is None


def _pump_until(peers, processes, deadline, predicate, *, min_quorum=None):
    while time.monotonic() < deadline:
        for peer in peers:
            peer.pump()
        for index, process in enumerate(processes):
            if process.poll() is not None and (index >= len(peers) or not peers[index].stopped):
                if min_quorum is None or index >= len(peers):
                    raise RuntimeError(f"learner {index} exited early; inspect learner_{index}/trainer.log")
                peers[index].mark_unavailable(f"learner {index} exited early; inspect learner_{index}/trainer.log")
        if min_quorum is not None and sum(peer.available for peer in peers) < min_quorum:
            raise RuntimeError(f"available learner count is below min_quorum={min_quorum}; cannot continue safely")
        if predicate():
            return
        time.sleep(0.005)
    raise TimeoutError("GPU run exceeded --training-timeout; inspect learner logs or increase the deadline")


def _send_available(peers, kind):
    """Isolate disconnects that race with start/pause/stop control messages."""
    for peer in peers:
        if not peer.available:
            continue
        try:
            peer.transport.send_reliable(kind, {}, deadline=peer.deadline, progress=lambda: [p.pump() for p in peers])
        except (ConnectionError, OSError) as exc:
            peer.mark_unavailable(exc)


def run_gpu_training(config, options, repo_root, *, timeout=1800.0, check_only=False, output_dir=None):
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ConfigError("--training-timeout must be finite and positive")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, devices = preflight(config, options, repo_root)
        if check_only:
            return 0
        if output_dir is None:
            parent = Path(repo_root) / options.log_dir
            parent.mkdir(parents=True, exist_ok=True)
            folder = parent / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(3))
        else:
            folder = (Path(repo_root) / output_dir).resolve()
        folder.mkdir(parents=True, exist_ok=False)
        return _run_training(config, options, model, devices, folder, timeout, repo_root)
    finally:
        torch.set_num_threads(old_threads)


def _run_training(config, options, model, devices, folder, timeout, repo_root, *, worker_command=None):
    """Private coordinator also exercised with CPU trainers in integration tests."""
    import yaml

    folder = Path(folder)
    (folder / "experiment.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False))
    (folder / "run-options.json").write_text(json.dumps(vars(options), indent=2))
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(options.islands)
    listener.settimeout(0.1)
    token = secrets.token_hex(32)
    deadline = time.monotonic() + timeout
    startup_deadline = min(deadline, time.monotonic() + options.ps_timeout)
    transports, processes, peers, logs = [], [], [], []
    from .evaluation import parameter_fingerprint

    manifest = {"status": "starting", "scope": "dense_single_gpu_localhost", "devices": devices, "steps_per_learner": options.steps if config.decoupled.stopping == "local_steps" else None, "checkpoint_resumable": False,
                "method": config.method, "outer_optimizer": outer_optimizer_metadata(config, options),
                "stopping_policy": STOPPING_POLICY if config.decoupled.stopping == "local_steps" else "shared_syncer_clock_with_final_revision_drain",
                "stopping_mode": config.decoupled.stopping, "syncer_step_target": config.decoupled.syncer_steps,
                "scheduler": config.decoupled.scheduler, "max_inflight_captures": config.decoupled.max_inflight_captures,
                "syncer_shards": config.decoupled.syncer_shards,
                "clock_lr_schedule": config.decoupled.clock_lr_schedule if config.decoupled.stopping == "syncer_steps" else None,
                "initial_parameters_sha256": parameter_fingerprint(dict(model.named_parameters()))}
    syncer = None
    monitor = None
    try:
        manager = validate_frame_sizes(model, config.decoupled.num_fragments)
        initial = _cpu_copy(dict(model.named_parameters()))
        used_ports = set()
        for learner_id, device in enumerate(devices):
            child_folder = folder / f"learner_{learner_id}"
            child_folder.mkdir()
            log = (child_folder / "trainer.log").open("w", encoding="utf-8")
            logs.append(log)
            # Reserve a distinct rendezvous port for each independent rank-0
            # process group. The inter-learner protocol uses the listener above.
            reservation = socket.socket()
            reservation.bind(("127.0.0.1", 0))
            rank_port = reservation.getsockname()[1]
            while rank_port in used_ports:
                reservation.close()
                reservation = socket.socket()
                reservation.bind(("127.0.0.1", 0))
                rank_port = reservation.getsockname()[1]
            used_ports.add(rank_port)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", LOCAL_WORLD_SIZE="1", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(rank_port), DECOUPLED_RUN_TOKEN=token, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
            for name in ("DILOCO_SERVER_ADDR", "DILOCO_HB_ADDR", "TORCHFT_LIGHTHOUSE"):
                env.pop(name, None)
            command = [*(worker_command or [sys.executable, "-m", "panoengine.decentralized.decoupled_heloco.gpu_worker"]), "--run-dir", str(folder), "--learner-id", str(learner_id), "--port", str(listener.getsockname()[1]), "--timeout", str(timeout)]
            reservation.close()
            processes.append(subprocess.Popen(command, cwd=repo_root, env=env, stdout=log, stderr=subprocess.STDOUT))
        by_id = {}
        while len(by_id) < options.islands:
            if time.monotonic() >= startup_deadline:
                raise TimeoutError("GPU learner startup timed out; inspect learner_*/trainer.log")
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("a GPU learner exited during startup; inspect learner_*/trainer.log")
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            transport = FramedTransport(connection)
            transports.append(transport)
            hello = _wait_message(transport, startup_deadline)
            body = hello["body"]
            learner_id = body.get("learner_id")
            if hello["kind"] != "hello" or not secrets.compare_digest(body.get("token", ""), token):
                raise ValueError("learner handshake token mismatch")
            if type(learner_id) is not int or not 0 <= learner_id < options.islands or learner_id in by_id:
                raise ValueError("invalid or duplicate learner identity")
            for fragment in manager.fragments:
                transport.send("initialize_fragment", {"fragment_id": fragment.fragment_id, "layout_signature": manager.layout_signature, "parameters": manager.select(fragment.fragment_id, initial)})
                ack = _wait_message(transport, startup_deadline)
                if ack["kind"] != "initialize_ack" or ack["body"].get("fragment_id") != fragment.fragment_id:
                    raise ValueError("initialization acknowledgement mismatch")
            ready = _wait_message(transport, startup_deadline)
            if ready["kind"] != "ready":
                raise ValueError("expected learner readiness acknowledgement")
            peer = RemoteLearner(transport, metadata_from_wire(ready["body"]["metadata"]))
            peer.pid = ready["body"]["pid"]
            if peer.metadata().learner_id != learner_id or peer.pid != processes[learner_id].pid:
                raise ValueError("learner handshake identity changed")
            by_id[learner_id] = peer
        del initial
        peers = [by_id[i] for i in range(options.islands)]
        for peer in peers:
            peer.deadline = deadline
        d = config.decoupled
        syncer = DecoupledSyncer(model, peers, d.num_fragments, min_quorum=d.min_quorum, overlap_steps=d.overlap_steps, outer_lr=options.outer_lr, outer_method=config.fragment_outer_method, outer_momentum=options.outer_momentum, heloco=d.heloco, weighting=d.weighting, merge=d.merge, scheduler=d.scheduler, sync_period=d.sync_period, fragment_offsets=d.fragment_offsets, min_local_steps=d.min_local_steps, max_inflight_captures=d.max_inflight_captures, syncer_shards=d.syncer_shards, syncer_timeout=d.syncer_timeout)
        del model
        from .monitoring import CentralMonitor
        monitor = CentralMonitor(folder, config.monitoring, replica_pids=syncer.optimizer.pids if d.syncer_shards > 1 else (), replica_resources=getattr(syncer.optimizer, "resource_paths", ()))
        manifest["syncer_replica_pids"] = list(getattr(syncer.optimizer, "pids", ()))
        syncer.monitor = monitor
        monitor.trajectory.clock_provider = lambda: syncer.global_step
        monitor.trajectory.save_from(0, syncer.optimizer.model_snapshot, force=True)
        controller = TimedSyncController(syncer, sync_interval=d.sync_interval, grace_window_factor=d.grace_window_factor, adaptive_grace=d.adaptive_grace, ema_alpha=d.timing_ema_alpha, max_global_step=d.syncer_steps if d.stopping == "syncer_steps" else None)
        print(f"GPU training started: syncer_pid={os.getpid()}, learner_pids={[p.pid for p in peers]}", flush=True)
        print(f"Logs: {folder}", flush=True)
        started = time.monotonic()
        monitor.trajectory.origin = started
        _send_available(peers, "start")
        with (folder / "syncs.csv").open("w", newline="", encoding="utf-8") as stream, (folder / "capture_events.csv").open("w", newline="", encoding="utf-8") as capture_stream:
            writer = csv.DictWriter(stream, fieldnames=("elapsed_s", "sync_step", "global_step", "fragment_id", "fragment_revision", "learner_ids", "local_steps", "tokens", "weights"))
            writer.writeheader()
            capture_writer = csv.DictWriter(capture_stream, fieldnames=("event", "elapsed_s", "sync_step", "global_step", "fragment_id", "inflight_captures"))
            capture_writer.writeheader()
            def record_capture(event):
                capture_writer.writerow({**event, "elapsed_s": event["elapsed_s"] - started})
                capture_stream.flush()
            controller.event_sink = record_capture

            terminal_sent = set()

            def synchronize():
                clock_mode = d.stopping == "syncer_steps"
                if clock_mode and syncer.scheduler.global_step > d.syncer_steps:
                    # No captures beyond T. Apply all committed revisions before
                    # announcing an idle terminal slot (e.g. sparse offsets).
                    if syncer.has_active_sync:
                        raise RuntimeError("an active capture crossed the syncer-clock target")
                    if not delivered(syncer, peers):
                        return False
                    syncer.global_step = d.syncer_steps
                    for peer in peers:
                        if not peer.available or peer.metadata().learner_id in terminal_sent:
                            continue
                        try:
                            peer.transport.send_reliable("syncer_clock", {"global_step": d.syncer_steps}, deadline=deadline, progress=lambda: [p.pump() for p in peers])
                            terminal_sent.add(peer.metadata().learner_id)
                        except (ConnectionError, OSError) as exc:
                            peer.mark_unavailable(exc)
                    if sum(peer.available for peer in peers) < d.min_quorum:
                        raise RuntimeError("available learner count fell below quorum during terminal clock delivery")
                    return all(peer.training_done and peer.metadata().syncer_step == d.syncer_steps
                               for peer in peers if peer.available) and delivered(syncer, peers)
                result = controller.tick()
                if result is not None:
                    row = asdict(result)
                    writer.writerow({"elapsed_s": time.monotonic() - started, **{key: json.dumps(row[key]) if isinstance(row[key], tuple) else row[key] for key in writer.fieldnames if key != "elapsed_s"}})
                    stream.flush()
                    print(f"sync={result.sync_step} global_step={result.global_step} fragment={result.fragment_id} revision={result.fragment_revision} learners={list(result.learner_ids)} tokens={list(result.tokens)}", flush=True)
                return False if clock_mode else drain_complete(syncer, peers)

            _pump_until(peers, processes, deadline, synchronize, min_quorum=d.min_quorum)
        _send_available(peers, "pause")
        _pump_until(peers, processes, deadline, lambda: all(peer.paused for peer in peers if peer.available), min_quorum=d.min_quorum)
        if not delivered(syncer, peers):
            raise RuntimeError("final fragment revisions were not applied by every learner")
        _send_available(peers, "stop")
        _pump_until(peers, processes, deadline, lambda: all(peer.stopped for peer in peers if peer.available), min_quorum=d.min_quorum)
        from .shutdown import wait_for_learner_shutdown
        wait_for_learner_shutdown({i:process for i,(peer,process) in enumerate(zip(peers, processes)) if peer.available}, deadline, folder)
        metadata = [asdict(peer.metadata()) for peer in peers]
        if d.stopping == "syncer_steps" and any(peer.metadata().syncer_step != d.syncer_steps for peer in peers if peer.available):
            raise RuntimeError("a learner did not reach the configured syncer-clock budget")
        if d.stopping == "local_steps" and any(peer.metadata().total_local_steps != options.steps for peer in peers if peer.available):
            raise RuntimeError("a learner did not complete its configured optimizer-step budget")
        if d.stopping == "local_steps" and not all(revision >= 1 for revision in syncer.fragment_revisions):
            raise RuntimeError("training ended without synchronizing every fragment; increase run.steps")
        final_parameters = syncer.optimizer.model_snapshot()
        monitor.trajectory.save(syncer.scheduler.sync_step, final_parameters, tokens=sum(p.metadata().total_tokens for p in peers), force=True, final=True)
        if any(not torch.isfinite(value).all() for value in final_parameters.values()):
            raise RuntimeError("final global model contains nonfinite parameters")
        # Retain the Phase 7 schema identifier for backward compatibility;
        # the explicit method tag identifies the selected outer optimizer.
        torch.save({"format": "decoupled_heloco_global_v1", "method": config.method, "parameters": final_parameters, "fragment_revisions": syncer.fragment_revisions, "layout_signature": manager.layout_signature, "resumable": False, "syncer_step": syncer.global_step}, folder / "global_model.pt")
        failed_learners = {str(peer.metadata().learner_id): peer.failure for peer in peers if not peer.available}
        manifest.update(status="passed", degraded=bool(failed_learners), failed_learners=failed_learners, completed_learner_ids=[peer.metadata().learner_id for peer in peers if peer.available], elapsed_s=time.monotonic() - started, fragment_revisions=syncer.fragment_revisions, sync_updates=syncer.scheduler.sync_step, syncer_step=syncer.global_step, peak_inflight_captures=syncer.peak_inflight_captures, capture_timeouts=controller.timeouts, next_global_schedule_step=syncer.scheduler.global_step, learners=metadata, wire_sent_bytes=sum(t.sent_bytes for t in transports), wire_received_bytes=sum(t.received_bytes for t in transports))
        print(f"Final fragment revisions: {list(syncer.fragment_revisions)}", flush=True)
        print(f"Local optimizer steps: {[p.metadata().total_local_steps for p in peers]}", flush=True)
        print(f"Capture timeouts: {controller.timeouts}", flush=True)
        print("GPU TRAINING CHECK PASSED: surviving learners completed their selected budgets, fragment transfers, final revisions, and clean shutdown." + (f" Degraded run: {failed_learners}" if failed_learners else ""), flush=True)
        return 0
    except KeyboardInterrupt:
        manifest.update(status="interrupted", error="interrupted by user")
        raise
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        print(f"GPU TRAINING CHECK FAILED: {exc}\nLogs: {folder}", file=sys.stderr, flush=True)
        return 2
    finally:
        if monitor is not None:
            monitor.close()
            monitor = None
        if syncer is not None:
            if syncer.syncer_shards > 1:
                with (folder / "syncer_replica_processing.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=("rank", "wall_s", "process_cpu_s", "collective_tensor_bytes"))
                    writer.writeheader()
                    writer.writerows(syncer.optimizer.stats)
            syncer.close()
        (folder / "transport_diagnostics.json").write_text(json.dumps([t.diagnostics() for t in transports], indent=2))
        for transport in transports:
            transport.close()
        listener.close()
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if monitor is not None:
            monitor.close()
        for log in logs:
            log.close()
        (folder / "summary.json").write_text(json.dumps(manifest, indent=2))
