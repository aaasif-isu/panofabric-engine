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

STOPPING_POLICY = "fixed_local_budgets_drain_to_first_unavailable_round_robin_quorum"


def outer_optimizer_metadata(config, options):
    diloco = config.fragment_outer_method == "diloco"
    return {"name": "nesterov_sgd" if diloco else "heloco",
            "lr": options.outer_lr, "momentum": options.outer_momentum,
            "momentum_convention": "sum" if diloco else "ema",
            "correction_enabled": False if diloco else config.decoupled.heloco.correction_enabled,
            "lookahead": False if diloco else config.decoupled.heloco.lookahead,
            "merge": "token_weighted_average"}


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
    cfg = build_recipe(options, repo_root, Path(repo_root) / options.log_dir)
    tokenizer = cfg.tokenizer.build(tokenizer_path=cfg.hf_assets_path)
    vocab_size, max_token_id = validate_tokenizer(tokenizer, cfg.model_spec.model.vocab_size)
    print(f"Tokenizer compatible: vocabulary={vocab_size}, max_token_id={max_token_id}, model_vocab_size={cfg.model_spec.model.vocab_size}")
    model = build_initial_model(cfg, options.seed)
    manager = validate_frame_sizes(model, config.decoupled.num_fragments)
    if options.tokens_per_parameter is not None:
        options.steps = math.ceil(options.tokens_per_parameter * manager.total_numel / (options.islands * options.batch * options.seq_len))
    if options.steps < config.decoupled.overlap_steps:
        raise ConfigError("the resolved training budget is smaller than overlap_steps; increase run.steps or the token budget")
    print(f"GPU preflight passed: learners={options.islands}, parameters={manager.total_numel:,}, steps_per_learner={options.steps}, device_masks={devices}")
    print("Recipe: dense FP32 Llama 15M, FlexAttention, plain AdamW; IID data sharded across learners.")
    print(f"Outer optimizer: {config.fragment_outer_method}; stopping: fixed local budgets and quorum-preserving drain.")
    print("Dataset iteration and CUDA forward/backward will be checked during training.")
    return model, devices


def delivered(syncer, peers):
    revisions = syncer.fragment_revisions
    return not syncer.retry_broadcast() and all(f.last_applied_revision == revisions[f.fragment_id] for peer in peers for f in peer.metadata().fragments)


def drain_complete(syncer, peers):
    """Stop at the first unavailable round-robin quorum after budgets end."""
    if not all(peer.training_done for peer in peers) or syncer.has_active_sync or not delivered(syncer, peers):
        return False
    fragment_id = syncer.scheduler.fragment_id
    return syncer.scheduler.plan((p.metadata() for p in peers), fragment_revision=syncer.fragment_revisions[fragment_id]) is None


def _pump_until(peers, processes, deadline, predicate):
    while time.monotonic() < deadline:
        for peer in peers:
            peer.pump()
        for index, process in enumerate(processes):
            if process.poll() is not None and (index >= len(peers) or not peers[index].stopped):
                raise RuntimeError(f"learner {index} exited early; inspect learner_{index}/trainer.log")
        if predicate():
            return
        time.sleep(0.005)
    raise TimeoutError("GPU run exceeded --training-timeout; inspect learner logs or increase the deadline")


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

    manifest = {"status": "starting", "scope": "dense_single_gpu_localhost", "devices": devices, "steps_per_learner": options.steps, "checkpoint_resumable": False,
                "method": config.method, "outer_optimizer": outer_optimizer_metadata(config, options),
                "stopping_policy": STOPPING_POLICY,
                "initial_parameters_sha256": parameter_fingerprint(dict(model.named_parameters()))}
    syncer = None
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
        peers = [by_id[i] for i in range(options.islands)]
        d = config.decoupled
        syncer = DecoupledSyncer(model, peers, d.num_fragments, min_quorum=d.min_quorum, overlap_steps=d.overlap_steps, outer_lr=options.outer_lr, outer_method=config.fragment_outer_method, outer_momentum=options.outer_momentum, heloco=d.heloco)
        controller = TimedSyncController(syncer, sync_interval=d.sync_interval, grace_window_factor=d.grace_window_factor)
        print(f"GPU training started: syncer_pid={os.getpid()}, learner_pids={[p.pid for p in peers]}", flush=True)
        print(f"Logs: {folder}", flush=True)
        started = time.monotonic()
        for peer in peers:
            peer.transport.send("start", {})
        with (folder / "syncs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("elapsed_s", "sync_step", "fragment_id", "fragment_revision", "learner_ids", "local_steps", "tokens", "weights"))
            writer.writeheader()

            def synchronize():
                result = controller.tick()
                if result is not None:
                    row = asdict(result)
                    writer.writerow({"elapsed_s": time.monotonic() - started, **{key: json.dumps(row[key]) if isinstance(row[key], tuple) else row[key] for key in writer.fieldnames if key != "elapsed_s"}})
                    stream.flush()
                    print(f"sync={result.sync_step} fragment={result.fragment_id} revision={result.fragment_revision} learners={list(result.learner_ids)} tokens={list(result.tokens)}", flush=True)
                return drain_complete(syncer, peers)

            _pump_until(peers, processes, deadline, synchronize)
        for peer in peers:
            peer.transport.send("pause", {})
        _pump_until(peers, processes, deadline, lambda: all(peer.paused for peer in peers))
        if not delivered(syncer, peers):
            raise RuntimeError("final fragment revisions were not applied by every learner")
        for peer in peers:
            peer.transport.send("stop", {})
        _pump_until(peers, processes, deadline, lambda: all(peer.stopped for peer in peers))
        for process in processes:
            process.wait(timeout=max(0.001, min(5.0, deadline - time.monotonic())))
            if process.returncode != 0:
                raise RuntimeError("a GPU learner failed during shutdown; inspect its trainer.log")
        metadata = [asdict(peer.metadata()) for peer in peers]
        if any(peer.metadata().total_local_steps != options.steps for peer in peers):
            raise RuntimeError("a learner did not complete its configured optimizer-step budget")
        if not all(revision >= 1 for revision in syncer.fragment_revisions):
            raise RuntimeError("training ended without synchronizing every fragment; increase run.steps")
        final_parameters = syncer.optimizer.model_snapshot()
        if any(not torch.isfinite(value).all() for value in final_parameters.values()):
            raise RuntimeError("final global model contains nonfinite parameters")
        # Retain the Phase 7 schema identifier for backward compatibility;
        # the explicit method tag identifies the selected outer optimizer.
        torch.save({"format": "decoupled_heloco_global_v1", "method": config.method, "parameters": final_parameters, "fragment_revisions": syncer.fragment_revisions, "layout_signature": manager.layout_signature, "resumable": False}, folder / "global_model.pt")
        manifest.update(status="passed", elapsed_s=time.monotonic() - started, fragment_revisions=syncer.fragment_revisions, sync_updates=syncer.scheduler.sync_step, capture_timeouts=controller.timeouts, learners=metadata, wire_sent_bytes=sum(t.sent_bytes for t in transports), wire_received_bytes=sum(t.received_bytes for t in transports))
        print(f"Final fragment revisions: {list(syncer.fragment_revisions)}", flush=True)
        print(f"Local optimizer steps: {[p.metadata().total_local_steps for p in peers]}", flush=True)
        print(f"Capture timeouts: {controller.timeouts}", flush=True)
        print("GPU TRAINING CHECK PASSED: fixed local budgets, fragment transfers, final revisions, and clean shutdown verified.", flush=True)
        return 0
    except KeyboardInterrupt:
        manifest.update(status="interrupted", error="interrupted by user")
        raise
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        print(f"GPU TRAINING CHECK FAILED: {exc}\nLogs: {folder}", file=sys.stderr, flush=True)
        return 2
    finally:
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
        for log in logs:
            log.close()
        (folder / "summary.json").write_text(json.dumps(manifest, indent=2))
