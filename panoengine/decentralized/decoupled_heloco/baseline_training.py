"""Matched dense training using the repository's ORIGINAL HTTP outer methods.

The coordinator owns the original CPU server; separate learners own their
models and optimizers. Files in a fresh run directory implement startup and
final-pull barriers only. Training exchanges use the original HTTP protocol.
"""

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

from .baseline_options import (BASELINE_FORMAT, BASELINE_SCOPE, BASELINE_STOPPING,
                               baseline_metadata, validate_baseline_options)
from .config import ConfigError
from .evaluation import parameter_fingerprint
from .fragment_manager import FragmentManager
from .gpu_recipe import build_initial_model, build_recipe, validate_dense_options, validate_tokenizer
from .gpu_training import select_devices
from .state import _cpu_copy


def write_json(path, value):
    """Same-directory atomic publication: readers never observe partial JSON."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def build_baseline_server(model, config, options):
    """Use the original classes, and the settings used by run_heloco.ps_cmd.

    No alternate optimizer implementation or model dtype conversion occurs.
    In particular, complex RoPE buffers stay complex.
    """
    from torchft.http import _IPv6HTTPServer
    from panoengine.decentralized.async_diloco import AsyncDiLoCoServer, DelayedNesterovOptimizer
    from panoengine.decentralized.heloco import HeLoCoOptimizer, HeLoCoServer
    from panoengine.decentralized.mla import MLAOptimizer, MLAServer

    validate_baseline_options(config, options)
    metadata = baseline_metadata(config, options)
    ipv6 = _IPv6HTTPServer.address_family == socket.AF_INET6
    kwargs = dict(port=0, bind_host="::1" if ipv6 else "127.0.0.1",
                #   advertise_host="[::1]" if ipv6 else "127.0.0.1",
                  advertise_host="localhost",
                  request_timeout=options.ps_timeout, dylu_H=0, grace_period=0.0,
                  should_quantize=False, num_fragments=1, sync_workers=0)
    if config.method == "heloco":
        optimizer = HeLoCoOptimizer(model.parameters(), lr=options.outer_lr, momentum=options.outer_momentum)
        return HeLoCoServer(model, optimizer, rho=metadata["rho"], correction_workers="all",
                            correction_scope="tensorwise", **kwargs)
    if config.method == "mla":
        optimizer = MLAOptimizer(model.parameters(), lr=options.outer_lr, momentum=options.outer_momentum)
        return MLAServer(model, optimizer, **kwargs)
    optimizer = DelayedNesterovOptimizer(model.parameters(), lr=options.outer_lr,
                                         momentum=options.outer_momentum, nesterov_period=options.islands)
    return AsyncDiLoCoServer(model, optimizer, **kwargs)


def preflight_baseline(config, options, repo_root):
    validate_baseline_options(config, options)
    validate_dense_options(options)
    if not torch.cuda.is_available():
        raise ConfigError("CUDA is unavailable; use your allocated GPU job and heloco environment")
    # Fail on missing original HTTP dependencies before launching any learner.
    from panoengine.decentralized.async_diloco import AsyncDiLoCo

    devices = select_devices(options, os.environ.get("CUDA_VISIBLE_DEVICES"), torch.cuda.device_count())
    assets = (Path(repo_root) / options.hf_assets).resolve()
    if not assets.is_dir():
        raise ConfigError(f"tokenizer directory does not exist: {assets}")
    recipe = build_recipe(options, repo_root, Path(repo_root) / options.log_dir)
    tokenizer = recipe.tokenizer.build(tokenizer_path=recipe.hf_assets_path)
    size, largest = validate_tokenizer(tokenizer, recipe.model_spec.model.vocab_size)
    model = build_initial_model(recipe, options.seed)
    manager = FragmentManager.from_model(model, 1)
    if options.tokens_per_parameter is not None:
        options.steps = math.ceil(options.tokens_per_parameter * manager.total_numel / (options.islands * options.batch * options.seq_len))
    if options.steps % options.sync_steps:
        raise ConfigError("resolved step budget must be divisible by run.sync_steps; no partial-window flush is added")
    print(f"Tokenizer compatible: vocabulary={size}, max_token_id={largest}")
    print(f"Matched GPU preflight passed: method={config.method}, learners={options.islands}, parameters={manager.total_numel:,}, steps_per_learner={options.steps}, device_masks={devices}")
    print("Recipe: dense FP32 Llama 15M, FlexAttention, plain AdamW; IID learner shards.")
    print(f"Original outer settings: {json.dumps(baseline_metadata(config, options))}")
    return model, devices


def run_baseline_training(config, options, repo_root, *, timeout=1800.0, check_only=False, output_dir=None):
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ConfigError("--training-timeout must be finite and positive")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, devices = preflight_baseline(config, options, repo_root)
        if check_only:
            return 0
        if output_dir is None:
            parent = Path(repo_root) / options.log_dir
            parent.mkdir(parents=True, exist_ok=True)
            folder = parent / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(3))
        else:
            folder = (Path(repo_root) / output_dir).resolve()
        folder.mkdir(parents=True, exist_ok=False)
        return _run_baseline(config, options, model, devices, folder, timeout, repo_root)
    finally:
        torch.set_num_threads(previous_threads)


def _run_baseline(config, options, model, devices, folder, timeout, repo_root, *, worker_command=None):
    """Also exercised with real CPU optimizer steps in separate subprocesses."""
    import yaml

    validate_baseline_options(config, options)
    if len(devices) != options.islands:
        raise ConfigError("device list does not match learner count")
    folder = Path(folder).resolve()
    (folder / "experiment.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False))
    write_json(folder / "run-options.json", vars(options))
    initial_hash = parameter_fingerprint(dict(model.named_parameters()))
    manifest = {"status": "starting", "scope": BASELINE_SCOPE, "method": config.method,
                "devices": devices, "steps_per_learner": options.steps,
                "initial_parameters_sha256": initial_hash, "checkpoint_resumable": False,
                "outer_optimizer": baseline_metadata(config, options), "stopping_policy": BASELINE_STOPPING,
                "pacing": "extra sleep = measured local train_step time * (factor - 1); HTTP time excluded",
                "communication_accounting": "original client protocol bytes including initial/final pulls; excludes HTTP headers and heartbeats"}
    processes, logs, server = [], [], None
    deadline = time.monotonic() + timeout
    startup_deadline = min(deadline, time.monotonic() + options.ps_timeout)
    started = None
    try:
        manager = FragmentManager.from_model(model, 1)
        server = build_baseline_server(model, config, options)
        write_json(folder / "server.json", {"address": server.address(), "heartbeat_address": server.heartbeat_address(),
                                             "initial_parameters_sha256": initial_hash})
        used_ports = set()
        for learner_id, device in enumerate(devices):
            child = folder / f"learner_{learner_id}"
            child.mkdir()
            log = (child / "trainer.log").open("w", encoding="utf-8")
            logs.append(log)
            while True:
                with socket.socket() as reservation:
                    reservation.bind(("127.0.0.1", 0))
                    rank_port = reservation.getsockname()[1]
                if rank_port not in used_ports:
                    break
            used_ports.add(rank_port)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, RANK="0", LOCAL_RANK="0", WORLD_SIZE="1",
                           LOCAL_WORLD_SIZE="1", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(rank_port),
                           PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", PF_ISLAND_ID=str(learner_id),
                           PF_WIRE_BF16="0", PF_ISLAND_SLOWNESS_FACTOR="1", PANOFABRIC_METHOD=config.method,
                           PANOFABRIC_COMM_LOG_DIR=str(child / "communication"))
            # Loopback HTTP must bypass any cluster proxy.
            for name in ("NO_PROXY", "no_proxy"):
                env[name] = env.get(name, "") + ",localhost,127.0.0.1"
            # for name in ("NO_PROXY", "no_proxy"):
            #     env[name] = env.get(name, "") + ",localhost,127.0.0.1,::1,[::1]"
            for name in ("DILOCO_SERVER_ADDR", "DILOCO_HB_ADDR", "TORCHFT_LIGHTHOUSE"):
                env.pop(name, None)
            command = [*(worker_command or [sys.executable, "-m", "panoengine.decentralized.decoupled_heloco.baseline_worker"]),
                       "--run-dir", str(folder), "--learner-id", str(learner_id), "--timeout", str(timeout)]
            processes.append(subprocess.Popen(command, cwd=repo_root, env=env, stdout=log, stderr=subprocess.STDOUT))

        def wait_for_files(name, limit):
            while time.monotonic() < limit:
                paths = [folder / f"learner_{i}" / name for i in range(options.islands)]
                if any(process.poll() is not None and (process.returncode != 0 or not path.is_file())
                       for process, path in zip(processes, paths)):
                    raise RuntimeError(f"baseline learner exited before {name}; inspect learner_*/trainer.log")
                if all(path.is_file() for path in paths):
                    return [json.loads(path.read_text()) for path in paths]
                time.sleep(0.02)
            raise TimeoutError(f"baseline deadline exceeded waiting for {name}; inspect learner logs")

        ready = wait_for_files("ready.json", startup_deadline)
        if any(row.get("learner_id") != i or row.get("pid") != processes[i].pid or row.get("initial_parameters_sha256") != initial_hash or row.get("revision") != 0
               for i, row in enumerate(ready)) or server.status()["revision"] != 0:
            raise RuntimeError("baseline learners did not all adopt the identical revision-0 initialization")
        started = time.monotonic()
        print(f"Matched GPU training started: method={config.method}, server_pid={os.getpid()}, learner_pids={[p.pid for p in processes]}", flush=True)
        print(f"Logs: {folder}", flush=True)
        write_json(folder / "start.json", {"start": True})
        completed = wait_for_files("training-done.json", deadline)
        expected_pushes = options.steps // options.sync_steps
        if any(row.get("local_steps") != options.steps or row.get("pushes") != expected_pushes or row.get("remaining_steps") != 0
               for row in completed):
            raise RuntimeError("baseline local budget or window accounting differs from the configured run")
        status = server.status()
        expected_revision = expected_pushes * options.islands
        if status["applied_pushes"] != expected_revision or status["revision"] != expected_revision:
            raise RuntimeError("not every completed local window was committed exactly once")
        with server._lock:
            parameters = _cpu_copy(dict(model.named_parameters()))
            dispatch = _cpu_copy(server._build_snapshot_locked(server._param_names, island_id=0))
        if any(not torch.isfinite(v).all() for v in (*parameters.values(), *dispatch.values())):
            raise RuntimeError("final global or dispatched model contains nonfinite parameters")
        dispatch_hash = parameter_fingerprint(dispatch)
        write_json(folder / "finalize.json", {"revision": expected_revision, "dispatch_sha256": dispatch_hash})
        learners = wait_for_files("result.json", deadline)
        for process in processes:
            process.wait(timeout=max(0.001, min(5.0, deadline - time.monotonic())))
            if process.returncode != 0:
                raise RuntimeError("baseline learner failed during finalization")
        for learner_id, row in enumerate(learners):
            if (row.get("learner_id") != learner_id or row.get("total_local_steps") != options.steps
                    or row.get("pushes") != expected_pushes or row.get("final_dispatch_sha256") != dispatch_hash
                    or row.get("fragments") != [{"fragment_id": 0, "local_steps": 0, "tokens": 0,
                                                 "last_received_revision": expected_revision, "last_applied_revision": expected_revision}]):
                raise RuntimeError("baseline learner final revision, dispatch or budget verification failed")
        if server.status()["revision"] != expected_revision or server.status()["finished_count"] != options.islands:
            raise RuntimeError("baseline server changed after final pull or a learner did not announce clean completion")
        torch.save({"format": BASELINE_FORMAT, "method": config.method, "parameters": parameters,
                    "fragment_revisions": [expected_revision], "layout_signature": manager.layout_signature,
                    "resumable": False}, folder / "global_model.pt")
        manifest.update(status="passed", elapsed_s=time.monotonic() - started, fragment_revisions=[expected_revision],
                        sync_updates=expected_revision, capture_timeouts=0, learners=learners,
                        protocol_sent_bytes=sum(row["protocol_received_bytes"] for row in learners),
                        protocol_received_bytes=sum(row["protocol_sent_bytes"] for row in learners),
                        final_dispatch_sha256=dispatch_hash, global_parameters_sha256=parameter_fingerprint(parameters))
        print(f"Global revision: {expected_revision}; local optimizer steps: {[r['total_local_steps'] for r in learners]}", flush=True)
        print("MATCHED BASELINE CHECK PASSED: all windows committed, final dispatch verified, and clean shutdown.", flush=True)
        return 0
    except KeyboardInterrupt:
        manifest.update(status="interrupted", error="interrupted by user")
        raise
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        print(f"MATCHED BASELINE CHECK FAILED: {exc}\nLogs: {folder}", file=sys.stderr, flush=True)
        return 2
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if server is not None:
            server.shutdown()
        for log in logs:
            log.close()
        write_json(folder / "summary.json", manifest)
