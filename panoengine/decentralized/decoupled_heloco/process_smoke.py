"""One syncer process and independently training CPU learners over localhost TCP."""

from dataclasses import asdict
import math
import multiprocessing
import os
import secrets
import socket
import sys
import threading
import time

import torch
from torch.nn import functional as F

from .config import ExperimentConfig
from .learner import DecoupledLearner
from .smoke import _TinyTokenModel
from .state import _cpu_copy
from .syncer import DecoupledSyncer
from .timing import TimedSyncController
from .transport import FramedTransport, RemoteLearner, metadata_from_wire, snapshot_to_wire


def _wait_message(transport, deadline):
    while time.monotonic() < deadline:
        message = transport.receive()
        if message is not None:
            if message["kind"] == "fatal":
                raise RuntimeError(message["body"]["error"])
            return message
        time.sleep(0.005)
    raise TimeoutError("learner handshake timed out")


def _serve_learner(transport, learner, started, pause, parked, stop, finished, failures, *, training_done=None, deadline=None):
    def send(kind, body):
        transport.send_reliable(kind, body, deadline=deadline)

    captures = {}
    last_report = 0.0
    pause_reported = False
    done_reported = False
    try:
        while True:
            if finished.is_set():
                send("stopped", {"metadata": asdict(learner.metadata())})
                transport.flush(timeout=60. if deadline is None else max(.001, deadline-time.monotonic()))
                return
            while True:
                message = transport.receive()
                if message is None:
                    break
                kind, body = message["kind"], message["body"]
                if kind == "start":
                    started.set()
                elif kind == "pull":
                    request_id = body["request_id"]
                    try:
                        future = learner.request_snapshot(request_id, body["fragment_id"], body["expected_revision"], min_local_steps=body["min_local_steps"])
                    except Exception as exc:
                        send("snapshot_error", {"request_id": request_id, "error": str(exc)})
                    else:
                        captures[request_id] = (future, False)
                elif kind == "release":
                    request_id = body["request_id"]
                    learner.release_snapshot(request_id)
                    captures.pop(request_id, None)
                elif kind == "update":
                    learner.queue_update(body["fragment_id"], body["parameters"], body["revision"], layout_signature=body["layout_signature"], global_step=body.get("global_step", 0))
                    send("update_ack", {"metadata": asdict(learner.metadata())})
                elif kind == "syncer_clock":
                    learner.queue_syncer_clock(body["global_step"])
                elif kind == "pause":
                    pause.set()
                elif kind == "stop":
                    stop.set()
                else:
                    raise ValueError(f"unexpected syncer message: {kind}")
            # Futures are inspected and serialized off the training thread.
            for request_id, (future, sent) in tuple(captures.items()):
                if sent or not future.done():
                    continue
                try:
                    snapshot = future.result()
                except Exception as exc:
                    send("snapshot_error", {"request_id": request_id, "error": str(exc)})
                else:
                    send("snapshot", {"request_id": request_id, "snapshot": snapshot_to_wire(snapshot)})
                captures[request_id] = (future, True)
            if parked.is_set() and not pause_reported:
                send("paused", {"metadata": asdict(learner.metadata())})
                pause_reported = True
            if training_done is not None and training_done.is_set() and not done_reported:
                send("training_done", {"metadata": asdict(learner.metadata())})
                done_reported = True
            now = time.monotonic()
            if started.is_set() and now - last_report >= 0.02:
                transport.send_progress({"metadata": asdict(learner.metadata())})
                last_report = now
            time.sleep(0.002)
    except Exception as exc:
        failures.append(exc)
        stop.set()


def _learner_process(address, token, learner_id, config, timeout):
    transport = None
    try:
        torch.set_num_threads(1)
        torch.manual_seed(config.run.get("seed", 42))
        transport = FramedTransport(socket.create_connection(address, timeout=5.0))
        transport.send("hello", {"learner_id": learner_id, "token": token})
        message = _wait_message(transport, time.monotonic() + timeout)
        if message["kind"] != "initialize":
            raise ValueError("expected global initialization before training")
        model = _TinyTokenModel(config.decoupled.num_fragments)
        model.load_state_dict(message["body"]["parameters"])
        learner = DecoupledLearner(model, config.decoupled.num_fragments, learner_id=learner_id, max_snapshot_requests=config.decoupled.max_inflight_captures)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        transport.send("ready", {"pid": os.getpid(), "metadata": asdict(learner.metadata())})
        started, pause, parked, stop, finished = (threading.Event() for _ in range(5))
        failures = []
        service = threading.Thread(target=_serve_learner, args=(transport, learner, started, pause, parked, stop, finished, failures), daemon=True)
        service.start()
        tokens = torch.arange(8).repeat(2, 1)
        targets = (tokens + 1) % 8
        try:
            while not stop.is_set():
                if not started.is_set() or pause.is_set():
                    learner.boundary()
                    if pause.is_set():
                        parked.set()
                    stop.wait(0.005)
                    continue
                with learner.training_step(tokens=tokens.numel()):
                    optimizer.zero_grad()
                    loss = F.cross_entropy(model(tokens).reshape(-1, 8), targets.flatten())
                    loss.backward()
                    optimizer.step()
                # Artificial CPU pacing, outside the optimizer step. There is
                # no wait for a pull, response, or broadcast in training.
                stop.wait(0.01 * (learner_id + 1))
        finally:
            learner.boundary()
            finished.set()
            service.join(timeout=3.0)
        if service.is_alive():
            raise TimeoutError("learner control thread did not stop")
        if failures:
            raise RuntimeError(str(failures[0]))
        transport.flush()
    except Exception as exc:
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


def _pump_until(peers, processes, deadline, predicate):
    while time.monotonic() < deadline:
        for peer in peers:
            peer.pump()
        for peer, process in zip(peers, processes):
            if process.exitcode is not None and not peer.stopped:
                raise RuntimeError(f"learner {peer.metadata().learner_id} exited before shutdown acknowledgement")
        if predicate():
            return
        time.sleep(0.005)
    raise TimeoutError("process smoke test exceeded --process-timeout")


def run_process_smoke(config: ExperimentConfig, *, learners: int = 2, cycles: int = 2, timeout: float = 60.0) -> int:
    """Run actual localhost transfers; tiny dense CPU recipe only.

    The parent is the syncer, children are spawned learner processes. Initial
    weights are sent explicitly before training. Interval/grace controls are
    real monotonic time, and the run ends after cycles * P committed updates.
    The process budget overrides run.islands for this demo only. Wire totals
    include framing, initialization, metadata, control, and tensor messages.
    """
    for name, value in (("learners", learners), ("cycles", cycles)):
        if type(value) is not int or value < 1:
            raise ValueError(f"process {name} must be a positive integer")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("process timeout must be finite and positive")
    if config.method not in {"decoupled_heloco", "decoupled_diloco"} or learners < config.decoupled.min_quorum:
        raise ValueError("process demo requires a decoupled method and at least min_quorum learners")
    old_threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.run.get("seed", 42))
        torch.set_num_threads(1)
        try:
            return _run(config, learners, cycles, timeout)
        finally:
            torch.set_num_threads(old_threads)


def _run(config, learners, cycles, timeout):
    decoupled = config.decoupled
    model = _TinyTokenModel(decoupled.num_fragments)
    tokens = torch.arange(8).repeat(2, 1)
    targets = (tokens + 1) % 8
    with torch.no_grad():
        initial_loss = float(F.cross_entropy(model(tokens).reshape(-1, 8), targets.flatten()))
    initial = _cpu_copy(dict(model.named_parameters()))
    context = multiprocessing.get_context("spawn")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(learners)
    listener.settimeout(0.1)
    token = secrets.token_hex(32)
    transports, processes, peers = [], [], []
    deadline = time.monotonic() + timeout
    syncer = None
    try:
        for learner_id in range(learners):
            process = context.Process(target=_learner_process, args=(listener.getsockname(), token, learner_id, config, timeout), name=f"decoupled-learner-{learner_id}")
            process.start()
            processes.append(process)
        by_id = {}
        while len(by_id) < learners:
            if time.monotonic() >= deadline:
                raise TimeoutError("learner startup exceeded --process-timeout")
            if any(process.exitcode is not None for process in processes):
                raise RuntimeError("a learner process exited during startup")
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            transport = FramedTransport(connection)
            transports.append(transport)
            hello = _wait_message(transport, deadline)
            body = hello["body"]
            learner_id = body.get("learner_id")
            if hello["kind"] != "hello" or not secrets.compare_digest(body.get("token", ""), token):
                raise ValueError("learner handshake token mismatch")
            if type(learner_id) is not int or not 0 <= learner_id < learners or learner_id in by_id:
                raise ValueError("invalid or duplicate learner identity")
            transport.send("initialize", {"parameters": initial})
            ready = _wait_message(transport, deadline)
            if ready["kind"] != "ready":
                raise ValueError("expected learner readiness acknowledgement")
            metadata = metadata_from_wire(ready["body"]["metadata"])
            if metadata.learner_id != learner_id:
                raise ValueError("learner handshake identity changed")
            peer = RemoteLearner(transport, metadata)
            peer.pid = ready["body"]["pid"]
            if peer.pid != processes[learner_id].pid:
                raise ValueError("learner process identity mismatch")
            by_id[learner_id] = peer
        peers = [by_id[i] for i in range(learners)]
        syncer = DecoupledSyncer(model, peers, decoupled.num_fragments, min_quorum=decoupled.min_quorum, overlap_steps=decoupled.overlap_steps, outer_lr=config.run.get("outer_lr", 0.7), outer_method=config.fragment_outer_method, outer_momentum=config.run.get("outer_momentum", 0.9), heloco=decoupled.heloco, weighting=decoupled.weighting, merge=decoupled.merge, scheduler=decoupled.scheduler, sync_period=decoupled.sync_period, fragment_offsets=decoupled.fragment_offsets, min_local_steps=decoupled.min_local_steps, max_inflight_captures=decoupled.max_inflight_captures, syncer_shards=decoupled.syncer_shards, syncer_timeout=decoupled.syncer_timeout)
        controller = TimedSyncController(syncer, sync_interval=decoupled.sync_interval, grace_window_factor=decoupled.grace_window_factor, adaptive_grace=decoupled.adaptive_grace, ema_alpha=decoupled.timing_ema_alpha)
        for peer in peers:
            peer.transport.send("start", {})
        started = time.monotonic()
        target_updates = cycles * decoupled.num_fragments
        print(f"CPU process smoke: syncer_pid={os.getpid()}, learner_pids={[peer.pid for peer in peers]}", flush=True)
        print(f"Outer optimizer: {config.fragment_outer_method}", flush=True)
        print(f"Localhost TCP: learners={learners}, fragments={decoupled.num_fragments}, quorum={decoupled.min_quorum}, overlap={decoupled.overlap_steps}", flush=True)
        print(f"Interval={decoupled.sync_interval}s; grace={decoupled.sync_interval * decoupled.grace_window_factor}s after captured quorum; target_updates={target_updates}", flush=True)

        def synchronize():
            result = controller.tick()
            if result is not None:
                print(f"elapsed={time.monotonic() - started:.2f}s sync={result.sync_step} fragment={result.fragment_id} revision={result.fragment_revision} learners={list(result.learner_ids)} local_steps={list(result.local_steps)} weights={[round(w, 3) for w in result.weights]}", flush=True)
            return syncer.scheduler.sync_step >= target_updates

        _pump_until(peers, processes, deadline, synchronize)
        revisions = syncer.fragment_revisions

        def delivered():
            return not syncer.retry_broadcast() and all(fragment.last_applied_revision == revisions[fragment.fragment_id] for peer in peers for fragment in peer.metadata().fragments)

        _pump_until(peers, processes, deadline, delivered)
        for peer in peers:
            peer.transport.send("pause", {})
        _pump_until(peers, processes, deadline, lambda: all(peer.paused for peer in peers))
        for peer in peers:
            peer.transport.send("stop", {})
        _pump_until(peers, processes, deadline, lambda: all(peer.stopped for peer in peers))
        for process in processes:
            process.join(timeout=3.0)
            if process.is_alive() or process.exitcode != 0:
                raise RuntimeError(f"{process.name} did not exit successfully")
        if any(peer.metadata().total_tokens != 16 * peer.metadata().total_local_steps for peer in peers):
            raise ValueError("learner token accounting mismatch")
        model.load_state_dict(syncer.optimizer.model_snapshot())
        with torch.no_grad():
            final_loss = float(F.cross_entropy(model(tokens).reshape(-1, 8), targets.flatten()))
        if not math.isfinite(final_loss) or revisions != (cycles,) * decoupled.num_fragments:
            raise ValueError("invalid final global state")
        print(f"Final fragment revisions: {list(revisions)}")
        print(f"Local optimizer steps: {[peer.metadata().total_local_steps for peer in peers]}")
        print(f"Wire bytes including initialization/control: syncer_sent={sum(t.sent_bytes for t in transports)}, syncer_received={sum(t.received_bytes for t in transports)}")
        print(f"Capture timeouts: {controller.timeouts}; global toy loss: {initial_loss:.6f} -> {final_loss:.6f}")
        print("PROCESS SMOKE TEST PASSED: separate learner processes transferred fragments, applied final revisions, and shut down cleanly.")
        return 0
    except Exception as exc:
        print(f"PROCESS SMOKE TEST FAILED: {exc}", file=sys.stderr)
        return 2
    finally:
        if syncer is not None:
            syncer.close()
        for transport in transports:
            transport.close()
        listener.close()
        for process in processes:
            process.join(timeout=0.2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
