"""Bounded, threaded localhost TCP messages and remote learner endpoints.

    Wire messages contain basic dictionaries and detached CPU tensors, never
    futures/dataclasses or references to live model storage. torch.load uses
    weights_only=True. Endpoints support explicit authenticated reattachment;
    automatic trainer restart and durable recovery are not implemented.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future
from collections import deque
import io
import queue
import socket
import struct
import threading
import time

import torch

from .learner import LearnerMetadata
from .state import FragmentMetadata, FragmentSnapshot


MAX_FRAME_BYTES = 32 * 1024 * 1024


def encode_message(kind: str, body: dict) -> bytes:
    if not isinstance(kind, str) or not isinstance(body, dict):
        raise ValueError("wire message requires a string kind and dictionary body")
    stream = io.BytesIO()
    torch.save({"version": 1, "kind": kind, "body": body}, stream)
    payload = stream.getvalue()
    if not 0 < len(payload) <= MAX_FRAME_BYTES:
        raise ValueError("wire message exceeds the frame limit")
    return payload


def decode_message(payload: bytes) -> dict:
    if not 0 < len(payload) <= MAX_FRAME_BYTES:
        raise ValueError("invalid wire frame length")
    message = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if (
        not isinstance(message, dict) or set(message) != {"version", "kind", "body"}
        or type(message["version"]) is not int or message["version"] != 1
        or not isinstance(message["kind"], str) or not isinstance(message["body"], dict)
    ):
        raise ValueError("invalid or unsupported wire message")
    return message


def metadata_from_wire(values: dict) -> LearnerMetadata:
    values = dict(values)
    clock = values.get("syncer_step", 0)
    if type(clock) is not int or clock < 0:
        raise ValueError("metadata syncer_step must be a nonnegative integer")
    values["fragments"] = tuple(FragmentMetadata(**fragment) for fragment in values["fragments"])
    return LearnerMetadata(**values)


def snapshot_to_wire(snapshot: FragmentSnapshot) -> dict:
    return {
        "fragment_id": snapshot.fragment_id, "layout_signature": snapshot.layout_signature,
        "base_revision": snapshot.base_revision, "local_steps": snapshot.local_steps,
        "tokens": snapshot.tokens, "baseline": dict(snapshot.baseline), "current": dict(snapshot.current),
    }


class TransportBackpressure(ConnectionError):
    """A live transport has no free required-message slot."""


class _Outbox:
    """FIFO required frames plus one replaceable, unsent progress report."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = deque()
        self.condition = threading.Condition()
        self.required = self.unfinished_tasks = self.peak = self.coalesced = 0

    def put_nowait(self, item):
        with self.condition:
            if self.required >= self.capacity:
                raise queue.Full
            self.required += 1
            self._append(item, False)

    def progress(self, body):
        with self.condition:
            for index, (_, replaceable) in enumerate(self.items):
                if replaceable:
                    del self.items[index]
                    self.unfinished_tasks -= 1
                    self.coalesced += 1
                    break
            # Append at the tail: newer counters cannot overtake an older ACK.
            self._append(("metadata", body), True)

    def _append(self, item, replaceable):
        self.items.append((item, replaceable))
        self.unfinished_tasks += 1
        self.peak = max(self.peak, len(self.items))
        self.condition.notify_all()

    def get(self, timeout):
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.items:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                self.condition.wait(remaining)
            item, replaceable = self.items.popleft()
            if not replaceable:
                self.required -= 1
            self.condition.notify_all()
            return item

    def task_done(self):
        with self.condition:
            self.unfinished_tasks -= 1
            self.condition.notify_all()


class FramedTransport:
    """Two I/O threads; send/receive on the caller never wait for socket I/O.

    Treat queued bodies/tensors as immutable. Queue limits bound retained frames. send is nonblocking; send_reliable
    waits for capacity off the training thread. Only send_progress coalesces.
    flush is for handshake/shutdown, never for an optimizer-step boundary.
    """

    def __init__(self, connection: socket.socket, *, capacity: int = 32):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("transport capacity must be a positive integer")
        self.socket = connection
        self.socket.settimeout(1.0)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._outgoing = _Outbox(capacity)
        self._stats = {"backpressure_retries": 0, "incoming_full_retries": 0, "encode_seconds": 0., "send_seconds": 0.}
        self._incoming = queue.Queue(maxsize=capacity)
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._error = None
        self.sent_bytes = 0
        self.received_bytes = 0
        self._threads = [threading.Thread(target=self._writer, daemon=True), threading.Thread(target=self._reader, daemon=True)]
        for thread in self._threads:
            thread.start()

    def _fail(self, error):
        with self._lock:
            if self._error is None and not self._closed.is_set():
                self._error = error

    def check(self):
        with self._lock:
            error = self._error
        if error is not None:
            raise ConnectionError(f"TCP transport failed: {error}") from error
        if self._closed.is_set():
            raise ConnectionError("TCP transport is closed")

    def send(self, kind: str, body: dict):
        self.check()
        try:
            self._outgoing.put_nowait((kind, body))
        except queue.Full as exc:
            raise TransportBackpressure("outgoing TCP queue is full") from exc

    def send_progress(self, body):
        self.check()
        self._outgoing.progress(body)

    def send_reliable(self, kind, body, *, deadline=None, progress=None):
        """Retry congestion only; socket failures and deadlines still fail."""
        if deadline is None:
            deadline = time.monotonic() + 60.
        while True:
            try:
                self.send(kind, body)
                return
            except TransportBackpressure:
                self._stats["backpressure_retries"] += 1
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"TCP queue deadline exceeded sending {kind}")
                if progress is not None:
                    progress()
                self._closed.wait(.005)

    def diagnostics(self):
        with self._outgoing.condition:
            return dict(self._stats, sent_bytes=self.sent_bytes,
                        received_bytes=self.received_bytes,
                        outgoing_peak=self._outgoing.peak,
                        progress_coalesced=self._outgoing.coalesced,
                        outgoing_pending=len(self._outgoing.items),
                        incoming_pending=self._incoming.qsize())

    def receive(self):
        # Drain already received messages, including a final stopped message,
        # before surfacing an EOF that followed them.
        try:
            message = self._incoming.get_nowait()
            self._incoming.task_done()
            return message
        except queue.Empty:
            self.check()
            return None

    def _read_exact(self, size):
        data = bytearray()
        while len(data) < size and not self._closed.is_set():
            try:
                chunk = self.socket.recv(size - len(data))
            except socket.timeout:
                continue
            if not chunk:
                raise EOFError("peer disconnected")
            data.extend(chunk)
        if len(data) != size:
            raise EOFError("transport closed during a frame")
        return bytes(data)

    def _reader(self):
        try:
            while not self._closed.is_set():
                header = self._read_exact(4)
                size = struct.unpack("!I", header)[0]
                if not 0 < size <= MAX_FRAME_BYTES:
                    raise ValueError("invalid wire frame length")
                message = decode_message(self._read_exact(size))
                self.received_bytes += 4 + size
                while not self._closed.is_set():
                    try:
                        self._incoming.put(message, timeout=.05)
                        break
                    except queue.Full:
                        self._stats["incoming_full_retries"] += 1
        except Exception as exc:
            self._fail(exc)

    def _writer(self):
        try:
            while not self._closed.is_set():
                try:
                    kind, body = self._outgoing.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    started = time.monotonic()
                    payload = encode_message(kind, body)
                    self._stats["encode_seconds"] += time.monotonic() - started
                    started = time.monotonic()
                    frame = memoryview(struct.pack("!I", len(payload)) + payload)
                    while frame and not self._closed.is_set():
                        try:
                            sent = self.socket.send(frame)
                        except socket.timeout:
                            continue
                        if not sent:
                            raise EOFError("peer disconnected while sending")
                        frame = frame[sent:]
                    self._stats["send_seconds"] += time.monotonic() - started
                    self.sent_bytes += 4 + len(payload)
                except Exception as exc:
                    self._fail(exc)
                    return
                finally:
                    self._outgoing.task_done()
        except Exception as exc:
            self._fail(exc)

    def flush(self, timeout: float = 2.0):
        deadline = time.monotonic() + timeout
        while self._outgoing.unfinished_tasks:
            self.check()
            if time.monotonic() >= deadline:
                raise TimeoutError("TCP send queue did not drain")
            self._closed.wait(0.005)
        self.check()

    def close(self):
        self._closed.set()
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.socket.close()
        for thread in self._threads:
            thread.join(timeout=1.2)


class RemoteLearner:
    """Remote endpoint implementing the syncer's learner interface.

    All endpoint methods/handle run on the syncer control thread. Metadata is
    the latest received report. A broadcast stays pending until its queue ACK
    arrives; final application is verified through separate metadata reports.
    """

    def __init__(self, transport: FramedTransport, metadata: LearnerMetadata):
        self.transport = transport
        self._metadata = metadata
        self.deadline = None
        self._requests: dict[str, Future] = {}
        self._request_arguments = {}
        self._updates = set()
        self.paused = False
        self.stopped = False
        self.training_done = False
        self.pid = None
        self.available = True
        self.failure = None

    def mark_unavailable(self, error):
        """Exclude this endpoint and fail captures without aborting its peers."""
        self.available = False
        self.failure = str(error)
        for future in self._requests.values():
            if not future.done():
                future.set_exception(ConnectionError(self.failure))
        self._updates.clear()

    def reconnect(self, transport, metadata):
        """Attach an authenticated transport; syncer.reconnect_learner catches up."""
        if metadata.learner_id != self._metadata.learner_id or metadata.layout_signature != self._metadata.layout_signature:
            raise ValueError("reconnected learner identity/layout mismatch")
        if self.available:
            raise RuntimeError("cannot replace an available learner connection")
        self.transport.close()
        self.transport = transport
        self._metadata = metadata
        self._requests.clear()
        self._request_arguments.clear()
        self._updates.clear()
        self.paused = self.stopped = self.training_done = False
        self.available, self.failure = True, None

    def metadata(self):
        return self._metadata

    def request_snapshot(self, request_id, fragment_id, expected_revision, *, min_local_steps=1):
        if not self.available:
            raise ConnectionError(self.failure)
        arguments = (fragment_id, expected_revision, min_local_steps)
        if request_id in self._requests:
            if self._request_arguments[request_id] != arguments:
                raise ValueError("a retried request_id must use identical snapshot arguments")
            return self._requests[request_id]
        future = Future()
        self._requests[request_id] = future
        self._request_arguments[request_id] = arguments
        try:
            self.transport.send_reliable("pull", {"request_id": request_id, "fragment_id": fragment_id, "expected_revision": expected_revision, "min_local_steps": min_local_steps}, deadline=self.deadline, progress=self.pump)
        except Exception:
            self._requests.pop(request_id, None)
            self._request_arguments.pop(request_id, None)
            raise
        return future

    def release_snapshot(self, request_id):
        future = self._requests.pop(request_id, None)
        self._request_arguments.pop(request_id, None)
        if future is not None:
            if not future.done():
                future.cancel()
            if self.available:
                self.transport.send_reliable("release", {"request_id": request_id}, deadline=self.deadline, progress=self.pump)
        return future is not None

    def queue_update(self, fragment_id, parameters: Mapping[str, torch.Tensor], revision, *, layout_signature, global_step=0):
        if not self.available:
            raise ConnectionError(self.failure)
        key = (fragment_id, revision)
        if self._metadata.fragments[fragment_id].last_server_revision >= revision:
            self._updates.discard(key)
            return False
        if key not in self._updates:
            self.transport.send_reliable("update", {"fragment_id": fragment_id, "revision": revision, "layout_signature": layout_signature, "parameters": dict(parameters), "global_step": global_step}, deadline=self.deadline, progress=self.pump)
            self._updates.add(key)
        return False  # The syncer retains its outbox until an ACK report arrives.

    def handle(self, message):
        kind, body = message["kind"], message["body"]
        if kind in {"metadata", "update_ack", "paused", "stopped", "training_done"}:
            metadata = metadata_from_wire(body["metadata"])
            if metadata.learner_id != self._metadata.learner_id or metadata.layout_signature != self._metadata.layout_signature:
                raise ValueError("remote metadata identity/layout mismatch")
            self._metadata = metadata
            self._updates = {key for key in self._updates if metadata.fragments[key[0]].last_server_revision < key[1]}
            self.paused |= kind == "paused"
            self.stopped |= kind == "stopped"
            self.training_done |= kind == "training_done"
        elif kind in {"snapshot", "snapshot_error"}:
            future = self._requests.get(body["request_id"])
            if future is None or future.done():
                return  # Released/duplicate captures cannot enter a later cycle.
            if kind == "snapshot_error":
                future.set_exception(RuntimeError(body["error"]))
            else:
                future.set_result(FragmentSnapshot(**body["snapshot"]))
        elif kind == "fatal":
            raise ConnectionError(f"learner {self._metadata.learner_id}: {body['error']}")
        else:
            raise ValueError(f"unexpected learner message: {kind}")

    def pump(self):
        if not self.available:
            return
        try:
            while not self.stopped:
                message = self.transport.receive()
                if message is None:
                    break
                self.handle(message)
        except (ConnectionError, OSError) as exc:
            self.mark_unavailable(exc)
