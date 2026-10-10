"""Bounded, threaded localhost TCP messages and remote learner endpoints.

    Wire messages contain basic dictionaries and detached CPU tensors, never
    futures/dataclasses or references to live model storage. torch.load uses
    weights_only=True. This prototype has no reconnect or durable recovery.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future
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
    values["fragments"] = tuple(FragmentMetadata(**fragment) for fragment in values["fragments"])
    return LearnerMetadata(**values)


def snapshot_to_wire(snapshot: FragmentSnapshot) -> dict:
    return {
        "fragment_id": snapshot.fragment_id, "layout_signature": snapshot.layout_signature,
        "base_revision": snapshot.base_revision, "local_steps": snapshot.local_steps,
        "tokens": snapshot.tokens, "baseline": dict(snapshot.baseline), "current": dict(snapshot.current),
    }


class FramedTransport:
    """Two I/O threads; send/receive on the caller never wait for socket I/O.

    Treat queued bodies/tensors as immutable. Queue limits bound retained
    frames; saturation raises an error rather than silently dropping work.
    flush is for handshake/shutdown, never for an optimizer-step boundary.
    """

    def __init__(self, connection: socket.socket, *, capacity: int = 32):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("transport capacity must be a positive integer")
        self.socket = connection
        self.socket.settimeout(1.0)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._outgoing = queue.Queue(maxsize=capacity)
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
            raise ConnectionError("outgoing TCP queue is full") from exc

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
                self._incoming.put(message, timeout=1.0)
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
                    payload = encode_message(kind, body)
                    self.socket.sendall(struct.pack("!I", len(payload)) + payload)
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
    """Nonblocking endpoint implementing the syncer's learner interface.

    All endpoint methods/handle run on the syncer control thread. Metadata is
    the latest received report. A broadcast stays pending until its queue ACK
    arrives; final application is verified through separate metadata reports.
    """

    def __init__(self, transport: FramedTransport, metadata: LearnerMetadata):
        self.transport = transport
        self._metadata = metadata
        self._requests: dict[str, Future] = {}
        self._request_arguments = {}
        self._updates = set()
        self.paused = False
        self.stopped = False
        self.training_done = False
        self.pid = None

    def metadata(self):
        return self._metadata

    def request_snapshot(self, request_id, fragment_id, expected_revision, *, min_local_steps=1):
        arguments = (fragment_id, expected_revision, min_local_steps)
        if request_id in self._requests:
            if self._request_arguments[request_id] != arguments:
                raise ValueError("a retried request_id must use identical snapshot arguments")
            return self._requests[request_id]
        self.transport.send("pull", {"request_id": request_id, "fragment_id": fragment_id, "expected_revision": expected_revision, "min_local_steps": min_local_steps})
        future = Future()
        self._requests[request_id] = future
        self._request_arguments[request_id] = arguments
        return future

    def release_snapshot(self, request_id):
        future = self._requests.pop(request_id, None)
        self._request_arguments.pop(request_id, None)
        if future is not None:
            if not future.done():
                future.cancel()
            self.transport.send("release", {"request_id": request_id})
        return future is not None

    def queue_update(self, fragment_id, parameters: Mapping[str, torch.Tensor], revision, *, layout_signature):
        key = (fragment_id, revision)
        if self._metadata.fragments[fragment_id].last_server_revision >= revision:
            self._updates.discard(key)
            return False
        if key not in self._updates:
            self.transport.send("update", {"fragment_id": fragment_id, "revision": revision, "layout_signature": layout_signature, "parameters": dict(parameters)})
            self._updates.add(key)
        return False  # The syncer retains its outbox until an ACK report arrives.

    def handle(self, message):
        kind, body = message["kind"], message["body"]
        if kind in {"metadata", "update_ack", "paused", "stopped", "training_done"}:
            metadata = metadata_from_wire(body["metadata"])
            if metadata.learner_id != self._metadata.learner_id or metadata.layout_signature != self._metadata.layout_signature:
                raise ValueError("remote metadata identity/layout mismatch")
            self._metadata = metadata
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
            raise RuntimeError(f"learner {self._metadata.learner_id}: {body['error']}")
        else:
            raise ValueError(f"unexpected learner message: {kind}")

    def pump(self):
        while not self.stopped:
            message = self.transport.receive()
            if message is None:
                break
            self.handle(message)
