"""Wire isolation, real TCP frames, acknowledgements, and stale responses."""

from dataclasses import asdict
import io
import socket
import struct
import threading
import time
import unittest
from unittest.mock import patch

import torch

from panoengine.decentralized.decoupled_heloco.learner import DecoupledLearner
from panoengine.decentralized.decoupled_heloco.transport import (
    MAX_FRAME_BYTES, FramedTransport, RemoteLearner, decode_message,
    encode_message, metadata_from_wire, snapshot_to_wire,
)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


def _wait(operation, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = operation()
        if result is not None:
            return result
        time.sleep(0.002)
    raise AssertionError("TCP test operation timed out")


class TransportTests(unittest.TestCase):
    def pair(self, *, capacity=32):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        client = socket.create_connection(listener.getsockname(), timeout=2.0)
        server, _ = listener.accept()
        listener.close()
        transports = (FramedTransport(client, capacity=capacity), FramedTransport(server, capacity=capacity))
        for transport in transports:
            self.addCleanup(transport.close)
        return transports

    def make(self):
        self.model = _Model()
        self.learner = DecoupledLearner(self.model, 1, learner_id=0)
        self.parent, self.worker = self.pair()
        self.remote = RemoteLearner(self.parent, self.learner.metadata())

    def pump_until(self, predicate):
        def poll():
            self.remote.pump()
            return True if predicate() else None
        _wait(poll)

    def test_snapshot_serialization_creates_independent_tensors_and_preserves_metadata(self):
        self.make()
        future = self.learner.request_snapshot("0", 0, 0)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        with self.learner.training_step(8):
            optimizer.zero_grad()
            self.model.weight.sum().backward()
            optimizer.step()
        snapshot = future.result()
        decoded = decode_message(encode_message("snapshot", snapshot_to_wire(snapshot)))["body"]
        torch.testing.assert_close(decoded["baseline"]["weight"], snapshot.baseline["weight"], rtol=0, atol=0)
        decoded["current"]["weight"].zero_()
        torch.testing.assert_close(snapshot.current["weight"], torch.tensor([0.9]))
        self.assertEqual(metadata_from_wire(asdict(self.learner.metadata())), self.learner.metadata())

    def test_actual_tcp_frame_roundtrip_and_wire_byte_counters(self):
        parent, worker = self.pair()
        parent.send("update", {"revision": 3, "tensor": torch.tensor([1.0, 2.0])})
        message = _wait(worker.receive)
        self.assertEqual(message["body"]["revision"], 3)
        torch.testing.assert_close(message["body"]["tensor"], torch.tensor([1.0, 2.0]))
        parent.flush()
        self.assertGreater(parent.sent_bytes, 4)
        self.assertEqual(parent.sent_bytes, worker.received_bytes)

    def test_empty_and_unsupported_protocol_frames_are_rejected(self):
        with self.assertRaises(ValueError):
            decode_message(b"")
        stream = io.BytesIO()
        torch.save({"version": 99, "kind": "metadata", "body": {}}, stream)
        with self.assertRaises(ValueError):
            decode_message(stream.getvalue())
        parent, worker = self.pair()
        parent.socket.sendall(struct.pack("!I", MAX_FRAME_BYTES + 1))
        with self.assertRaises(ConnectionError):
            _wait(worker.receive)

    def test_disconnect_surfaces_without_a_blocking_receive(self):
        parent, worker = self.pair()
        parent.close()
        with self.assertRaises(ConnectionError):
            _wait(worker.receive)

    def test_saturated_outgoing_queue_raises_instead_of_blocking_the_caller(self):
        gate = threading.Event()
        with patch.object(FramedTransport, "_writer", lambda _: gate.wait(2.0)):
            parent, _ = self.pair(capacity=1)
        try:
            parent.send("metadata", {})
            with self.assertRaisesRegex(ConnectionError, "queue is full"):
                parent.send("metadata", {})
        finally:
            gate.set()

    def test_remote_pull_captures_at_boundary_and_ignores_released_duplicate(self):
        self.make()
        future = self.remote.request_snapshot("0", 0, 0)
        self.assertIs(self.remote.request_snapshot("0", 0, 0), future)
        with self.assertRaises(ValueError):
            self.remote.request_snapshot("0", 0, 1)
        command = _wait(self.worker.receive)["body"]
        local = self.learner.request_snapshot(command["request_id"], command["fragment_id"], command["expected_revision"], min_local_steps=command["min_local_steps"])
        self.assertFalse(local.done())
        with self.learner.training_step(8):
            with torch.no_grad():
                self.model.weight.sub_(0.1)
        body = {"request_id": "0", "snapshot": snapshot_to_wire(local.result())}
        self.worker.send("snapshot", body)
        self.pump_until(future.done)
        torch.testing.assert_close(future.result().pseudo_gradient()["weight"], torch.tensor([0.1]))
        self.assertTrue(self.remote.release_snapshot("0"))
        release = _wait(self.worker.receive)
        self.assertEqual(release["kind"], "release")
        self.learner.release_snapshot(release["body"]["request_id"])
        self.remote.handle({"kind": "snapshot", "body": body})
        self.assertEqual(self.remote._requests, {})

    def test_remote_broadcast_remains_pending_until_queue_ack_and_applies_at_boundary(self):
        self.make()
        signature = self.learner.manager.layout_signature
        update = {"weight": torch.tensor([0.5])}
        self.assertFalse(self.remote.queue_update(0, update, 1, layout_signature=signature))
        command = _wait(self.worker.receive)["body"]
        self.learner.queue_update(command["fragment_id"], command["parameters"], command["revision"], layout_signature=command["layout_signature"])
        torch.testing.assert_close(self.model.weight, torch.ones(1))
        self.assertEqual(self.remote.metadata().fragments[0].last_server_revision, 0)
        self.worker.send("update_ack", {"metadata": asdict(self.learner.metadata())})
        self.pump_until(lambda: self.remote.metadata().fragments[0].last_server_revision == 1)
        self.assertFalse(self.remote.queue_update(0, update, 1, layout_signature=signature))
        self.assertIsNone(self.worker.receive())  # No second update is sent.
        self.assertEqual(self.remote.metadata().fragments[0].last_applied_revision, 0)
        self.learner.boundary()
        torch.testing.assert_close(self.model.weight, torch.tensor([0.5]))
        self.worker.send("metadata", {"metadata": asdict(self.learner.metadata())})
        self.pump_until(lambda: self.remote.metadata().fragments[0].last_applied_revision == 1)

    def test_remote_metadata_identity_mismatch_is_rejected(self):
        self.make()
        values = asdict(self.learner.metadata())
        values["learner_id"] = 1
        with self.assertRaises(ValueError):
            self.remote.handle({"kind": "metadata", "body": {"metadata": values}})
        self.assertEqual(self.remote.metadata().learner_id, 0)


if __name__ == "__main__":
    unittest.main()
