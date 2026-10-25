import sys

with open('tests/decoupled_heloco/test_transport.py', 'r') as f:
    text = f.read()

old_test = """    def test_actual_tcp_frame_roundtrip_and_wire_byte_counters(self):
        parent, worker = self.pair()
        parent.send("update", {"revision": 3, "tensor": torch.tensor([1.0, 2.0])})
        message = _wait(worker.receive)
        self.assertEqual(message["body"]["revision"], 3)
        torch.testing.assert_close(message["body"]["tensor"], torch.tensor([1.0, 2.0]))
        parent.flush()
        self.assertGreater(parent.sent_bytes, 4)
        self.assertEqual(parent.sent_bytes, worker.received_bytes)"""

new_test = """    def test_actual_tcp_frame_roundtrip_and_wire_byte_counters(self):
        parent, worker = self.pair()
        parent.send("test_frame", {"revision": 3, "tensor": torch.tensor([1.0, 2.0])})
        message = _wait(worker.receive)
        self.assertEqual(message["body"]["revision"], 3)
        torch.testing.assert_close(message["body"]["tensor"], torch.tensor([1.0, 2.0]))
        parent.flush()
        self.assertGreater(parent.sent_bytes, 4)
        self.assertEqual(parent.sent_bytes, worker.received_bytes)"""

if old_test in text:
    with open('tests/decoupled_heloco/test_transport.py', 'w') as f:
        f.write(text.replace(old_test, new_test))
    print("Patched test_tcp")
else:
    print("Could not find old test")
