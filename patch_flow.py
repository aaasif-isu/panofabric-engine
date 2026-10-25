import sys
import re

with open('tests/decoupled_heloco/test_flow_control.py', 'r') as f:
    text = f.read()

# 1. test_progress_order_and_bound
text = text.replace(
    "('snapshot', {'id': 1}), ('update_ack', {'id': 2}),\n            ('metadata', {'steps': 500})",
    "('metadata', {'steps': 500}), ('snapshot', {'id': 1}),\n            ('update_ack', {'id': 2})"
)

# 2. test_slow_reader_survives_more_than_old_timeout
old_slow_reader = """        for number in range(3):
            sender.send_reliable('snapshot', {'id': number}, deadline=time.monotonic()+3)
        # Old incoming.put(timeout=1) killed the reader here.
        time.sleep(1.2)
        received = [_wait(receiver.receive, 3)['body']['id'] for _ in range(3)]"""
new_slow_reader = """        for number in range(3):
            body = {
                "request_id": str(number),
                "snapshot": {
                    "fragment_id": 0, "layout_signature": "sig", "base_revision": 0,
                    "local_steps": 1, "tokens": 0, "pseudo_gradient": {}
                }
            }
            sender.send_reliable('snapshot', body, deadline=time.monotonic()+3)
        # Old incoming.put(timeout=1) killed the reader here.
        time.sleep(1.2)
        received = [int(_wait(receiver.receive, 3)['body']['request_id']) for _ in range(3)]"""
text = text.replace(old_slow_reader, new_slow_reader)

# 3. test_required_retry_and_latest_counters
old_req = """        def slow(kind, body):
            if kind == 'snapshot' and body.get('id') == 0:
                entered.set()
                if not resume.wait(3): raise TimeoutError('test writer blocked')
            return original(kind, body)
        with patch('panoengine.decentralized.decoupled_heloco.transport.encode_message', slow):
            sender.send('snapshot', {'id': 0})"""
new_req = """        def slow(kind, body):
            if kind == 'snapshot_begin':
                entered.set()
                if not resume.wait(3): raise TimeoutError('test writer blocked')
            return original(kind, body)
        with patch('panoengine.decentralized.decoupled_heloco.transport.encode_message', slow):
            body = {
                "request_id": "0",
                "snapshot": {
                    "fragment_id": 0, "layout_signature": "sig", "base_revision": 0,
                    "local_steps": 1, "tokens": 0, "pseudo_gradient": {}
                }
            }
            sender.send('snapshot', body)"""
text = text.replace(old_req, new_req)

# 4. test_large_frames_retry_partial_socket_sends
old_large = """                for number in range(5):
                    sender.send_reliable('snapshot', {'id': number, 'tensor': tensor}, deadline=time.monotonic()+10)
                sender.flush(10)
            except Exception as exc:
                failures.append(exc)
        thread = threading.Thread(target=produce)
        thread.start()
        # Fill the receiver queue and TCP buffers; exceed socket timeout.
        time.sleep(1.3)
        try:
            for number in range(5):
                frame = _wait(receiver.receive, 5)
                self.assertEqual(frame['body']['id'], number)
                self.assertTrue(torch.equal(frame['body']['tensor'], tensor))"""
new_large = """                for number in range(5):
                    body = {
                        "request_id": str(number),
                        "snapshot": {
                            "fragment_id": 0, "layout_signature": "sig", "base_revision": 0,
                            "local_steps": 1, "tokens": 0, "pseudo_gradient": {"tensor": tensor}
                        }
                    }
                    sender.send_reliable('snapshot', body, deadline=time.monotonic()+10)
                sender.flush(10)
            except Exception as exc:
                failures.append(exc)
        thread = threading.Thread(target=produce)
        thread.start()
        # Fill the receiver queue and TCP buffers; exceed socket timeout.
        time.sleep(1.3)
        try:
            for number in range(5):
                frame = _wait(receiver.receive, 5)
                self.assertEqual(int(frame['body']['request_id']), number)
                self.assertTrue(torch.equal(frame['body']['snapshot']['pseudo_gradient']['tensor'], tensor))"""
text = text.replace(old_large, new_large)

# Now, we also need to output some debug info if it fails
text = text.replace(
    "frames = [_wait(receiver.receive, 3) for _ in range(3)]",
    "frames = []\n            for _ in range(3):\n                f = _wait(receiver.receive, 3)\n                print('GOT FRAME:', f['kind'])\n                frames.append(f)"
)
text = text.replace(
    "frames.append(_wait(receiver.receive, 3))",
    "f2 = _wait(receiver.receive, 3)\n            print('GOT FRAME:', f2['kind'])\n            frames.append(f2)"
)

with open('tests/decoupled_heloco/test_flow_control.py', 'w') as f:
    f.write(text)

