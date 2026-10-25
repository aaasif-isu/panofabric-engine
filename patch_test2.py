import sys

with open('tests/decoupled_heloco/test_flow_control.py', 'r') as f:
    text = f.read()

# 1. test_slow_reader_survives_more_than_old_timeout
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

# 2. test_required_retry_and_latest_counters
old_required_retry = """        def slow(kind, body):
            if kind == 'snapshot' and body.get('id') == 0:
                entered.set()
                if not resume.wait(3): raise TimeoutError('test writer blocked')
            return original(kind, body)
        with patch('panoengine.decentralized.decoupled_heloco.transport.encode_message', slow):
            sender.send('snapshot', {'id': 0})"""

new_required_retry = """        def slow(kind, body):
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

# 3. test_large_frames_retry_partial_socket_sends
old_large_frames = """                for number in range(5):
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

new_large_frames = """                for number in range(5):
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

text = text.replace(old_slow_reader, new_slow_reader)
text = text.replace(old_required_retry, new_required_retry)
text = text.replace(old_large_frames, new_large_frames)

with open('tests/decoupled_heloco/test_flow_control.py', 'w') as f:
    f.write(text)

