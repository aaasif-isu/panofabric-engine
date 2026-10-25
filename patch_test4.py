import sys

with open('tests/decoupled_heloco/test_flow_control.py', 'r') as f:
    text = f.read()

old_code = """            sender.send('update_ack', {'id': 1})
            for step in range(500): sender.send_progress({'steps': step+1, 'tokens': (step+1)*512})
            with self.assertRaises(TimeoutError):
                body2 = {
                    "request_id": "99",
                    "snapshot": {
                        "fragment_id": 0, "layout_signature": "sig", "base_revision": 0,
                        "local_steps": 1, "tokens": 0, "pseudo_gradient": {}
                    }
                }
                sender.send_reliable('snapshot', body2, deadline=time.monotonic()+.03)"""

new_code = """            sender.send('update_ack', {'id': 1})
            for step in range(500): sender.send_progress({'steps': step+1, 'tokens': (step+1)*512})
            body_fill = {
                "request_id": "88",
                "snapshot": {
                    "fragment_id": 0, "layout_signature": "sig", "base_revision": 0,
                    "local_steps": 1, "tokens": 0, "pseudo_gradient": {}
                }
            }
            sender.send('snapshot', body_fill)  # Fills the bulk capacity
            with self.assertRaises(TimeoutError):
                body2 = {
                    "request_id": "99",
                    "snapshot": {
                        "fragment_id": 0, "layout_signature": "sig", "base_revision": 0,
                        "local_steps": 1, "tokens": 0, "pseudo_gradient": {}
                    }
                }
                sender.send_reliable('snapshot', body2, deadline=time.monotonic()+.03)"""

text = text.replace(old_code, new_code)

# We also need to expect the filler snapshot to arrive at the receiver
old_assert = "self.assertEqual([x['kind'] for x in frames], ['snapshot', 'update_ack', 'metadata', 'stopped'])"
new_assert = """frames.append(_wait(receiver.receive, 3))
        self.assertEqual([x['kind'] for x in frames], ['snapshot', 'update_ack', 'metadata', 'snapshot', 'stopped'])"""

text = text.replace(old_assert, new_assert)

with open('tests/decoupled_heloco/test_flow_control.py', 'w') as f:
    f.write(text)

