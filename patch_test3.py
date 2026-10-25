import sys

with open('tests/decoupled_heloco/test_flow_control.py', 'r') as f:
    text = f.read()

old_paused = "sender.send_reliable('paused', {}, deadline=time.monotonic()+.03)"
new_paused = """body2 = {
                    "request_id": "99",
                    "snapshot": {
                        "fragment_id": 0, "layout_signature": "sig", "base_revision": 0,
                        "local_steps": 1, "tokens": 0, "pseudo_gradient": {}
                    }
                }
                sender.send_reliable('snapshot', body2, deadline=time.monotonic()+.03)"""

text = text.replace(old_paused, new_paused)

with open('tests/decoupled_heloco/test_flow_control.py', 'w') as f:
    f.write(text)

