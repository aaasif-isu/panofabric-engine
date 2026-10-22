import sys

with open('tests/decoupled_heloco/test_flow_control.py', 'r') as f:
    text = f.read()

old_slow = """        def slow(kind, body):
            if kind == 'snapshot' and body.get('id') == 0:"""
new_slow = """        def slow(kind, body):
            if kind in ('snapshot', 'snapshot_begin') and body.get('id', body.get('request_id')) == 0:"""
text = text.replace(old_slow, new_slow)

with open('tests/decoupled_heloco/test_flow_control.py', 'w') as f:
    f.write(text)
