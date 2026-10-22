import sys

with open('tests/decoupled_heloco/test_flow_control.py', 'r') as f:
    text = f.read()

old_slow = """        def slow(kind, body):
            is_snap = kind == 'snapshot' and body.get('id') == 0
            is_bulk = kind == 'bulk_begin' and body.get('original_kind') == 'snapshot' and body.get('body_safe', {}).get('id') == 0
            if is_snap or is_bulk:
                entered.set()
                if not resume.wait(3): raise TimeoutError('test writer blocked')
            return original(kind, body)"""
new_slow = """        def slow(kind, body):
            print(f"SLOW CALLED: {kind}, {body}", flush=True)
            is_snap = kind == 'snapshot' and body.get('id') == 0
            is_bulk = kind == 'bulk_begin' and body.get('original_kind') == 'snapshot' and body.get('body_safe', {}).get('id') == 0
            if is_snap or is_bulk:
                print("SLOW MATCHED!", flush=True)
                entered.set()
                if not resume.wait(3): raise TimeoutError('test writer blocked')
            return original(kind, body)"""
text = text.replace(old_slow, new_slow)

with open('tests/decoupled_heloco/test_flow_control.py', 'w') as f:
    f.write(text)
