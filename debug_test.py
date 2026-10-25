import sys

with open('tests/decoupled_heloco/test_flow_control.py', 'r') as f:
    text = f.read()

debug_code = """            frames = []
            try:
                for _ in range(3):
                    frames.append(_wait(receiver.receive, 3))
                sender.send_reliable('stopped', {}, deadline=time.monotonic()+3)
                frames.append(_wait(receiver.receive, 3))
            except Exception as e:
                print(f"FAILED AT: len(frames)={len(frames)}, frames={[f['kind'] for f in frames]}")
                raise"""
                
text = text.replace("            frames = [_wait(receiver.receive, 3) for _ in range(3)]\n            sender.send_reliable('stopped', {}, deadline=time.monotonic()+3)\n            frames.append(_wait(receiver.receive, 3))", debug_code)

with open('tests/decoupled_heloco/test_flow_control.py', 'w') as f:
    f.write(text)

