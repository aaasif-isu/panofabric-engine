import sys

with open('tests/decoupled_heloco/test_flow_control.py', 'r') as f:
    text = f.read()

debug_code = """            frames = []
            for _ in range(3):
                frame = _wait(receiver.receive, 3)
                print("GOT FRAME 1-3:", frame["kind"], file=sys.stderr)
                frames.append(frame)
            sender.send_reliable('stopped', {}, deadline=time.monotonic()+3)
            frame = _wait(receiver.receive, 3)
            print("GOT FRAME 4:", frame["kind"], file=sys.stderr)
            frames.append(frame)"""
            
text = text.replace("            frames = []\n            try:\n                for _ in range(3):\n                    frames.append(_wait(receiver.receive, 3))\n                sender.send_reliable('stopped', {}, deadline=time.monotonic()+3)\n                frames.append(_wait(receiver.receive, 3))\n            except Exception as e:\n                print(f'\\n!!! FAILED AT: len(frames)={len(frames)}, frames={[f[\"kind\"] for f in frames]} \\n', file=sys.stderr)\n                raise", debug_code)

with open('tests/decoupled_heloco/test_flow_control.py', 'w') as f:
    f.write(text)

