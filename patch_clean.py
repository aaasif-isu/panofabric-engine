import sys

with open('tests/decoupled_heloco/test_flow_control.py', 'r') as f:
    text = f.read()

text = text.replace(
    "frames = []\n            for _ in range(3):\n                f = _wait(receiver.receive, 3)\n                print('GOT FRAME:', f['kind'])\n                frames.append(f)",
    "frames = [_wait(receiver.receive, 3) for _ in range(3)]"
)

text = text.replace(
    "f2 = _wait(receiver.receive, 3)\n            print('GOT FRAME:', f2['kind'])\n            frames.append(f2)",
    "frames.append(_wait(receiver.receive, 3))"
)

with open('tests/decoupled_heloco/test_flow_control.py', 'w') as f:
    f.write(text)

