import sys

with open('tests/decoupled_heloco/test_flow_control.py', 'r') as f:
    text = f.read()

text = text.replace("except Exception as e:", "except Exception as e:\n                print(f'\\n!!! FAILED AT: len(frames)={len(frames)}, frames={[f[\"kind\"] for f in frames]} \\n', file=sys.stderr)")

with open('tests/decoupled_heloco/test_flow_control.py', 'w') as f:
    f.write(text)

