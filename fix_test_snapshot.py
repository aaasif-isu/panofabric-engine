import sys

with open('tests/decoupled_heloco/test_transport.py', 'r') as f:
    text = f.read()

old_test = """        payload, tensors = encode_message("snapshot", snapshot_to_wire(snapshot))
        decoded = decode_message(payload, tensors)["body"]"""

new_test = """        payload, tensors = encode_message("snapshot", snapshot_to_wire(snapshot))
        tensors = [t.clone() for t in tensors]
        decoded = decode_message(payload, tensors)["body"]"""

if old_test in text:
    with open('tests/decoupled_heloco/test_transport.py', 'w') as f:
        f.write(text.replace(old_test, new_test))
    print("Patched test_snapshot")
else:
    print("Could not find old test_snapshot")
