import sys

with open('tests/decoupled_heloco/test_flow_control.py', 'r') as f:
    text = f.read()

old_order = """        self.assertEqual([out.get(.1) for _ in range(3)], [
            ('snapshot', {'id': 1}), ('update_ack', {'id': 2}),
            ('metadata', {'steps': 500})])"""

new_order = """        self.assertEqual([out.get(.1) for _ in range(3)], [
            ('metadata', {'steps': 500}), ('snapshot', {'id': 1}),
            ('update_ack', {'id': 2})])"""

text = text.replace(old_order, new_order)

with open('tests/decoupled_heloco/test_flow_control.py', 'w') as f:
    f.write(text)

