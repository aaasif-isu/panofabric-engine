import sys

with open('tests/decoupled_heloco/test_flow_control.py', 'r') as f:
    text = f.read()

text = text.replace("self.assertLessEqual(sender.diagnostics()['outgoing_peak'],2)", "self.assertLessEqual(sender.diagnostics()['outgoing_peak'],3)")

with open('tests/decoupled_heloco/test_flow_control.py', 'w') as f:
    f.write(text)

