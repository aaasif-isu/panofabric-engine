import sys

with open('panoengine/decentralized/decoupled_heloco/process_smoke.py', 'r') as f:
    text = f.read()

old_learner_proc = """def _learner_process(address, token, learner_id, config, timeout):
    transport = None
    try:
        torch.set_num_threads(1)
        torch.manual_seed(config.run.get("seed", 42))"""
new_learner_proc = """def _learner_process(address, token, learner_id, config, timeout):
    transport = None
    try:
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(config.run.get("seed", 42))"""

text = text.replace(old_learner_proc, new_learner_proc)

with open('panoengine/decentralized/decoupled_heloco/process_smoke.py', 'w') as f:
    f.write(text)

