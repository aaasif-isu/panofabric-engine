import sys
import re

with open('panoengine/decentralized/decoupled_heloco/gpu_recipe.py', 'r') as f:
    text = f.read()

text = re.sub(r'    class _HfConfigWrapper:.*?return cfg', '    cfg.model_spec.model.update_from_config(config=cfg)\n    return cfg', text, flags=re.DOTALL)

with open('panoengine/decentralized/decoupled_heloco/gpu_recipe.py', 'w') as f:
    f.write(text)

print("Removed wrapper!")
