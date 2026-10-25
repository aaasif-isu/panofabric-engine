import sys

with open('panoengine/decentralized/decoupled_heloco/gpu_recipe.py', 'r') as f:
    text = f.read()

old_code = """    cfg.model_spec.model.update_from_config(config=cfg)
    return cfg"""

new_code = """    class _HfConfigWrapper:
        def __init__(self, c, h):
            self._cfg = c
            self.hf_model = h
        def __getattr__(self, name):
            return getattr(self._cfg, name)
    
    cfg.model_spec.model.update_from_config(config=_HfConfigWrapper(cfg, getattr(recipe, 'hf_model', '')))
    return cfg"""

if old_code in text:
    text = text.replace(old_code, new_code)
    with open('panoengine/decentralized/decoupled_heloco/gpu_recipe.py', 'w') as f:
        f.write(text)
    print("Patched build_recipe!")
else:
    print("Could not find code!")
