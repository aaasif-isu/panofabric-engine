import sys

with open('panoengine/decentralized/decoupled_heloco/gpu_recipe.py', 'r') as f:
    text = f.read()

old_config_init = """    # Construct base Trainer explicitly: the selected registry normally owns
    # a FaultTolerantTrainer.Config, whose build() would enter the old FT plane.
    cfg = Trainer.Config(**{field.name: getattr(recipe, field.name) for field in fields(Trainer.Config)})"""

new_config_init = """    # Construct base Trainer explicitly: the selected registry normally owns
    # a FaultTolerantTrainer.Config, whose build() would enter the old FT plane.
    from dataclasses import dataclass
    @dataclass(kw_only=True)
    class _DecoupledConfig(Trainer.Config):
        hf_model: str = ""
    
    cfg = _DecoupledConfig(**{field.name: getattr(recipe, field.name) for field in fields(Trainer.Config)})
    cfg.hf_model = getattr(recipe, "hf_model", "")"""

old_wrapper = """    class _HfConfigWrapper:
        def __init__(self, c, h):
            self._cfg = c
            self.hf_model = h
        def __getattr__(self, name):
            return getattr(self._cfg, name)

    cfg.model_spec.model.update_from_config(config=_HfConfigWrapper(cfg, getattr(recipe, 'hf_model', '')))"""

new_wrapper = """    cfg.model_spec.model.update_from_config(config=cfg)"""

text = text.replace(old_config_init, new_config_init)
text = text.replace(old_wrapper, new_wrapper)

with open('panoengine/decentralized/decoupled_heloco/gpu_recipe.py', 'w') as f:
    f.write(text)

print("Patched decoupled config!")
