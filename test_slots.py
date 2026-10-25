from dataclasses import dataclass, fields
from torchtitan.trainer import Trainer

@dataclass(kw_only=True)
class _DecoupledConfig(Trainer.Config):
    hf_model: str = ""

recipe = Trainer.Config()
cfg = _DecoupledConfig(**{field.name: getattr(recipe, field.name) for field in fields(Trainer.Config)})
cfg.hf_model = "my_model"

print(hasattr(cfg, '__slots__'))
print(cfg.hf_model)
