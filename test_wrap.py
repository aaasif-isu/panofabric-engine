from dataclasses import dataclass, fields
from torchtitan.trainer import Trainer

@dataclass
class DummyConfig:
    training: int = 1
    hf_model: str = "qwen"

recipe = DummyConfig()

# If we just do this:
try:
    cfg = Trainer.Config(**{field.name: getattr(recipe, field.name, None) for field in fields(Trainer.Config)})
    print(cfg)
except Exception as e:
    print("Error:", e)

class MockWrapper:
    def __init__(self, cfg, hf_model):
        self._cfg = cfg
        self.hf_model = hf_model
    def __getattr__(self, name):
        return getattr(self._cfg, name)

w = MockWrapper(cfg, "test_hf")
print("Wrapped:", w.training, w.hf_model)
