from dataclasses import dataclass
@dataclass
class X:
    seq_len: int = 4
    
class HfConfigWrapper:
    def __init__(self, cfg, hf_model):
        self._cfg = cfg
        self.hf_model = hf_model
    def __getattr__(self, name):
        return getattr(self._cfg, name)

cfg = X()
wrapper = HfConfigWrapper(cfg, "qwen")
print(wrapper.seq_len)
print(wrapper.hf_model)
