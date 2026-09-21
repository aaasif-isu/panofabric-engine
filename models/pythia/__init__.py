import transformers
from transformers import GPTNeoXForCausalLM

from torchtitan.experiments.transformers_modeling_backend import (
    HFTransformerModel,
    TitanModelConfig,
    parallelize_hf_transformers,
    pipeline_hf_transformers,
)
from torchtitan.experiments.transformers_modeling_backend.state_dict_adapter import (
    HFTransformerStateDictAdapter,
)
from torchtitan.experiments.torchft.config.job_config import (
    FaultTolerantModelSpec,
)


class _GPTNeoXBackboneProxy:
    """Expose GPT-NeoX using the Llama-like names expected by TorchTitan's
    HF backend, without registering duplicate PyTorch modules."""

    def __init__(self, backbone):
        object.__setattr__(self, "_backbone", backbone)

    @property
    def layers(self):
        return self._backbone.layers

    @layers.setter
    def layers(self, value):
        self._backbone.layers = value

    @property
    def embed_tokens(self):
        return self._backbone.embed_in

    @embed_tokens.setter
    def embed_tokens(self, value):
        self._backbone.embed_in = value

    @property
    def norm(self):
        return self._backbone.final_layer_norm

    @norm.setter
    def norm(self, value):
        self._backbone.final_layer_norm = value

    @property
    def rotary_emb(self):
        return self._backbone.rotary_emb

    @rotary_emb.setter
    def rotary_emb(self, value):
        self._backbone.rotary_emb = value

    def __call__(self, *args, **kwargs):
        return self._backbone(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._backbone, name)


class PanoPythiaForCausalLM(GPTNeoXForCausalLM):
    """GPT-NeoX with compatibility aliases for TorchTitan's HF backend."""

    def __init__(self, config):
        super().__init__(config)

        # Pythia/GPT-NeoX initialization.
        self.config.initializer_range = 0.02

    @property
    def model(self):
        return _GPTNeoXBackboneProxy(self.gpt_neox)

    @property
    def lm_head(self):
        return self.embed_out

    @lm_head.setter
    def lm_head(self, value):
        self.embed_out = value


# HFTransformerModel dynamically looks up architectures in `transformers`.
transformers.PanoPythiaForCausalLM = PanoPythiaForCausalLM


def model_registry(
    flavor: str,
    hf_assets_path: str | None = None,
) -> FaultTolerantModelSpec:
    del hf_assets_path

    # No architecture dimensions here.
    # They come from each flavor's real HF config.json.
    cfg = HFTransformerModel.Config(TitanModelConfig())

    # Force our structural compatibility wrapper after HF config loading.
    cfg._titan_injected_model_args["architectures"] = [
        "PanoPythiaForCausalLM"
    ]
    cfg._titan_injected_model_args["tie_word_embeddings"] = False

    return FaultTolerantModelSpec(
        name="ft/pythia",
        flavor=flavor,
        model=cfg,
        parallelize_fn=parallelize_hf_transformers,
        pipelining_fn=pipeline_hf_transformers,
        post_optimizer_build_fn=None,
        state_dict_adapter=HFTransformerStateDictAdapter,
        fragment_fn=None,
    )
