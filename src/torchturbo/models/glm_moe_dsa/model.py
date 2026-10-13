from typing import Any

try:
    from typing import override  # Python 3.12+
except ImportError:
    from typing_extensions import override  # Older versions

import torch

from torchturbo.runtime import is_npu_available, get_device_name
from .modeling_glm_moe_dsa import GlmMoeDsaForCausalLM
from .configuration_glm_moe_dsa import GlmMoeDsaConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer

from ..base import TurboModel
from ..registry import register_model
from transformers import AutoModelForCausalLM

import transformers

if not hasattr(transformers, "TokenizersBackend"):
    from transformers.models.auto import tokenization_auto as _auto_mod

    _orig_fn = _auto_mod.tokenizer_class_from_name

    from transformers import PreTrainedTokenizerFast

    class TokenizersBackend(PreTrainedTokenizerFast):
        def __init__(self, **kwargs):
            extra = kwargs.get("extra_special_tokens")
            if isinstance(extra, list):
                kwargs.pop("extra_special_tokens")
            super().__init__(**kwargs)

    def _patched(class_name):
        if class_name in ("TokenizersBackend", "TokenizersBackendFast"):
            from transformers import PreTrainedTokenizerFast

            return TokenizersBackend
        return _orig_fn(class_name)

    _auto_mod.tokenizer_class_from_name = _patched


@register_model(
    "glm_moe_dsa", config_cls=GlmMoeDsaConfig, auto_class=AutoModelForCausalLM
)
class TurboGlmMoeDsaForCausalLM(TurboModel, GlmMoeDsaForCausalLM):

    def __init__(self, config: GlmMoeDsaConfig) -> None:
        self._pre_init_patches(config)
        super().__init__(config)
        self._post_init_patches()

    @classmethod
    def get_default_parallel_plan(cls, **kwargs):
        """Return the default parallel plan for this model."""
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(**kwargs)

    @staticmethod
    def _pre_init_patches(config):
        from torchturbo.ops import ops
        import torchturbo.models.glm_moe_dsa.modeling_glm_moe_dsa as glm5_model

        def rms_norm_forward(self, hidden_states):
            return ops.rms_norm(hidden_states, self.weight, self.variance_epsilon)

        glm5_model.GlmMoeDsaRMSNorm.forward = rms_norm_forward

        # patch moe_layer forward
        origin_moe_forward = MoELayer.forward

        def deepseek_moe_forward(self, hidden_states: torch.Tensor):
            output, _ = origin_moe_forward(self, hidden_states)
            return output

        MoELayer.forward = deepseek_moe_forward

        # code path of model creation with sparse attn is same as eager mode
        config._attn_implementation = "eager"

    def _post_init_patches(self):
        # code path of model forward with sparse attn is same as FA
        self.model._use_flash_attention_2 = True
        self.config._attn_implementation = "flash_attention_2"

    def on_parallelized(self, parallel_context):
        sp_info = parallel_context["sp"]
        sp_group = sp_info["sp_group"]
        sp_size = sp_info["sp_size"]
        self.model._sp_size = sp_size

        if is_npu_available():
            from .npu import npu_fused_dsa_attn_forward
            from .modeling_glm_moe_dsa import GlmMoeDsaAttention

            def _patched_forward(self, *args, **kwargs):
                return npu_fused_dsa_attn_forward(
                    self, *args, sp_group=sp_group, sp_size=sp_size, **kwargs
                )

            GlmMoeDsaAttention.forward = _patched_forward
