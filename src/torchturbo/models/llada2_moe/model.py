from transformers import AutoModelForCausalLM

from .configuration_llada2_moe import LLaDA2MoeConfig
from .modeling_llada2_moe import LLaDA2MoeModelLM
from ..base import TurboModel
from ..registry import register_model
from ..weight_converter import BaseWeightConverter
from torchturbo.runtime import get_device_runtime


class LLaDA2WeightConverter(BaseWeightConverter):
    def get_hf_gate_bias_name(self):
        return "gate.expert_bias"

    def is_expert(self, name):
        return self.get_experts_name() in name and "shared_experts" not in name


@register_model("llada2_moe", config_cls=LLaDA2MoeConfig, auto_class=AutoModelForCausalLM)
class TurboLLaDA2MoeForCausalLM(TurboModel, LLaDA2MoeModelLM):
    config_class = LLaDA2MoeConfig
    base_model_prefix = "model"
    is_diffusion_model = True

    def __init__(self, config: LLaDA2MoeConfig) -> None:
        self._pre_init_patches(config)
        # Block diffusion requires a 4D attention mask that FA2 cannot represent.
        config._attn_implementation = "sdpa"
        super().__init__(config)
        self._post_init_patches()

    @classmethod
    def get_default_parallel_plan(cls, **kwargs):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(**kwargs)

    def get_weight_converter(self, parallel_context, model_path):
        return LLaDA2WeightConverter(self, parallel_context, model_path)

    @staticmethod
    def _pre_init_patches(config):
        from torchturbo.ops import ops
        from .modeling_llada2_moe import LLaDA2MoeRMSNorm

        def rms_norm_forward(self, hidden_states):
            return ops.rms_norm(
                hidden_states,
                self.weight,
                self.variance_epsilon,
            )

        LLaDA2MoeRMSNorm.forward = rms_norm_forward

    @staticmethod
    def _post_init_patches():
        pass

    def on_materialized(self) -> None:
        rotary_emb = self.model.rotary_emb
        device = get_device_runtime().device

        inv_freq, attention_scaling = rotary_emb.compute_default_rope_parameters(
            rotary_emb.config, device
        )
        rotary_emb.attention_scaling = attention_scaling
        rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)
        rotary_emb.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    def on_parallelized(self, parallel_context) -> None:
        sp_info = parallel_context["sp"]
        sp_group = sp_info["sp_group"]
        sp_size = sp_info["sp_size"]
        sp_rank = sp_info["sp_rank"]
        if sp_size <= 1 or sp_group is None:
            return
        num_attention_heads = self.config.num_attention_heads
        sp_repeat_q = max(sp_size // num_attention_heads, 1)
        self._patch_ulysses_attn_for_sp(sp_group, sp_size, sp_rank, sp_repeat_q)

    @classmethod
    def _patch_ulysses_attn_for_sp(cls, sp_group, sp_size, sp_rank, sp_repeat_q=1):
        import functools
        from .modeling_llada2_moe import (
            LLaDA2MoeAttention,
            apply_rotary_pos_emb,
            ALL_ATTENTION_FUNCTIONS,
        )
        from ..modules.attention.ulysses.ulysses import (
            gather_seq_scatter_heads,
            gather_heads_scatter_seq,
        )

        origin_attr = "_torchturbo_llada2_sp_origin_forward"
        if getattr(LLaDA2MoeAttention, origin_attr, None) is not None:
            return
        setattr(LLaDA2MoeAttention, origin_attr, LLaDA2MoeAttention.forward)

        @functools.wraps(LLaDA2MoeAttention.forward)
        def sp_forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            position_embeddings=None,
            **kwargs,
        ):
            if past_key_value is not None:
                raise NotImplementedError(
                    "LLaDA2 Ulysses SP attention does not support KV cache."
                )

            bsz, q_len, _ = hidden_states.size()

            qkv = self.query_key_value(hidden_states).view(
                bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim
            )
            query_states, key_states, value_states = qkv.split(
                [self.num_heads, self.num_key_value_heads, self.num_key_value_heads],
                dim=-2,
            )
            # [B, h, T_local, head_dim]
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            if self.config.use_qk_norm:
                query_states = self.query_layernorm(query_states)
                key_states = self.key_layernorm(key_states)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            repeat_q_factor = sp_size // self.num_heads
            repeat_kv_factor = sp_size // self.num_key_value_heads

            if repeat_q_factor > 1:
                query_states = query_states.repeat_interleave(repeat_q_factor, dim=1)
            if repeat_kv_factor > 1:
                key_states = key_states.repeat_interleave(repeat_kv_factor, dim=1)
                value_states = value_states.repeat_interleave(repeat_kv_factor, dim=1)

            query_states = gather_seq_scatter_heads(
                query_states, seq_dim=2, head_dim=1,
                sp_group=sp_group, sp_size=sp_size,
            )
            key_states = gather_seq_scatter_heads(
                key_states, seq_dim=2, head_dim=1,
                sp_group=sp_group, sp_size=sp_size,
            )
            value_states = gather_seq_scatter_heads(
                value_states, seq_dim=2, head_dim=1,
                sp_group=sp_group, sp_size=sp_size,
            )

            _q_loc = query_states.shape[1]
            _kv_loc = key_states.shape[1]
            _eff_gqa = max(_q_loc // _kv_loc, 1)
            _orig_gqa = self.num_key_value_groups
            self.num_key_value_groups = _eff_gqa

            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

            self.num_key_value_groups = _orig_gqa

            attn_output = gather_heads_scatter_seq(
                attn_output, head_dim=2, seq_dim=1,
                sp_group=sp_group, sp_size=sp_size,
            )

            if repeat_q_factor > 1:
                attn_output = attn_output[:, :, ::repeat_q_factor, :]

            attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
            attn_output = self.dense(attn_output)

            return attn_output, None, past_key_value

        LLaDA2MoeAttention.forward = sp_forward
