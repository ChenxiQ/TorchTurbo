import torch
from torchturbo.engine.metrics.flops import BaseModelFlops
from transformers import AutoModelForImageTextToText
from typing import Any
from typing_extensions import override

from torchturbo.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from torchturbo.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)
from torchturbo.runtime import get_device_runtime

from ...engine.config import ParameterGroup
from ..base import TurboVLMModel
from ..modules.attention.ulysses import SequenceParallelInputSlicingSpec
from ..registry import register_model
from ..weight_converter import BaseWeightConverter
from .inputs import prepare_packed_position_ids


def _interleave_qkv_conv_parameter(q, k, v, num_key_heads):
    """Match depthwise-conv channels to the post-all-to-all QKV layout."""
    trailing_shape = q.shape[1:]
    q = q.reshape(num_key_heads, -1, *trailing_shape)
    k = k.reshape(num_key_heads, -1, *trailing_shape)
    v = v.reshape(num_key_heads, -1, *trailing_shape)
    return torch.cat((q, k, v), dim=1).reshape(-1, *trailing_shape)


class Qwen3_5WeightConverter(BaseWeightConverter):
    def __init__(self, model, parallel_context, model_path):
        super().__init__(model, parallel_context, model_path)
        config = self.config
        if hasattr(config, "text_config"):
            self.config = config.text_config
            self.model_type = self.config.model_type


@register_model(
    "qwen3_5",
    config_cls=Qwen3_5Config,
    auto_class=AutoModelForImageTextToText,
)
class TurboQwen3_5ForConditionalGeneration(
    TurboVLMModel, Qwen3_5ForConditionalGeneration
):

    config_class = Qwen3_5Config
    base_model_prefix = "model"
    multi_modal_projector_module_names = ("model.visual.merger",)
    language_model_module_names = ("model.language_model",)

    @property
    def visual(self):
        return self.model.visual

    def get_parameter_groups(self):
        groups = super().get_parameter_groups()
        groups[ParameterGroup.VISION_TOWER] = (
            (name, param)
            for name, param in self.model.visual.named_parameters(prefix="model.visual")
            if name != "model.visual.merger"
            and not name.startswith("model.visual.merger.")
        )
        return groups

    def get_norm(self):
        return self.model.language_model.norm

    def get_flops_class(self) -> type[BaseModelFlops]:
        """Qwen3.5 dense flops recipe (hetero attention + gated-MLP FFN + vision)."""
        from .mfu import Qwen3_5DenseFlops

        return Qwen3_5DenseFlops

    def __init__(self, config: Qwen3_5Config) -> None:
        self._pre_init_patches()
        super().__init__(config)
        self._post_init_patches()

    @classmethod
    def get_default_parallel_plan(cls, **kwargs):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan()

    @staticmethod
    def _pre_init_patches():
        # No MoE token-dispatcher patching for dense (no experts).
        pass

    def _post_init_patches(self):
        pass

    def get_sequence_parallel_input_slicing_spec(self) -> SequenceParallelInputSlicingSpec:
        from .modeling_qwen3_5 import Qwen3_5TextModel

        return SequenceParallelInputSlicingSpec(
            self.model.language_model,
            Qwen3_5TextModel.forward,
        )

    def on_prepare_inputs(
        self,
        inputs: dict[str, Any],
        *,
        source_inputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build per-sample mRoPE from the packed row's media grids."""
        inputs = super().on_prepare_inputs(inputs, source_inputs=source_inputs)
        return prepare_packed_position_ids(self.model, inputs)

    def on_parallelized(self, parallel_context):
        sp_info = parallel_context["sp"]
        sp_group = sp_info["sp_group"]
        sp_size = sp_info["sp_size"]
        sp_rank = sp_info["sp_rank"]
        if sp_size > 1:
            from torchturbo.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5GatedDeltaNet,
                Qwen3_5MultiTokenPredictor,
            )

            # Vision keeps flash_attention_2 on its own subconfig and stays
            # on the native flash path; only text attention switches to the
            # registered SP interface.
            self.register_sp_attention_interface(sp_info)

            device_runtime = get_device_runtime()
            a2a_stream = device_runtime.new_stream()
            for module in self.modules():
                if isinstance(module, Qwen3_5GatedDeltaNet):
                    module._sp_a2a_stream = a2a_stream

            self.patch_gated_delta_net_for_sp(
                Qwen3_5GatedDeltaNet, sp_group, sp_size, sp_rank
            )
            if self.mtp is not None:

                self.patch_mtp_for_sp(
                    Qwen3_5MultiTokenPredictor, sp_group, sp_size, sp_rank
                )

    # ------------------------------------------------------------------
    # MTP: SP input slicing
    # ------------------------------------------------------------------

    @classmethod
    def patch_mtp_for_sp(cls, mtp_cls, sp_group, sp_size, sp_rank):
        """Keep MTP next-token inputs correct under sequence parallelism.

        The VLM path provides global IDs, while the text path provides SP-local
        IDs. The text path gathers local IDs before MTP builds next-token
        embeddings so SP-boundary tokens use the next rank's first ID.
        """
        from ..modules.attention.ulysses import slice_input_tensor

        original_forward = mtp_cls.forward

        def _mtp_sp_forward(self_mtp, input_ids, hidden_states, embed_tokens, **kwargs):
            position_embeddings = kwargs.pop("position_embeddings", None)
            attention_mask = kwargs.pop("attention_mask", None)
            position_ids = kwargs.pop("position_ids", None)
            pad_token_id = int(kwargs.pop("pad_token_id", 0))
            # The backbone is already SP-local. Reconstruct only IDs so MTP
            # rolling sees the next token across an SP boundary.
            local_seq_len = hidden_states.size(1)
            if input_ids.size(1) == local_seq_len:
                input_ids_list = [torch.empty_like(input_ids) for _ in range(sp_size)]
                torch.distributed.all_gather(input_ids_list, input_ids, group=sp_group)
                input_ids = torch.cat(input_ids_list, dim=1)

            def localize(tensor, dim):
                if tensor is None or tensor.size(dim) == local_seq_len:
                    return tensor
                return slice_input_tensor(
                    tensor, dim=dim, group=sp_group, rank=sp_rank
                )

            position_ids = localize(position_ids, -1)
            if position_embeddings is not None:
                cos, sin = position_embeddings
                position_embeddings = (localize(cos, 1), localize(sin, 1))

            def sp_embed_tokens(ids):
                return localize(embed_tokens(ids), 1)

            return original_forward(
                self_mtp,
                input_ids,
                hidden_states,
                sp_embed_tokens,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pad_token_id=pad_token_id,
                **kwargs,
            )

        mtp_cls.forward = _mtp_sp_forward

    @classmethod
    def patch_gated_delta_net_for_sp(cls, attn_cls, sp_group, sp_size, sp_rank):
        from ..modules.attention.ulysses.ulysses import (
            gather_seq_scatter_heads,
            gather_heads_scatter_seq,
        )
        from torchturbo.ops import ops
        import torch.nn.functional as F

        def _slice_param(weight, dim):
            slices = [slice(None)] * weight.dim()
            d = weight.size(dim)
            slices[dim] = slice(sp_rank * d // sp_size, (sp_rank + 1) * d // sp_size)
            return weight[tuple(slices)]

        def sp_forward(self_attn, hidden_states, **kwargs):
            # Global packed boundaries ride with the forward kwargs (passed by
            # the decoder layer as cu_seq_lens); the a2a gathers below rebuild
            # the full sequence on each rank.
            cu_seq_lens = kwargs.get("cu_seq_lens")
            B, local_T, _ = hidden_states.shape

            nk_full, nv_full = self_attn.num_k_heads, self_attn.num_v_heads
            dk, dv = self_attn.head_k_dim, self_attn.head_v_dim

            mixed_qkv = self_attn.in_proj_qkv(hidden_states)

            q_raw, k_raw, v_raw = torch.split(
                mixed_qkv, [self_attn.key_dim, self_attn.key_dim, self_attn.value_dim], dim=-1
            )

            sp_repeat = max(1, sp_size // nk_full)
            sh = nk_full * sp_repeat
            G_per_sh = nv_full // sh

            q_shaped = q_raw.reshape(B, local_T, nk_full, dk)
            k_shaped = k_raw.reshape(B, local_T, nk_full, dk)
            if sp_repeat > 1:
                q_shaped = q_shaped.repeat_interleave(sp_repeat, dim=2)
                k_shaped = k_shaped.repeat_interleave(sp_repeat, dim=2)

            v_shaped = v_raw.reshape(B, local_T, sh, G_per_sh * dv)
            interleaved_qkv = torch.cat([q_shaped, k_shaped, v_shaped], dim=-1)

            a2a_stream = self_attn._sp_a2a_stream
            interleaved_qkv = gather_seq_scatter_heads(
                interleaved_qkv,
                seq_dim=1,
                head_dim=2,
                sp_group=sp_group,
                sp_size=sp_size,
                a2a_stream=a2a_stream,
            )
            z = self_attn.in_proj_z(hidden_states)
            b = self_attn.in_proj_b(hidden_states)
            a = self_attn.in_proj_a(hidden_states)
            get_device_runtime().current_stream().wait_stream(a2a_stream)
            T = interleaved_qkv.shape[1]
            nk = nk_full * sp_repeat // sp_size
            nv = nv_full // sp_size

            # The all-to-all result is already contiguous in per-key-head
            # [Q, K, grouped-V] order. Causal conv is depthwise, so keep this
            # layout and reorder only its tiny width-4 parameter instead of
            # copying the full 64K activation into grouped Q/K/V blocks.
            local_qkv = interleaved_qkv.flatten(2)

            w = self_attn.conv1d.weight
            wq, wk, wv = torch.split(
                w, [self_attn.key_dim, self_attn.key_dim, self_attn.value_dim], dim=0
            )

            K = self_attn.conv_kernel_size
            if sp_repeat > 1:
                wq = wq.reshape(nk_full, dk, 1, K).repeat_interleave(sp_repeat, dim=0).reshape(-1, 1, K)
                wk = wk.reshape(nk_full, dk, 1, K).repeat_interleave(sp_repeat, dim=0).reshape(-1, 1, K)

            local_conv_weight = _interleave_qkv_conv_parameter(
                _slice_param(wq, 0),
                _slice_param(wk, 0),
                _slice_param(wv, 0),
                nk,
            )

            local_conv_bias = None
            if self_attn.conv1d.bias is not None:
                bq, bk, bv = torch.split(
                    self_attn.conv1d.bias, [self_attn.key_dim, self_attn.key_dim, self_attn.value_dim], dim=0
                )
                if sp_repeat > 1:
                    bq = bq.reshape(nk_full, dk).repeat_interleave(sp_repeat, dim=0).reshape(-1)
                    bk = bk.reshape(nk_full, dk).repeat_interleave(sp_repeat, dim=0).reshape(-1)
                local_conv_bias = _interleave_qkv_conv_parameter(
                    _slice_param(bq, 0),
                    _slice_param(bk, 0),
                    _slice_param(bv, 0),
                    nk,
                )

            local_qkv, _ = ops.causal_conv1d(
                x=local_qkv,
                weight=local_conv_weight.squeeze(1),
                bias=local_conv_bias,
                activation=self_attn.activation,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=cu_seq_lens,
            )

            local_qkv = local_qkv.reshape(B, T, nk, -1)
            q = local_qkv[..., :dk]
            k = local_qkv[..., dk : 2 * dk]
            v = local_qkv[..., 2 * dk :].reshape(B, T, nv, dv)

            if nk < nv:
                r = nv // nk
                q = q.repeat_interleave(r, dim=2)
                k = k.repeat_interleave(r, dim=2)

            # Z is consumed only after ``out`` returns to local-sequence/full-head
            # layout, which is already Z's projection layout. Keep it local and
            # avoid a redundant round trip through sequence parallelism.
            local_z = z.reshape(B, local_T, nv_full, dv)
            packed_ba = gather_seq_scatter_heads(
                torch.stack((b, a), dim=-1),
                seq_dim=1,
                head_dim=2,
                sp_group=sp_group,
                sp_size=sp_size,
            )
            b, a = packed_ba.unbind(dim=-1)

            beta = b.sigmoid()
            A_log_local = _slice_param(self_attn.A_log, 0)
            dt_bias_local = _slice_param(self_attn.dt_bias, 0)
            g = -A_log_local.float().exp() * F.softplus(a.float() + dt_bias_local)

            out, _ = ops.gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seq_lens,
            )

            out = gather_heads_scatter_seq(
                out, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
            )

            out = self_attn.norm(out.reshape(-1, dv), local_z.reshape(-1, dv))
            return self_attn.out_proj(out.reshape(B, local_T, -1))

        attn_cls.forward = sp_forward

    def get_weight_converter(self, parallel_context, model_path):
        return Qwen3_5WeightConverter(self, parallel_context, model_path)

    def on_materialized(self):
        if getattr(self.config, "tie_word_embeddings", False):
            self.tie_weights()

        device = get_device_runtime().device
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        lm = self.model.language_model
        rotary_emb = lm.rotary_emb
        rope_init_fn = rotary_emb.compute_default_rope_parameters
        if rotary_emb.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[rotary_emb.rope_type]
        inv_freq, rotary_emb.attention_scaling = rope_init_fn(rotary_emb.config, device)
        rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)
        rotary_emb.register_buffer(
            "original_inv_freq", inv_freq.clone(), persistent=False
        )

        vis = self.model.visual
        if hasattr(vis, "rotary_pos_emb"):
            vis_rope = vis.rotary_pos_emb
            vis_inv_freq = 1.0 / (
                vis_rope.theta
                ** (torch.arange(0, vis_rope.dim, 2, dtype=torch.float) / vis_rope.dim)
            ).to(device)
            vis_rope.register_buffer("inv_freq", vis_inv_freq, persistent=False)
