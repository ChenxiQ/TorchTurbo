"""Bailing V3 hybrid parameter semantics for TorchTurbo's distributed Muon."""

from __future__ import annotations

from torch import nn
from torch.distributed.tensor import Shard

from torchturbo.engine.optimizers import (
    AttentionPerHeadComputeView,
    ExcludeFromMuon,
    MLAQKClipSpec,
    MuonParameterOverride,
    MuonParameterSpec,
    MuonRecipe,
)

from .modeling_bailing_moe_v3 import (
    BailingMoeV3KimiDeltaAttention,
    BailingMoeV3MultiLatentAttention,
)


def get_muon_recipe(model: nn.Module) -> MuonRecipe:
    """Describe the Bailing V3 parameters that differ from the Muon defaults.

    MLA and KDA layers sit behind the same ``attention`` attribute and reuse
    projection names for differently shaped weights: MLA ``g_proj`` holds one
    scalar per head where KDA ``g_proj`` holds a full head_dim block. Entries
    therefore name exact layers instead of matching ``layers.*``. 
    """
    per_head = MuonParameterSpec(
        compute_placement=Shard(0),
        compute_view=AttentionPerHeadComputeView(
            int(model.config.num_attention_heads)
        ),
    )
    # Rows of a per-head projection form one contiguous block per head, and each
    # block is one Muon matrix. The excluded projections hold a single scalar per
    # head, so their rows carry no matrix for Newton-Schulz.
    query_projection = "q_proj" if model.config.q_lora_rank is None else "q_b_proj"
    specs_by_attention_type = {
        BailingMoeV3MultiLatentAttention: {
            f"{query_projection}.weight": per_head,
            "kv_b_proj.weight": per_head,
            "g_proj.weight": ExcludeFromMuon(),
        },
        BailingMoeV3KimiDeltaAttention: {
            "q_proj.weight": per_head,
            "k_proj.weight": per_head,
            "v_proj.weight": per_head,
            "f_proj.weight": per_head,
            "g_proj.weight": per_head,
            "b_proj.weight": ExcludeFromMuon(),
        },
    }
    parameter_overrides: dict[str, MuonParameterOverride] = {}
    qk_clip_specs: dict[str, MLAQKClipSpec] = {}
    for layer_index, layer in enumerate(model.get_decoder().layers):
        attention_fqn = f"model.layers.{layer_index}.attention"
        attention_class = type(layer.attention)
        for suffix, spec in specs_by_attention_type[attention_class].items():
            parameter_overrides[f"{attention_fqn}.{suffix}"] = spec
        # QK clipping applies to the MLA layers only: KDA linear attention
        # has no per-head QK logits to observe.
        if attention_class is BailingMoeV3MultiLatentAttention:
            qk_clip_specs[attention_fqn] = MLAQKClipSpec(
                query_projection=query_projection,
                key_value_projection="kv_b_proj",
                num_heads=int(model.config.num_attention_heads),
                qk_nope_head_dim=int(model.config.qk_nope_head_dim),
                qk_rope_head_dim=int(model.config.qk_rope_head_dim),
                v_head_dim=int(model.config.v_head_dim),
            )
    return MuonRecipe(
        parameter_overrides=parameter_overrides,
        qk_clip_specs=qk_clip_specs,
    )
