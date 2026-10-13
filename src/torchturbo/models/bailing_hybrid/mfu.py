"""Bailing-MoE-V3 forward-FLOP recipe (hybrid MLA + KDA attention, dense + MoE FFN).

Bailing v3 interleaves two attention families per decoder layer and two FFN
families independently of the attention choice:

* attention type (per-layer, mirroring ``BailingMoeV3DecoderLayer``):
    - ``BailingMoeV3MultiLatentAttention`` (MLA, O(S²), segment-aware) when
      ``(layer_idx + 1) % layer_group_size == 0`` or the layer falls in the
      trailing partial group; everything else is
    - ``BailingMoeV3KimiDeltaAttention`` (KDA / chunked gated-delta linear
      attention, O(S), segment-independent).
* FFN type: a gated dense MLP for the first ``first_k_dense_replace`` layers and
  a routed MoE block (SwiGLU experts + 1 shared expert + router) thereafter.

There is no vision tower (pure causal LM). ``num_nextn_predict_layers`` MTP
streams add ``eh_proj`` (2H→H) + one MLA layer + one MoE FFN + a per-stream
``lm_head`` each. The recipe returns the full-input upper bound; the estimator
divides the sum by ``sp_size``.

The MLA/KDA FLOPs reuse the shared primitives in
:mod:`torchturbo.engine.metrics.op_flops`. KDA's ``chunk_kda`` recurrence is
approximated by the gated-delta-rule recurrence (same state-update count) — it
adds gate/``dt_bias``/``safe_gate`` modulation that does not change the dominant
outer-product cost.
"""

from __future__ import annotations

from ...engine.metrics import op_flops as flops
from ...engine.metrics.flops import Component, LLMFlops

__all__ = ["BailingMoeV3Flops"]


class BailingMoeV3Flops(LLMFlops):
    """Bailing v3 theoretical forward FLOPs (hybrid MLA+KDA attention, MoE FFN)."""

    # -- config parsing ---------------------------------------------------- #
    def _parse_config(self) -> None:
        super()._parse_config()  # LLMFlops reads hidden/heads/vocab/...

        # Hybrid attention layout (must mirror BailingMoeV3DecoderLayer).
        self.layer_group_size: int = int(self._cfg("layer_group_size", default=1) or 1)
        self.first_k_dense_replace: int = int(self._cfg("first_k_dense_replace", default=0))

        # MLA dims.
        self.q_lora_rank = self._cfg("q_lora_rank", default=None)
        self.kv_lora_rank: int = int(self._cfg("kv_lora_rank", default=0))
        self.qk_nope_head_dim: int = int(self._cfg("qk_nope_head_dim", default=0))
        self.qk_rope_head_dim: int = int(self._cfg("qk_rope_head_dim", default=0))
        self.v_head_dim: int = int(self._cfg("v_head_dim", default=self.qk_nope_head_dim))
        self.gated_attention_proj_granularity_type = self._cfg(
            "gated_attention_proj_granularity_type", default=None
        )

        # KDA (linear attention) dims.
        self.kda_head_dim: int = int(self._cfg("head_dim", default=self.head_dim))
        self.kda_num_heads: int = int(
            self._cfg("num_attention_heads", "num_heads", default=self.num_heads)
        )
        # head_k_dim equals head_dim and num_k_heads equals num_heads (see modeling).
        self.kda_head_k_dim: int = self.kda_head_dim
        self.kda_num_k_heads: int = self.kda_num_heads
        self.no_kda_lora: bool = bool(self._cfg("no_kda_lora", default=False))
        self.short_conv_kernel_size: int = int(self._cfg("short_conv_kernel_size", default=1))

        # MoE FFN.
        self.num_experts: int = int(self._cfg("num_experts", "n_routed_experts", default=0))
        self.top_k: int = int(self._cfg("num_experts_per_tok", default=0))
        self.moe_intermediate_size: int = int(self._cfg("moe_intermediate_size", default=0))
        self.num_shared_experts: int = int(self._cfg("num_shared_experts", "n_shared_experts", default=0))
        self.moe_shared_expert_intermediate_size: int = int(
            self._cfg("moe_shared_expert_intermediate_size", default=0)
        )
        if self.num_shared_experts > 0 and self.moe_shared_expert_intermediate_size > 0:
            self.moe_shared_expert_intermediate_size *= self.num_shared_experts

        # MTP (NextN prediction).
        self.mtp_layers: int = int(self._cfg("num_nextn_predict_layers", default=0))

        # Layer counts by attention and FFN family (decoder layer index space).
        self.n_mla: int = sum(
            1 for i in range(self.num_layers) if self._layer_is_mla(i)
        )
        self.n_kda: int = self.num_layers - self.n_mla
        self.n_dense_ffn: int = min(self.first_k_dense_replace, self.num_layers)
        self.n_moe_ffn: int = self.num_layers - self.n_dense_ffn

    # -- helpers ----------------------------------------------------------- #
    def _layer_is_mla(self, layer_idx: int) -> bool:
        """Mirror ``BailingMoeV3DecoderLayer.attention_layer_type`` selection."""
        return (
            (layer_idx + 1) % self.layer_group_size == 0
            or layer_idx >= self.num_layers // self.layer_group_size * self.layer_group_size
        )

    def _mla_flops_for_seg(self, seg: int) -> int:
        """Per MLA layer forward FLOPs for one packed segment (O(seg²) qk/pv)."""
        g_proj = 0
        if self.gated_attention_proj_granularity_type == "head_wise":
            g_proj = flops.linear_flops(self.B * seg, self.hidden_size, self.num_heads)
        elif self.gated_attention_proj_granularity_type == "element_wise":
            g_proj = flops.linear_flops(
                self.B * seg, self.hidden_size, self.num_heads * self.v_head_dim
            )
        attn = flops.mla_attention_flops(
            self.B, seg, self.hidden_size, self.num_heads,
            self.q_lora_rank, self.kv_lora_rank,
            self.qk_nope_head_dim, self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            has_q_down_proj=self.q_lora_rank is not None,
            has_kv_down_proj=True,
        )
        return attn + g_proj

    def _mla_flops(self) -> int:
        return sum(self._mla_flops_for_seg(s) for s in self.seq_segments)

    def _kda_per_layer(self) -> int:
        """Per KDA layer forward FLOPs (O(S), segment-independent; uses S_eff)."""
        nh = self.kda_num_heads
        nk = self.kda_num_k_heads
        head_dim = self.kda_head_dim
        head_k_dim = self.kda_head_k_dim
        key_dim = head_k_dim * nk          # q/k projection width
        value_dim = head_dim * nh          # v/g/o projection width
        bs = self.B * self.S_eff
        conv_k = self.short_conv_kernel_size

        f = 0
        # q/k/v projections (unfused).
        f += flops.linear_flops(bs, self.hidden_size, key_dim)   # q_proj
        f += flops.linear_flops(bs, self.hidden_size, key_dim)   # k_proj
        f += flops.linear_flops(bs, self.hidden_size, value_dim) # v_proj
        # Depthwise short convolutions (silu) on q/k/v.
        f += flops.depthwise_conv1d_flops(self.B, self.S_eff, key_dim, conv_k)
        f += flops.depthwise_conv1d_flops(self.B, self.S_eff, key_dim, conv_k)
        f += flops.depthwise_conv1d_flops(self.B, self.S_eff, value_dim, conv_k)
        # f-gate: the ``g`` attenuation gate fed to chunk_kda.
        f += self._kda_gate_flops(value_dim, head_dim)
        # b_proj: the per-head beta (delta-rule) gate.
        f += flops.linear_flops(bs, self.hidden_size, nh)
        # chunk_kda recurrence (~ gated-delta-rule state updates over v-heads).
        f += flops.recurrence_flops(self.B, self.S_eff, nh, head_k_dim, head_dim)
        # g-gate: the post-recurrence o_norm gate.
        f += self._kda_gate_flops(value_dim, head_dim)
        # output projection.
        f += flops.linear_flops(bs, value_dim, self.hidden_size)
        return f

    def _kda_gate_flops(self, value_dim: int, head_dim: int) -> int:
        """Cost of one KDA gate projection (full or low-rank LoRA variant)."""
        bs = self.B * self.S_eff
        if self.no_kda_lora:
            return flops.linear_flops(bs, self.hidden_size, value_dim)
        return (
            flops.linear_flops(bs, self.hidden_size, head_dim)
            + flops.linear_flops(bs, head_dim, value_dim)
        )

    def _dense_ffn_per_layer(self) -> int:
        if self.intermediate_size <= 0:
            return 0
        return flops.gated_mlp_flops(self.B, self.S_eff, self.hidden_size, self.intermediate_size)

    def _moe_ffn_per_layer(self) -> int:
        if self.num_experts <= 0 or self.top_k <= 0:
            return self._dense_ffn_per_layer()
        return flops.moe_flops(
            self.B, self.S_eff, self.hidden_size,
            self.moe_intermediate_size,
            self.num_experts, self.top_k,
            shared_expert_intermediate_size=self.moe_shared_expert_intermediate_size,
            use_gate=True,
        )

    # -- breakdown --------------------------------------------------------- #
    def mtp_components(self) -> list[Component]:
        if self.mtp_layers <= 0:
            return []
        mtp_attn_per = self._mla_flops()  # each MTP layer uses one MLA attention
        mtp_ffn_per = self._moe_ffn_per_layer()  # each MTP layer uses one MoE FFN
        # eh_proj: fuse [embeds || hidden] -> hidden; one per layer.
        mtp_eh = flops.linear_flops(self.B * self.S_eff, 2 * self.hidden_size, self.hidden_size)
        # Per-stream lm_head (shared head, one hidden->vocab per layer).
        mtp_logits = flops.logits_flops(self.B, self.S_eff, self.hidden_size, self.vocab_size)
        per = mtp_attn_per + mtp_ffn_per + mtp_eh + mtp_logits
        total = per * self.mtp_layers
        return [Component(
            f"mtp ({self.mtp_layers} layers, mla+moe+eh_proj+lm_head)",
            total,
            note="approx: per layer = 1 MLA attn + 1 MoE FFN + fusion fc + per-stream lm_head",
        )]

    def components(self) -> list[Component]:
        # Full-input upper bound; the estimator divides the sum by ``sp_size``.
        comps: list[Component] = []

        if self.n_mla:
            mla_per = self._mla_flops()
            comps.append(Component(
                f"attention/mla ({self.n_mla} layers)",
                mla_per * self.n_mla,
                note=f"Multi-Latent Attention O(S^2), kv_lora={self.kv_lora_rank}; "
                     f"per-layer fwd={mla_per:,}",
            ))
        if self.n_kda:
            kda_per = self._kda_per_layer()
            comps.append(Component(
                f"attention/kda ({self.n_kda} layers)",
                kda_per * self.n_kda,
                note=f"Kimi Delta linear attention O(S), head_dim={self.kda_head_dim}; "
                     f"per-layer fwd={kda_per:,}",
            ))

        if self.n_dense_ffn:
            dense_per = self._dense_ffn_per_layer()
            comps.append(Component(
                f"ffn/dense ({self.n_dense_ffn} layers)",
                dense_per * self.n_dense_ffn,
                note=f"gated MLP inter={self.intermediate_size}; per-layer fwd={dense_per:,}",
            ))
        if self.n_moe_ffn:
            moe_per = self._moe_ffn_per_layer()
            comps.append(Component(
                f"ffn/moe ({self.n_moe_ffn} layers)",
                moe_per * self.n_moe_ffn,
                note=f"top_k={self.top_k} experts={self.num_experts} "
                     f"shared_inter={self.moe_shared_expert_intermediate_size}; "
                     f"per-layer fwd={moe_per:,}",
            ))

        comps.extend(self.mtp_components())

        lg = self.logits_component()
        comps.append(Component(lg.name, lg.fwd_flops, lg.note))
        return comps