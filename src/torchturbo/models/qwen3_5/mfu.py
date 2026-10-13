"""Qwen3.5 forward-FLOP recipes (dense + MoE share this base).

Both use the same hetero attention (``layer_types``: gated GQA ``full_attention``
+ GatedDeltaNet ``linear_attention``) and MTP predictor. The recipe returns the
full-input upper bound (attn/FFN/MTP/logits + vision from the real per-batch
``grid_thw``); the estimator divides the total by ``sp_size``.
"""

from __future__ import annotations

from ...engine.metrics import op_flops as flops
from ...engine.metrics.flops import Component, VLMFlops

__all__ = ["Qwen3_5BaseFlops", "Qwen3_5DenseFlops"]


class Qwen3_5BaseFlops(VLMFlops):
    """Qwen3.5 shared forward FLOPs: hetero attention + MTP + vision.

    Subclasses implement :meth:`ffn_flops` (FFN cost per layer) and
    :meth:`ffn_component` (the breakdown line item) — dense gated-MLP vs MoE.
    """

    def _parse_config(self) -> None:
        super()._parse_config()  # LLMFlops reads hidden/heads/vocab/...
        g = self._cfg

        layer_types: list[str] = list(g("layer_types", default=[]) or [])
        if layer_types:
            self.layer_types = layer_types
            self.num_layers = len(layer_types)
        else:
            self.layer_types = ["full_attention"] * self.num_layers

        # Linear-attention dims (GatedDeltaNet).
        self.linear_num_k_heads: int = int(g("linear_num_key_heads", default=0))
        self.linear_num_v_heads: int = int(g("linear_num_value_heads", default=0))
        self.linear_key_head_dim: int = int(g("linear_key_head_dim", default=0))
        self.linear_value_head_dim: int = int(g("linear_value_head_dim", default=0))
        self.linear_conv_kernel: int = int(g("linear_conv_kernel_dim", default=1))

        # MTP.
        self.mtp_layers: int = int(g("mtp_num_hidden_layers", default=0))

        self.n_full: int = sum(1 for t in self.layer_types if t == "full_attention")
        self.n_linear: int = sum(1 for t in self.layer_types if t == "linear_attention")

    # -- attention: heterogeneous, segment-aware -------------------------- #
    def _full_attention_flops_for_seg(self, seg: int) -> int:
        # Qwen3.5 gated GQA: q_proj emits q + gate in one matmul (out = nh*hd*2),
        # and the gate is applied as attn_output * sigmoid(gate) after attention —
        # no extra matmul, qk/pv/o_proj match standard GQA. Count standard GQA and
        # add the extra gate channel of q_proj.
        base = flops.mha_attention_flops(
            self.B, seg, self.hidden_size,
            self.num_heads, self.head_dim, self.num_kv_heads,
        )
        gate_proj = flops.linear_flops(self.B * seg, self.hidden_size, self.num_heads * self.head_dim)
        return base + gate_proj

    def _linear_attention_per_layer(self) -> int:
        # Linear attention is O(S) per token — independent of segmentation.
        return flops.linear_attention_flops(
            self.B, self.S_eff, self.hidden_size,
            self.linear_num_k_heads, self.linear_key_head_dim,
            self.linear_num_v_heads, self.linear_value_head_dim,
            conv_kernel_size=self.linear_conv_kernel,
        )

    # -- FFN: subclass hook ----------------------------------------------- #
    def ffn_component(self) -> Component:
        """Per-layer FFN line item (gated-MLP for dense, MoE for the MoE model).

        Uses :meth:`ffn_flops` for the cost and multiplies by ``num_layers``.
        """
        raise NotImplementedError

    # -- extra: MTP; vision is provided via vision_components() ------------ #
    def extra_components(self) -> list[Component]:
        return self.mtp_components()

    def mtp_components(self) -> list[Component]:
        if self.mtp_layers <= 0:
            return []
        # Each MTP layer ≈ one full-attn layer + one FFN.
        mtp_attn_per = (
            sum(self._full_attention_flops_for_seg(s) for s in self.seq_segments)
            if self.n_full else 0
        )
        mtp_attn = mtp_attn_per * self.mtp_layers
        mtp_ffn = self.ffn_flops() * self.mtp_layers
        # Predictor fusion fc: Linear(2*hidden -> hidden); one per layer.
        mtp_fc = flops.linear_flops(self.B * self.S_eff, 2 * self.hidden_size, self.hidden_size) * self.mtp_layers
        # Each MTP stream passes through the shared lm_head; one hidden->vocab per layer.
        mtp_logits = flops.logits_flops(self.B, self.S_eff, self.hidden_size, self.vocab_size) * self.mtp_layers
        return [Component(
            f"mtp ({self.mtp_layers} layers, attn+ffn+fc+lm_head)",
            mtp_attn + mtp_ffn + mtp_fc + mtp_logits,
            note="approx: full-attention + FFN + fusion fc + per-stream lm_head per layer",
        )]

    def vision_components(self) -> list[Component]:
        """Vision-tower forward from the real per-batch ``grid_thw`` (full input)."""
        prof = self.profile
        image_grids = list(prof.image_grid_thw or [])
        video_grids = list(prof.video_grid_thw or [])
        if not image_grids and not video_grids:
            return []
        vc = self._cfg("vision_config", default=None)
        if not vc:
            return []
        try:
            img, img_merged = flops.vision_encoder_flops(image_grids, vc) if image_grids else (0, 0)
            vid, _ = flops.vision_encoder_flops(video_grids, vc) if video_grids else (0, 0)
        except Exception:
            return []
        total = int(img + vid)
        if total <= 0:
            return []
        n_imgs = len(image_grids) + len(video_grids)
        note = f"real grid_thw, imgs+videos={n_imgs}; merged_tokens={img_merged}"
        return [Component(
            f"vision encoder ({int(getattr(vc, 'depth', 0))} blocks, {n_imgs} items)",
            total,
            note=note,
        )]

    # -- breakdown: attn + FFN + MTP + vision + logits --------------------- #
    def components(self) -> list[Component]:
        # Full-input upper bound; the estimator divides the sum by ``sp_size``.
        full_per_seg = (
            [self._full_attention_flops_for_seg(s) for s in self.seq_segments]
            if self.n_full else []
        )
        full_per = sum(full_per_seg) if self.n_full else 0
        linear_per = self._linear_attention_per_layer() if self.n_linear else 0

        comps: list[Component] = []
        if self.n_full:
            comps.append(Component(
                f"attention/full ({self.n_full} layers)",
                full_per * self.n_full,
                note=f"gated GQA, q_proj emits q+gate; per-layer fwd={full_per:,}",
            ))
        if self.n_linear:
            comps.append(Component(
                f"attention/linear ({self.n_linear} layers)",
                linear_per * self.n_linear,
                note=f"GatedDeltaNet O(S); per-layer fwd={linear_per:,}",
            ))

        comps.append(self.ffn_component())

        comps.extend(self.mtp_components())

        comps.append(self.logits_component())

        comps.extend(self.vision_components())
        return comps

class Qwen3_5DenseFlops(Qwen3_5BaseFlops):
    """Qwen3.5 dense theoretical forward FLOPs (hetero attention + gated-MLP FFN)."""

    def _parse_config(self) -> None:
        super()._parse_config()
        self.intermediate_size = int(self._cfg("intermediate_size", default=0))

    def ffn_flops(self) -> int:
        # Dense gated-MLP (gate/up/down), one expert.
        if self.intermediate_size <= 0:
            return 0
        return flops.gated_mlp_flops(self.B, self.S_eff, self.hidden_size, self.intermediate_size)

    def ffn_component(self) -> Component:
        ffn_per = self.ffn_flops()
        return Component(
            f"ffn ({self.num_layers} layers)",
            ffn_per * self.num_layers,
            note=f"gated MLP inter={self.intermediate_size}; per-layer fwd={ffn_per:,}",
        )