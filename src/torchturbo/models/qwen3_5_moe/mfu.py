"""Qwen3.5-MoE forward-FLOP recipe — MoE FFN on top of the shared Qwen3.5 base.

Subclasses :class:`~torchturbo.models.qwen3_5.mfu.Qwen3_5BaseFlops`, only
swapping the FFN for routed MoE experts (plus an optional shared expert).
"""

from __future__ import annotations

from ...engine.metrics import op_flops as flops
from ...engine.metrics.flops import Component
from ..qwen3_5.mfu import Qwen3_5BaseFlops

__all__ = ["Qwen3_5MoeFlops"]


class Qwen3_5MoeFlops(Qwen3_5BaseFlops):
    """Qwen3.5-MoE theoretical forward FLOPs (gated GQA + GatedDeltaNet + MoE FFN)."""

    def _parse_config(self) -> None:
        super()._parse_config()  # Qwen3_5BaseFlops reads hetero attention / MTP
        g = self._cfg
        self.num_experts: int = int(g("num_experts", "n_routed_experts", default=0))
        self.top_k: int = int(g("num_experts_per_tok", default=0))
        self.expert_intermediate_size: int = int(
            g("moe_intermediate_size", "intermediate_size", default=self.intermediate_size)
        )
        self.shared_expert_intermediate_size: int = int(g("shared_expert_intermediate_size", default=0))
        self.num_shared_experts: int = int(g("num_shared_experts", "n_shared_experts", default=0))
        if self.num_shared_experts > 0 and self.shared_expert_intermediate_size > 0:
            self.shared_expert_intermediate_size *= self.num_shared_experts

    def ffn_flops(self) -> int:
        if self.num_experts <= 0 or self.top_k <= 0:
            return super().ffn_flops()
        return flops.moe_flops(
            self.B, self.S_eff, self.hidden_size,
            self.expert_intermediate_size,
            self.num_experts, self.top_k,
            shared_expert_intermediate_size=self.shared_expert_intermediate_size,
            use_gate=True,
        )

    def ffn_component(self) -> Component:
        ffn_per = self.ffn_flops()
        return Component(
            f"ffn/moe ({self.num_layers} layers)",
            ffn_per * self.num_layers,
            note=f"top_k={self.top_k} experts={self.num_experts} "
                 f"shared_inter={self.shared_expert_intermediate_size}; per-layer fwd={ffn_per:,}",
        )