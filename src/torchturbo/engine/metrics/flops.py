"""MFU recipe data types + reusable forward-FLOP recipe bases.

Recipes return the theoretical upper bound for a full input (no parallelism
split); :class:`~torchturbo.engine.metrics.mfu_estimator.MFUEstimator` divides
the sum by ``sp_size`` since the SP group cooperates on one input.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from transformers.configuration_utils import PretrainedConfig

from . import op_flops as flops

@dataclass
class ComputeProfile:
    """Runtime conditions under which MFU/HFU are reported."""

    batch_size: int
    seq_length: int
    num_devices: int = 1
    step_time: float | None = None           # seconds per optimizer step
    gradient_checkpointing: bool = False
    vision_images: int = 0                    # VLM: images per device step
    vision_frames: int = 1
    image_height: int = 0
    image_width: int = 0
    # Per-block attention lengths (single batch row); attention uses B*sum(S_i^2), FFN/logits use B*sum(S_i).
    seq_segments: list[int] | None = None
    # Real per-batch vision grids (list[(t,h,w)]) for VLM micros.
    image_grid_thw: list | None = None
    video_grid_thw: list | None = None
    peak_tflops: float = 0.0
    hardware_label: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

@dataclass
class Component:
    """One architectural line item with its forward FLOPs."""

    name: str
    fwd_flops: int
    note: str = ""

    def __post_init__(self) -> None:
        self.fwd_flops = int(self.fwd_flops)


class BaseModelFlops(ABC):
    """Abstract recipe base. Subclasses build forward-FLOP ``Component``s from config.

    ``MFUEstimator`` instantiates a recipe as ``MyFlops(config, profile)`` per
    micro-batch and sums ``components()`` (the full-input upper bound). ``config``
    is the model's ``PretrainedConfig`` (the VLM wrapper, e.g. ``Qwen3_5Config``);
    ``_cfg`` falls back to its nested ``text_config`` for decoder dims.
    """

    def __init__(self, config: PretrainedConfig, profile: ComputeProfile) -> None:
        assert isinstance(config, PretrainedConfig), (
            f"recipe config must be a PretrainedConfig, got {type(config).__name__}"
        )
        self.config: PretrainedConfig = config
        self.profile: ComputeProfile = profile
        self._parse_config()

    def _cfg(self, *keys: str, default: Any = None) -> Any:
        """First present attribute on config (or nested text_config)."""
        cfg = self.config
        text_cfg = getattr(cfg, "text_config", None)
        for k in keys:
            if hasattr(cfg, k):
                return getattr(cfg, k)
            if text_cfg is not None and hasattr(text_cfg, k):
                return getattr(text_cfg, k)
        return default

    def _parse_config(self) -> None:
        """Read dims from config into ``self.`` attributes. Override per model."""

    @abstractmethod
    def components(self) -> list[Component]:
        """Forward-FLOP components for this model."""

    def extra_components(self) -> list[Component]:
        """Hook for add-ons (MTP, vision, resampler). Empty by default."""
        return []


# LLM bases

class LLMFlops(BaseModelFlops):
    """Standard causal-LM decoder: per-layer attention + FFN + logits."""

    def _parse_config(self) -> None:
        self.num_layers: int = int(self._cfg("num_hidden_layers", "num_layers", default=0))
        self.hidden_size: int = int(self._cfg("hidden_size", default=0))
        self.num_heads: int = int(self._cfg("num_attention_heads", "num_heads", default=0))
        self.num_kv_heads: int = int(self._cfg("num_key_value_heads", default=self.num_heads))
        self.head_dim: int = int(
            self._cfg("head_dim", default=(self.hidden_size // max(self.num_heads, 1)) if self.num_heads else 0)
        )
        self.intermediate_size: int = int(self._cfg("intermediate_size", default=0))
        self.vocab_size: int = int(self._cfg("vocab_size", default=0))
        self.tie_word_embeddings: bool = bool(self._cfg("tie_word_embeddings", default=False))

    @property
    def B(self) -> int:
        return self.profile.batch_size

    @property
    def S(self) -> int:
        return self.profile.seq_length

    @property
    def seq_segments(self) -> list[int]:
        segs = self.profile.seq_segments
        return list(segs) if segs else [self.S]

    @property
    def S_eff(self) -> int:
        """Total effective tokens per batch row (sum of segment lengths)."""
        return sum(self.seq_segments)

    def attention_flops_for_seg(self, seg: int) -> int:
        """Per-layer attention FLOPs for one segment (B rows, O(seg²) each). Override for non-GQA."""
        return flops.mha_attention_flops(
            self.B, seg, self.hidden_size, self.num_heads, self.head_dim, self.num_kv_heads
        )

    def attention_flops(self) -> int:
        """Per-layer attention summed over packed segments (B*ΣS_i²); FFN/logits use S_eff."""
        return sum(self.attention_flops_for_seg(s) for s in self.seq_segments)

    def ffn_flops(self) -> int:
        """Per-layer FFN forward FLOPs over the total effective length."""
        if self.intermediate_size <= 0:
            return 0
        return flops.gated_mlp_flops(self.B, self.S_eff, self.hidden_size, self.intermediate_size)

    def components(self) -> list[Component]:
        # Full-input upper bound; the estimator divides the sum by ``sp_size``.
        comps: list[Component] = []
        attn_per = self.attention_flops()
        ffn_per = self.ffn_flops()
        comps.append(Component(
            f"attention ({self.num_layers} layers)",
            attn_per * self.num_layers,
            note=f"per-layer fwd={attn_per:,}",
        ))
        comps.append(Component(
            f"ffn ({self.num_layers} layers)",
            ffn_per * self.num_layers,
            note=f"per-layer fwd={ffn_per:,}",
        ))
        comps.extend(self.extra_components())
        lg = self.logits_component()
        comps.append(Component(lg.name, lg.fwd_flops, lg.note))
        return comps

    def logits_component(self) -> Component:
        lf = flops.logits_flops(self.B, self.S_eff, self.hidden_size, self.vocab_size)
        return Component("logits (lm_head)", lf, note=f"hidden={self.hidden_size} vocab={self.vocab_size}")

class DenseLLMFlops(LLMFlops):
    """Plain dense GQA + gated-MLP causal LM."""

class MoELLMFlops(LLMFlops):
    """MoE causal LM (Qwen/DeepSeek-MoE config keys: num_experts, num_experts_per_tok, moe_intermediate_size, shared_expert_intermediate_size)."""

    def _parse_config(self) -> None:
        super()._parse_config()
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

class MlaLLMFlops(LLMFlops):
    """Multi-head Latent Attention mixin; set mla_* dims in _parse_config."""

    mla_q_lora_rank: int | None = None
    mla_kv_lora_rank: int | None = None
    mla_qk_nope_head_dim: int = 0
    mla_qk_rope_head_dim: int = 0
    mla_v_head_dim: int | None = None

    def attention_flops(self) -> int:
        return flops.mla_attention_flops(
            self.B, self.S_eff, self.hidden_size, self.num_heads,
            self.mla_q_lora_rank, self.mla_kv_lora_rank,
            self.mla_qk_nope_head_dim, self.mla_qk_rope_head_dim,
            v_head_dim=self.mla_v_head_dim,
        )

class VLMFlops(LLMFlops):
    """LLM recipe with an optional vision tower (override :meth:`vision_components`).

    Vision is part of the full-input total and is divided by ``sp_size`` along
    with the decoder.
    """

    def vision_components(self) -> list[Component]:
        return []

    def components(self) -> list[Component]:
        comps = super().components()
        comps.extend(self.vision_components())
        return comps