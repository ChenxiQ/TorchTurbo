"""Bailing-MoE-V3-VL forward-FLOP recipe (text hybrid MLA+KDA + Qwen-style ViT).

The decoder reuses :class:`BailingMoeV3Flops` unchanged: config lookups fall
back to the nested ``text_config``, so the hybrid MLA/KDA layer split, the
dense/MoE FFN split, and the MTP streams carry over as-is. Vision adds a
Qwen-style ViT tower with an optional PatchMerger projection, per-index
deepstack mergers, and the ``linear_proj`` projector that maps merged visual
tokens into the text hidden size. The recipe returns the full-input upper
bound; the estimator divides the sum by ``sp_size``.
"""

from __future__ import annotations

from ...engine.metrics import op_flops as flops
from ...engine.metrics.flops import Component
from ..bailing_hybrid.mfu import BailingMoeV3Flops

__all__ = ["BailingMoeV3VLFlops"]


class BailingMoeV3VLFlops(BailingMoeV3Flops):
    """Bailing v3 VL theoretical forward FLOPs (text recipe + vision tower)."""

    def _parse_config(self) -> None:
        super()._parse_config()  # text decoder dims via nested text_config

        self.vision_config = self.config.vision_config
        vc = self.vision_config
        # Deepstack needs the merger projections; disable_merger_proj
        # checkpoints (like the current one) skip the whole deepstack path.
        self.deepstack_indexes: list[int] = []
        if not vc.disable_merger_proj:
            self.deepstack_indexes = list(vc.deepstack_visual_indexes or [])

    def _merger_flops_per_item(self, patches: int) -> int:
        vc = self.vision_config
        merge_in = vc.hidden_size * vc.spatial_merge_size**2
        merged = patches // vc.spatial_merge_size**2
        return flops.linear_flops(merged, merge_in, merge_in) + flops.linear_flops(
            merged, merge_in, vc.out_hidden_size
        )

    def vision_components(self) -> list[Component]:
        """Vision-tower forward from the real per-batch ``grid_thw`` lists."""
        prof = self.profile
        vc = self.vision_config
        grids = list(prof.image_grid_thw or []) + list(prof.video_grid_thw or [])
        if not grids:
            return []
        # The tower attends within each temporal patch, so every frame is its
        # own attention sequence. Counting frames also makes per-video and
        # per-frame video grids report the same FLOPs.
        frames = [(1, h, w) for t, h, w in grids for _ in range(t)]
        include_merger = not vc.disable_merger_proj
        encoder, merged = flops.vision_encoder_flops(
            frames, vc, include_merger=include_merger
        )
        deepstack = len(self.deepstack_indexes) * sum(
            self._merger_flops_per_item(h * w) for _, h, w in frames
        )

        # linear_proj: Sequential(in -> H, H -> H) over the merged main stream.
        # Deepstack features are sliced off before the projector. Without the
        # merger projections, its input is the concatenated merge group.
        if vc.disable_merger_proj:
            projector_in = vc.hidden_size * vc.spatial_merge_size**2
        else:
            projector_in = vc.out_hidden_size
        text_hidden = self.hidden_size
        linear_proj = flops.linear_flops(
            merged, projector_in, text_hidden
        ) + flops.linear_flops(merged, text_hidden, text_hidden)

        return [
            Component(
                f"vision encoder ({vc.depth} blocks, {len(grids)} items)",
                encoder + deepstack + linear_proj,
                note=f"real grid_thw; merged_tokens={merged}, "
                f"deepstack mergers={len(self.deepstack_indexes)}, "
                f"include_merger={include_merger}",
            )
        ]

    def components(self) -> list[Component]:
        comps = super().components()
        comps.extend(self.vision_components())
        return comps
