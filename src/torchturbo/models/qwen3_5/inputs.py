"""Packed multimodal input preparation shared by the Qwen3.5 models."""

from bisect import bisect_left
from typing import Any

import torch
import torch.nn.functional as F


def prepare_packed_position_ids(
    model: torch.nn.Module, inputs: dict[str, Any]
) -> dict[str, Any]:
    """Build per-sample mRoPE from flat media grids before SP padding.

    Args:
        model: Qwen3.5 backbone providing its config and get_rope_index.
        inputs: Packed batch with text positions and global cu/max metadata.

    Returns:
        Inputs with text and mRoPE coordinates in a four-row position tensor.
        Text-only batches and precomputed mRoPE positions are unchanged.
    """
    text_positions = inputs["position_ids"]
    if text_positions.ndim == 3:
        return inputs
    input_ids = inputs["input_ids"][0]
    config = model.config
    if inputs.get("image_grid_thw") is None and inputs.get("video_grid_thw") is None:
        media_tokens = (input_ids == config.image_token_id) | (
            input_ids == config.video_token_id
        )
        if not media_tokens.any().item():
            return inputs

    boundaries = inputs["cu_seq_lens_q"].tolist()
    sample_grids = [{} for _ in boundaries[1:]]
    merge_size = config.vision_config.spatial_merge_size
    for key, token_id in (
        ("image_grid_thw", config.image_token_id),
        ("video_grid_thw", config.video_token_id),
    ):
        grids = inputs.get(key)
        if grids is None:
            continue
        # Match sample boundaries to media-token counts, since one sample can
        # contain multiple images and one video grid can describe many frames.
        media_token_offsets = F.pad((input_ids == token_id).cumsum(0), (1, 0))
        sample_token_offsets = media_token_offsets[boundaries].tolist()
        grid_token_counts = grids.prod(-1) // merge_size**2
        grid_token_offsets = [0, *grid_token_counts.cumsum(0).tolist()]
        grid_offsets = [
            bisect_left(grid_token_offsets, offset) for offset in sample_token_offsets
        ]
        for sample, start, end in zip(sample_grids, grid_offsets, grid_offsets[1:]):
            sample[key] = grids[start:end]

    # Only the valid prefix gets media positions; existing tail padding stays 0.
    mm_positions = text_positions.new_zeros((3, *text_positions.shape))
    for start, end, grids in zip(boundaries, boundaries[1:], sample_grids):
        positions, _ = model.get_rope_index(
            input_ids=input_ids[start:end].unsqueeze(0), **grids
        )
        mm_positions[..., start:end] = positions
    inputs["position_ids"] = torch.cat((text_positions.unsqueeze(0), mm_positions), dim=0)
    return inputs
