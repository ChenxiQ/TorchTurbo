"""Packed-sequence helpers for DeepSeek-V4."""

import torch


def _packed_compress_block_metadata(
    segment_ids: torch.Tensor,
    ratio: int,
    position_ids: torch.LongTensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-row metadata for complete packed compression blocks."""
    bsz, seqlen = segment_ids.shape
    capacity = seqlen // ratio
    if capacity == 0:
        empty = segment_ids.new_empty((bsz, 0))
        empty_positions = position_ids.new_empty((bsz, 0))
        empty_starts = segment_ids.new_empty((bsz, 0), dtype=torch.long)
        empty_valid = torch.empty(
            (bsz, 0), dtype=torch.bool, device=segment_ids.device
        )
        return empty, empty_positions, empty_starts, empty_valid

    token_positions = torch.arange(seqlen, device=segment_ids.device).view(1, -1)
    positive = segment_ids > 0
    segment_starts = positive & torch.cat(
        (
            torch.ones_like(positive[:, :1]),
            segment_ids[:, 1:] != segment_ids[:, :-1],
        ),
        dim=1,
    )
    segment_start_positions = torch.cummax(
        torch.where(segment_starts, token_positions, torch.zeros_like(token_positions)),
        dim=1,
    ).values
    block_end_ids = torch.cat(
        (
            segment_ids[:, ratio - 1 :],
            segment_ids.new_zeros((bsz, ratio - 1)),
        ),
        dim=1,
    )
    block_candidates = (
        positive
        & ((token_positions - segment_start_positions).remainder(ratio) == 0)
        & (block_end_ids == segment_ids)
    )

    block_slots = (block_candidates.cumsum(dim=1) - 1).clamp_(
        min=0, max=capacity - 1
    )
    block_valid = (
        segment_ids.new_zeros((bsz, capacity), dtype=torch.long)
        .scatter_add_(1, block_slots, block_candidates.long())
        .bool()
    )
    block_ids = segment_ids.new_zeros((bsz, capacity)).scatter_add_(
        1,
        block_slots,
        torch.where(block_candidates, segment_ids, 0),
    )
    block_positions = position_ids.new_zeros((bsz, capacity)).scatter_add_(
        1,
        block_slots,
        torch.where(block_candidates, position_ids, 0),
    )
    block_starts = segment_ids.new_zeros(
        (bsz, capacity), dtype=torch.long
    ).scatter_add_(
        1,
        block_slots,
        torch.where(block_candidates, token_positions, 0),
    )

    # Keep the static capacity to avoid a host-side sync on every compressor
    # forward and recompute. Invalid trailing blocks are masked by block_valid.
    return (
        block_ids,
        block_positions,
        block_starts,
        block_valid,
    )


def compress_packed_kv(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    segment_ids: torch.Tensor,
    position_ids: torch.LongTensor,
    ratio: int,
    head_dim: int,
    overlap: bool,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Pool all complete packed blocks without per-segment device launches."""
    block_ids, block_positions, block_starts, block_valid = (
        _packed_compress_block_metadata(segment_ids, ratio, position_ids)
    )
    num_blocks = block_ids.size(1)
    if num_blocks == 0:
        return None, block_positions

    block_offsets = torch.arange(ratio, device=kv.device).view(1, 1, -1)
    token_indices = block_starts.unsqueeze(-1) + block_offsets
    flat_indices = token_indices.flatten(1)
    block_kv = kv.gather(
        1, flat_indices.unsqueeze(-1).expand(-1, -1, kv.size(-1))
    ).view(kv.size(0), num_blocks, ratio, kv.size(-1))
    block_score = score.gather(
        1, flat_indices.unsqueeze(-1).expand(-1, -1, score.size(-1))
    ).view(score.size(0), num_blocks, ratio, score.size(-1))
    block_score = block_score + ape

    if overlap:
        same_segment = (
            block_valid[:, 1:]
            & block_valid[:, :-1]
            & (block_ids[:, 1:] == block_ids[:, :-1])
            & (block_starts[:, 1:] == block_starts[:, :-1] + ratio)
        )
        same_segment = torch.cat(
            (torch.zeros_like(block_valid[:, :1]), same_segment), dim=1
        )
        previous_kv = torch.cat(
            (
                torch.zeros_like(block_kv[:, :1, :, :head_dim]),
                block_kv[:, :-1, :, :head_dim],
            ),
            dim=1,
        )
        previous_score = torch.cat(
            (
                torch.full_like(
                    block_score[:, :1, :, :head_dim], float("-inf")
                ),
                block_score[:, :-1, :, :head_dim],
            ),
            dim=1,
        )
        previous_kv = torch.where(
            same_segment[:, :, None, None],
            previous_kv,
            torch.zeros_like(previous_kv),
        )
        previous_score = torch.where(
            same_segment[:, :, None, None],
            previous_score,
            torch.full_like(previous_score, float("-inf")),
        )
        block_kv = torch.cat((previous_kv, block_kv[..., head_dim:]), dim=2)
        block_score = torch.cat(
            (previous_score, block_score[..., head_dim:]), dim=2
        )

    valid = block_valid[:, :, None, None]
    block_kv = torch.where(valid, block_kv, 0)
    block_score = torch.where(valid, block_score, 0)
    compressed = (block_kv * block_score.softmax(dim=2)).sum(dim=2)
    return compressed * block_valid.unsqueeze(-1), block_positions


def packed_compress_kv_ids_and_positions(
    segment_ids: torch.Tensor,
    ratio: int,
    position_ids: torch.LongTensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_ids, block_positions, _, _ = _packed_compress_block_metadata(
        segment_ids, ratio, position_ids
    )
    return block_ids, block_positions


def packed_compress_mask(
    segment_ids: torch.Tensor,
    compress_kv_ids: torch.Tensor,
    compress_pos: torch.Tensor,
    ratio: int,
    query_positions: torch.LongTensor,
) -> torch.Tensor:
    """Mask blocks outside each query's segment or physical causal prefix."""
    current_ids = segment_ids.unsqueeze(-1)
    block_ends = compress_pos + ratio
    return (
        (block_ends > query_positions.unsqueeze(-1) + 1)
        | (compress_kv_ids != current_ids)
        | (current_ids == 0)
    )


def _segment_spans(
    row: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    change = torch.nonzero(row[1:] != row[:-1]).flatten() + 1
    boundaries = torch.cat(
        [
            row.new_zeros(1, dtype=torch.long),
            change,
            row.new_tensor([row.numel()], dtype=torch.long),
        ]
    )
    starts, ends = boundaries[:-1], boundaries[1:]
    ids = row[starts]
    positive = ids > 0
    return starts[positive], ends[positive], ids[positive]


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    inverse: bool = False,
    position_ids: torch.LongTensor | None = None,
) -> torch.Tensor:
    """Apply rotary embeddings."""
    original_dtype = x.dtype
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if position_ids is not None:
        freqs_cis = torch.view_as_complex(
            torch.view_as_real(freqs_cis)[position_ids].contiguous()
        )
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_complex.ndim == 3:
        freqs_cis = freqs_cis.view(
            x_complex.size(0), x_complex.size(1), x_complex.size(-1)
        )
    else:
        freqs_cis = freqs_cis.view(
            x_complex.size(0), x_complex.size(1), 1, x_complex.size(-1)
        )
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rotated.to(original_dtype)
