"""Packed-token helpers for RL-style TorchTurbo training."""

from __future__ import annotations

from typing import Any

import torch


def shifted_packed_tensor(
    tensor_full: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    remove_start: int = 0,
    remove_end: int = 1,
) -> torch.Tensor:
    """Remove tokens from the start/end of each sequence in a packed tensor."""

    if cu_seqlens is None:
        n = len(tensor_full)
        if remove_start + remove_end >= n:
            return tensor_full
        return tensor_full[remove_start : n - remove_end]

    shifted_parts = []
    num_seqs = cu_seqlens.shape[0] - 1
    for i in range(num_seqs):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        if end - start > remove_start + remove_end:
            shifted_parts.append(tensor_full[start + remove_start : end - remove_end])
    return torch.cat(shifted_parts, dim=0) if shifted_parts else tensor_full


def _gather_logprobs_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = torch.nn.functional.log_softmax(logits.float() / temperature, dim=-1)
    entropy = -torch.sum(log_probs.exp() * log_probs, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_labels, entropy


def gather_logprobs_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather token log-probabilities and entropy from dense logits in chunks."""

    total = logits.shape[0]
    lp_parts, ent_parts = [], []
    for i in range(0, total, chunk_size):
        end = min(i + chunk_size, total)
        lp, ent = _gather_logprobs_entropy(logits[i:end], labels[i:end], temperature)
        lp_parts.append(lp)
        ent_parts.append(ent)
    return torch.cat(lp_parts), torch.cat(ent_parts)


def get_packed_mb_orig(mb_item: Any) -> dict[str, Any]:
    """Return the original micro-batch view from a dict or attribute object."""
    return mb_item["orig_mb"]


def get_packed_mb_padded(mb_item: Any) -> dict[str, Any]:
    """Return the padded model-input view from a dict or attribute object."""
    return mb_item["padded_mb"]


def prepare_packed_mb_inputs(mb_item: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prepare model inputs and training context from a packed micro-batch item."""

    inputs = get_packed_mb_padded(mb_item).copy()
    pad_length = inputs.pop("_pad_length", 0)
    sp_pad_size = inputs.pop("_sp_pad_size", 0)
    input_ids_rolled = inputs.pop("_input_ids_rolled", None)

    ctx = {
        "model_inputs": inputs,
        "mb_input": get_packed_mb_orig(mb_item),
        "pad_length": pad_length,
        "sp_pad_size": sp_pad_size,
    }
    if input_ids_rolled is not None:
        ctx["mb_input"]["_input_ids_rolled"] = input_ids_rolled

    return inputs, ctx
