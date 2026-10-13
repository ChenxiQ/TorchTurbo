"""Native NPU attention with HF padding helpers and per-head maxima."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import transformers.modeling_flash_attention_utils as hf_attention

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


def _npu_fusion_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scaling: float,
    dropout: float,
    is_causal: bool,
    *,
    cu_seqlens_q: Tensor | None = None,
    cu_seqlens_k: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    from torch_npu import npu_fusion_attention
    from transformers.integrations.npu_flash_attention import SPARSE_MODE, get_attn_mask_npu

    num_heads = query.shape[-2]
    if query.numel() == 0:
        # Empty unpadding must retain zero gradients for every Q/K/V input.
        output_shape = (*query.shape[:-1], value.shape[-1])
        output = query.new_zeros(output_shape) + (query.sum() + key.sum() + value.sum()) * 0
        head_max = torch.full((num_heads,), -torch.inf, dtype=torch.float32, device=query.device)
        return output, head_max

    kernel_kwargs: dict[str, Any] = {
        "head_num": num_heads,
        "scale": scaling,
        "keep_prob": 1.0 - dropout,
    }
    if is_causal:
        kernel_kwargs["atten_mask"] = get_attn_mask_npu(query.device)
        kernel_kwargs["sparse_mode"] = SPARSE_MODE

    if cu_seqlens_q is None:
        output, softmax_max, *_ = npu_fusion_attention(
            query, key, value, input_layout="BSND", **kernel_kwargs
        )
        if softmax_max.shape != (query.shape[0], num_heads, query.shape[1], 8):
            raise RuntimeError("NPU BSND attention returned an unsupported softmax_max layout.")
        return output, softmax_max.detach().amax(dim=(0, 2, 3))

    query_boundaries = tuple(cu_seqlens_q[1:].tolist())
    key_boundaries = tuple(cu_seqlens_k[1:].tolist())
    output, softmax_max, *_ = npu_fusion_attention(
        query,
        key,
        value,
        input_layout="TND",
        actual_seq_qlen=query_boundaries,
        actual_seq_kvlen=key_boundaries,
        **kernel_kwargs,
    )
    if softmax_max.shape != (query.shape[0], num_heads, 8):
        raise RuntimeError("NPU TND attention returned an unsupported softmax_max layout.")

    # TND statistics store each sequence in head-major order despite their
    # [total_tokens, num_heads, 8] shape.
    starts = (0, *query_boundaries[:-1])
    sequence_sizes = tuple((end - start) * num_heads for start, end in zip(starts, query_boundaries))
    sequences = softmax_max.detach().view(-1, 8).split(sequence_sizes)
    sequence_maxima = [
        sequence.view(num_heads, -1, 8).amax(dim=(1, 2))
        for sequence in sequences
        if sequence.numel()
    ]
    return output, torch.stack(sequence_maxima).amax(dim=0)


def npu_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    scaling: float,
    dropout: float,
    is_causal: bool,
    *,
    cu_seq_lens_q: Tensor | None = None,
    cu_seq_lens_k: Tensor | None = None,
    max_length_q: int | Tensor | None = None,
    max_length_k: int | Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Run native NPU attention and retain its per-head maximum logits.

    The attention interface supplies dtype-corrected BSND tensors and
    optional sequence metadata. HF helpers handle padding and output restoration.

    Args:
        query: Query activations in BSND layout after any SP exchange.
        key: Key activations in BSND layout.
        value: Value activations in BSND layout.
        attention_mask: Optional 2D segment IDs, with zero marking padding.
        scaling: Scale applied to the QK product.
        dropout: Attention probability dropout.
        is_causal: Whether to apply causal attention.
        cu_seq_lens_q: Query sequence boundaries when no mask is supplied.
        cu_seq_lens_k: Key sequence boundaries when no mask is supplied.
        max_length_q: Accepted for the shared op interface; unused by NPU.
        max_length_k: Accepted for the shared op interface; unused by NPU.

    Returns:
        BSND output and detached FP32 maxima in execution-head order.
    """
    if attention_mask is None:
        if cu_seq_lens_q is not None:
            output, head_max = _npu_fusion_attention(
                query.flatten(0, 1), key.flatten(0, 1), value.flatten(0, 1),
                scaling, dropout, is_causal,
                cu_seqlens_q=cu_seq_lens_q, cu_seqlens_k=cu_seq_lens_k,
            )
            return output.reshape(*query.shape[:-1], value.shape[-1]), head_max
        return _npu_fusion_attention(
            query, key, value, scaling, dropout, is_causal
        )

    batch_size, query_length = query.shape[:2]
    if torch.any(attention_mask > 1):
        from torchturbo.utils.flash_attn_utils import get_unpad_data

        # HF's unpadding helper sums a binary mask. Packed IDs need their
        # own cumulative lengths, while the actual indexing is shared.
        indices_q, cu_seq_lens_q, _ = get_unpad_data(attention_mask)
        cu_seq_lens_k = cu_seq_lens_q
        q, k, v = (
            hf_attention._index_first_axis(tensor, indices_q)
            for tensor in (query, key, value)
        )
    else:
        q, k, v, indices_q, lengths, _ = hf_attention._upad_input(
            query, key, value, attention_mask, query_length,
            hf_attention._unpad_input,
        )
        cu_seq_lens_q, cu_seq_lens_k = lengths

    output, head_max = _npu_fusion_attention(
        q, k, v, scaling, dropout, is_causal,
        cu_seqlens_q=cu_seq_lens_q, cu_seqlens_k=cu_seq_lens_k,
    )
    output = hf_attention._pad_input(output, indices_q, batch_size, query_length)
    return output, head_max


@register_op(
    op_name="attention_with_max_logits",
    impl_name="npu_attention_with_max_logits",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuAttentionWithMaxLogits(OpImpl):
    """Expose native NPU attention statistics through the op dispatcher."""

    def available(self) -> bool:
        try:
            from torch_npu import npu_fusion_attention  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        scaling: float,
        dropout: float,
        is_causal: bool,
        *,
        cu_seq_lens_q: Tensor | None = None,
        cu_seq_lens_k: Tensor | None = None,
        max_length_q: int | Tensor | None = None,
        max_length_k: int | Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if attention_mask is not None and attention_mask.ndim == 4:
            raise ValueError(
                "NPU attention with maximum logits does not support 4D masks."
            )
        return npu_attention_forward(
            query, key, value, attention_mask, scaling, dropout, is_causal,
            cu_seq_lens_q=cu_seq_lens_q, cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q, max_length_k=max_length_k,
        )
