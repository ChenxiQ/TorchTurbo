"""CPU maximum-logit attention reference provider."""

import torch
import torch.nn.functional as F
from torch import Tensor

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="attention_with_max_logits",
    impl_name="torch_attention_with_max_logits",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(TurboDevice.CPU,),
)
class TorchAttentionWithMaxLogits(OpImpl):
    """Provide a CPU reference for attention outputs and maximum logits."""

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
        output_shape = value.shape
        if attention_mask is None and cu_seq_lens_q is not None:
            from torchturbo.utils.flash_attn_utils import cu_seqlens_to_segment_ids

            num_tokens = query.shape[0] * query.shape[1]
            query, key, value = (
                tensor.reshape(1, num_tokens, *tensor.shape[2:])
                for tensor in (query, key, value)
            )
            attention_mask = cu_seqlens_to_segment_ids(cu_seq_lens_q, num_tokens)
        batch_size, seq_length, _, _ = query.shape
        scores = query.transpose(1, 2).float() @ key.transpose(1, 2).float().transpose(-1, -2)
        scores = scores * scaling
        allowed = torch.ones(seq_length, seq_length, device=query.device, dtype=torch.bool)
        if is_causal:
            allowed = allowed.tril()
        allowed = allowed[None, None].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                valid = attention_mask != 0
                same_sequence = attention_mask[:, :, None] == attention_mask[:, None, :]
                allowed = allowed & (same_sequence & valid[:, :, None] & valid[:, None, :])[:, None]
            elif attention_mask.dtype == torch.bool:
                allowed = allowed & attention_mask
            else:
                mask_valid = torch.isfinite(attention_mask)
                mask_valid = mask_valid & (attention_mask != torch.finfo(attention_mask.dtype).min)
                allowed = allowed & mask_valid
                scores = scores + attention_mask.float()
        scores = scores.masked_fill(~allowed, -torch.inf)
        max_logits = scores.detach().amax(dim=(0, 2, 3))
        valid_queries = allowed.any(dim=-1, keepdim=True)
        probabilities = scores.masked_fill(~valid_queries, 0).softmax(dim=-1)
        probabilities = probabilities.masked_fill(~valid_queries, 0).to(value.dtype)
        probabilities = F.dropout(probabilities, p=dropout, training=dropout > 0)
        output = probabilities @ value.transpose(1, 2)
        return output.transpose(1, 2).reshape(output_shape), max_logits
