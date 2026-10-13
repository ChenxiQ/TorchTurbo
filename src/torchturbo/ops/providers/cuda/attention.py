"""CUDA attention providers with per-head pre-softmax maxima."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
import os
from typing import TYPE_CHECKING, Any

from packaging.version import Version
import torch
from torch import Tensor, nn

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask


@lru_cache(maxsize=1)
def is_transformer_engine_available() -> bool:
    """Return whether the installed release supports THD attention logit maxima."""
    try:
        installed_version = version("transformer-engine")
    except PackageNotFoundError:
        return False
    return Version(installed_version) >= Version("2.11.0")


@lru_cache(maxsize=128)
def _get_attention(
    num_heads: int,
    qk_head_dim: int,
    value_head_dim: int,
    scaling: float,
    dropout: float,
    qkv_format: str,
    attn_mask_type: str,
    device: torch.device,
    dtype: torch.dtype,
    deterministic: bool,
) -> nn.Module:
    """Reuse DPA state for one execution signature, including its device policy."""
    from transformer_engine.pytorch import DotProductAttention

    # Ulysses has already redistributed the inputs. TE owns no parallel groups.
    return DotProductAttention(
        num_attention_heads=num_heads,
        kv_channels=(qk_head_dim, value_head_dim),
        num_gqa_groups=num_heads,
        attention_dropout=dropout,
        qkv_format=qkv_format,
        attn_mask_type=attn_mask_type,
        softmax_scale=scaling,
        return_max_logit=True,
        sequence_parallel=False,
        tp_size=1,
        tp_group=None,
        cp_group=None,
    )


@register_op(
    op_name="attention_with_max_logits",
    impl_name="te_attention_with_max_logits",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.CUDA,),
)
class TransformerEngineAttentionWithMaxLogits(OpImpl):
    """Execute Transformer Engine attention with per-head maximum logits."""

    def available(self) -> bool:
        return is_transformer_engine_available()

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
        """Execute BSND self-attention and return detached FP32 head maxima.

        Args:
            query: Queries shaped [batch, sequence, heads, qk_head_dim].
            key: Keys with the same shape as query.
            value: Values shaped [batch, sequence, heads, value_head_dim].
            attention_mask: Normalized segment IDs, with zero marking padding.
            scaling: Multiplier applied to QK logits before softmax.
            dropout: Effective attention dropout probability.
            is_causal: Whether attention is causal within each sequence.
            cu_seq_lens_q: Query sequence boundaries in the flattened batch.
            cu_seq_lens_k: Key sequence boundaries in the flattened batch.
            max_length_q: Maximum query sequence length, inferred by TE when omitted.
            max_length_k: Maximum key sequence length, inferred by TE when omitted.

        Returns:
            Attention in BSND layout and one maximum per execution head.
        """
        if attention_mask is not None and attention_mask.ndim == 4:
            raise ValueError("Maximum logits support 4D masks only in the CPU reference.")

        from transformer_engine.pytorch.quantization import FP8GlobalStateManager

        if FP8GlobalStateManager.is_fp8_enabled():
            recipe = FP8GlobalStateManager.get_fp8_recipe()
            if (recipe.fp8_dpa or recipe.fp8_mha) and os.getenv("NVTE_DPA_FP8_RECIPE") != "F16":
                raise ValueError(
                    "Maximum logits do not support Transformer Engine FP8 attention. "
                    "Disable fp8_dpa/fp8_mha or set NVTE_DPA_FP8_RECIPE=F16 before importing TE."
                )

        batch_size, seq_length, num_heads, qk_head_dim = query.shape
        value_head_dim = value.shape[-1]
        if attention_mask is None and cu_seq_lens_q is None:
            qkv_format = "bshd"
            attn_mask_type = "causal" if is_causal else "no_mask"
            attention_kwargs = {}
        elif attention_mask is None:
            qkv_format = "thd"
            attn_mask_type = "padding_causal" if is_causal else "padding"
            query, key, value = (tensor.flatten(0, 1) for tensor in (query, key, value))
            attention_kwargs = {
                "cu_seqlens_q": cu_seq_lens_q.to(dtype=torch.int32),
                "cu_seqlens_kv": cu_seq_lens_k.to(dtype=torch.int32),
                "max_seqlen_q": None if max_length_q is None else int(max_length_q),
                "max_seqlen_kv": None if max_length_k is None else int(max_length_k),
                "pad_between_seqs": False,
            }
        else:
            from torchturbo.utils.flash_attn_utils import (
                get_unpad_data,
                index_first_axis,
                pad_input,
            )

            qkv_format = "thd"
            attn_mask_type = "padding_causal" if is_causal else "padding"
            indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask)
            query = index_first_axis(query.flatten(0, 1), indices)
            key = index_first_axis(key.flatten(0, 1), indices)
            value = index_first_axis(value.flatten(0, 1), indices)
            if indices.numel() == 0:
                # TE cannot execute an empty packed batch; preserve all QKV gradients.
                output = value * 0 + (query.sum() + key.sum()) * 0
                maxima = torch.full(
                    (num_heads,), -torch.inf, device=query.device, dtype=torch.float32
                )
                return pad_input(output, indices, batch_size, seq_length), maxima
            attention_kwargs = {
                "cu_seqlens_q": cu_seqlens,
                "cu_seqlens_kv": cu_seqlens,
                "max_seqlen_q": max_seqlen,
                "max_seqlen_kv": max_seqlen,
                "pad_between_seqs": False,
            }

        # TE captures this policy at construction, so it must partition the cache.
        deterministic = (
            not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))
            or torch.are_deterministic_algorithms_enabled()
        )
        attention = _get_attention(
            num_heads,
            qk_head_dim,
            value_head_dim,
            scaling,
            dropout,
            qkv_format,
            attn_mask_type,
            query.device,
            query.dtype,
            deterministic,
        )
        output, maxima = attention(query, key, value, **attention_kwargs)
        if maxima.shape != (num_heads,):
            raise RuntimeError("Transformer Engine must return one maximum per attention head.")
        if attention_mask is None:
            output = output.reshape(batch_size, seq_length, num_heads, value_head_dim)
        else:
            output = output.reshape(indices.numel(), num_heads, value_head_dim)
            output = pad_input(output, indices, batch_size, seq_length)
        return output, maxima.detach().float()


@lru_cache(maxsize=1)
def _compiled_flex_attention() -> Callable[..., tuple[Tensor, Any]]:
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, fullgraph=True)


@lru_cache(maxsize=1)
def _compiled_create_block_mask() -> Callable[..., BlockMask]:
    from torch.nn.attention.flex_attention import create_block_mask

    # Fuse token-mask construction into block reduction instead of allocating S x S tensors.
    return torch.compile(create_block_mask, fullgraph=True)


@lru_cache(maxsize=16)
def _dense_block_mask(is_causal: bool, seq_length: int, device: torch.device) -> BlockMask:
    def dense_mask_mod(batch, head, query_idx, key_idx):
        return (query_idx >= key_idx) if is_causal else (query_idx >= 0)

    return _compiled_create_block_mask()(
        dense_mask_mod,
        B=None,
        H=None,
        Q_LEN=seq_length,
        KV_LEN=seq_length,
        device=device,
    )


# BlockMask retains its mask_mod closure and mask tensor until eviction.
# The tensor version prevents reusing block metadata after in-place updates.
@lru_cache(maxsize=16)
def _masked_block_mask(
    attention_mask: Tensor,
    is_causal: bool,
    seq_length: int,
    device: torch.device,
    _version: int | None = None,
) -> BlockMask:
    def mask_mod(batch, head, query_idx, key_idx):
        causal = (query_idx >= key_idx) | (not is_causal)
        query_segment = attention_mask[batch, query_idx]
        key_segment = attention_mask[batch, key_idx]
        return causal & (query_segment != 0) & (query_segment == key_segment)

    return _compiled_create_block_mask()(
        mask_mod,
        B=attention_mask.shape[0],
        H=None,
        Q_LEN=seq_length,
        KV_LEN=seq_length,
        device=device,
    )


@register_op(
    op_name="attention_with_max_logits",
    impl_name="flex_attention_with_max_logits",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(TurboDevice.CUDA,),
)
class FlexAttentionWithMaxLogits(OpImpl):
    """Execute PyTorch FlexAttention with per-head maximum logits."""

    def available(self) -> bool:
        try:
            from torch.nn.attention.flex_attention import (  # noqa: F401
                AuxRequest,
                create_block_mask,
                flex_attention,
            )
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
            raise ValueError("Maximum logits support 4D masks only in the CPU reference.")
        if dropout:
            raise ValueError("FlexAttention maximum logits require attention_dropout=0.")

        from torch.nn.attention.flex_attention import AuxRequest

        output_shape = value.shape
        if attention_mask is None and cu_seq_lens_q is not None:
            from torchturbo.utils.flash_attn_utils import cu_seqlens_to_segment_ids

            query, key, value = (
                tensor.flatten(0, 1).unsqueeze(0) for tensor in (query, key, value)
            )
            attention_mask = cu_seqlens_to_segment_ids(cu_seq_lens_q, query.shape[1])

        seq_length = query.shape[1]
        if attention_mask is None:
            if torch.is_inference_mode_enabled():
                block_mask = _dense_block_mask.__wrapped__(is_causal, seq_length, query.device)
            else:
                block_mask = _dense_block_mask(is_causal, seq_length, query.device)
        else:
            if torch.is_inference_mode_enabled() or attention_mask.is_inference():
                block_mask = _masked_block_mask.__wrapped__(
                    attention_mask, is_causal, seq_length, query.device
                )
            else:
                block_mask = _masked_block_mask(
                    attention_mask, is_causal, seq_length, query.device, attention_mask._version
                )
        output, auxiliary = _compiled_flex_attention()(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            block_mask=block_mask,
            scale=scaling,
            return_aux=AuxRequest(max_scores=True),
        )
        output = output.transpose(1, 2).reshape(output_shape)
        return output, auxiliary.max_scores.detach().amax(dim=(0, 2))
