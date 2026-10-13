"""NPU provider for the CANN TND lightning-indexer ops."""

from __future__ import annotations

from typing import Any

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


def _load_ops():
    """Load CANN ops lazily."""
    from cann_ops_transformer import ops as cann_ops

    return cann_ops


@register_op(
    op_name="lightning_indexer",
    impl_name="npu_lightning_indexer",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuLightningIndexer(OpImpl):
    def available(self) -> bool:
        try:
            from cann_ops_transformer import ops

            return callable(ops.lightning_indexer)
        except (ImportError, AttributeError, OSError, RuntimeError):
            return False

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        topk: int,
        *,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        cmp_residual_k: torch.Tensor | None = None,
        metadata: Any = None,
        max_seqlen_q: int = -1,
        layout_q: str = "TND",
        layout_k: str = "TND",
        mask_mode: int = 3,
        cmp_ratio: int = 1,
        return_value: int = 0,
        **kwargs: Any,
    ) -> Any:
        return _load_ops().lightning_indexer(
            query,
            key,
            weights,
            topk,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=cmp_residual_k,
            metadata=metadata,
            max_seqlen_q=max_seqlen_q,
            layout_q=layout_q,
            layout_k=layout_k,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
            return_value=return_value,
            **kwargs,
        )


@register_op(
    op_name="lightning_indexer_metadata",
    impl_name="npu_lightning_indexer_metadata",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuLightningIndexerMetadata(OpImpl):
    """Provider for the metadata entry point paired with lightning_indexer."""

    def available(self) -> bool:
        try:
            from cann_ops_transformer import ops

            return callable(ops.lightning_indexer_metadata)
        except (ImportError, AttributeError, OSError, RuntimeError):
            return False

    def __call__(
        self,
        num_heads_q: int,
        num_heads_k: int,
        head_dim: int,
        *,
        topk: int,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        cmp_residual_k: torch.Tensor,
        batch_size: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        layout_q: str = "TND",
        layout_k: str = "TND",
        mask_mode: int = 3,
        cmp_ratio: int = 1,
        **kwargs: Any,
    ) -> Any:
        return _load_ops().lightning_indexer_metadata(
            num_heads_q=num_heads_q,
            num_heads_k=num_heads_k,
            head_dim=head_dim,
            topk=topk,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=cmp_residual_k,
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            layout_q=layout_q,
            layout_k=layout_k,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
            **kwargs,
        )
