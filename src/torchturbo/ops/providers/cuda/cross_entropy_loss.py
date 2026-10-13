from __future__ import annotations

import torch
import torch.nn.functional as F

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="cross_entropy_loss",
    impl_name="flash_attn_cross_entropy_loss",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.CUDA,),
)
class FlashAttnCrossEntropyLoss(OpImpl):
    def available(self) -> bool:
        try:
            from flash_attn.ops.triton.cross_entropy import (  # noqa: F401
                cross_entropy_loss,
            )
        except ImportError:
            return False
        return True

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        reduction: str = "sum",
        ignore_index: int = -100,
    ):
        if reduction != "sum":
            return F.cross_entropy(
                logits,
                labels,
                reduction=reduction,
                ignore_index=ignore_index,
            )

        from flash_attn.ops.triton.cross_entropy import (
            cross_entropy_loss as fa_cross_entropy_loss,
        )

        loss, _ = fa_cross_entropy_loss(logits, labels, ignore_index=ignore_index)
        if isinstance(loss, tuple):
            loss = loss[0]
        if loss.dim() > 0:
            loss = loss.sum()
        return loss
