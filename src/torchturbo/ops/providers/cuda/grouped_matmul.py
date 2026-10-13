from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="grouped_matmul",
    impl_name="grouped_gemm",
    kind=OpImplKind.CUSTOM,
    priority=0,
    devices=(TurboDevice.CUDA,),
)
class GroupedGemmMatmul(OpImpl):
    def available(self) -> bool:
        try:
            import grouped_gemm  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
        *,
        trans_b: bool = False,
    ):
        import grouped_gemm

        return grouped_gemm.ops.gmm(x, weight, group_list, trans_b=trans_b)
