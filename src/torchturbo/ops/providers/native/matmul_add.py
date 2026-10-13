from __future__ import annotations

import torch

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="matmul_add",
    impl_name="torch_matmul_add",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchMatmulAdd(OpImpl):
    def __call__(
        self,
        accumulator: torch.Tensor,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        *,
        transpose_lhs: bool = False,
    ) -> None:
        # Native addmm generally requires the accumulator and matmul operands
        # to share a dtype. Per-chunk casts avoid a full [V, H] temporary.
        if transpose_lhs:
            lhs = lhs.t()
        accumulator.addmm_(
            lhs.to(accumulator.dtype),
            rhs.to(accumulator.dtype),
        )
