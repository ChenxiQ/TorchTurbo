from __future__ import annotations

import torch
import torch.nn.functional as F

from ...base import OpImpl, OpImplKind
from ...registry import register_op


def _torch_swiglu(x: torch.Tensor) -> torch.Tensor:
    gate, up = torch.chunk(x, 2, dim=-1)
    return (F.silu(gate) * up).to(x.dtype)


@register_op(
    op_name="swiglu",
    impl_name="torch_swiglu",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchSwiGLU(OpImpl):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return _torch_swiglu(x)
