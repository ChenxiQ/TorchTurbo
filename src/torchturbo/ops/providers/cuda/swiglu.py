from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


def _torch_swiglu(x: torch.Tensor) -> torch.Tensor:
    gate, up = torch.chunk(x, 2, dim=-1)
    return (F.silu(gate) * up).to(x.dtype)


@register_op(
    op_name="swiglu",
    impl_name="torch_compile_swiglu",
    kind=OpImplKind.CUSTOM,
    priority=50,
    devices=(TurboDevice.CUDA,),
)
class TorchCompileSwiGLU(OpImpl):
    def __init__(self) -> None:
        self._compiled: Any | None = None

    def available(self) -> bool:
        return hasattr(torch, "compile")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self._compiled is None:
            self._compiled = torch.compile(_torch_swiglu)
        return self._compiled(x)
