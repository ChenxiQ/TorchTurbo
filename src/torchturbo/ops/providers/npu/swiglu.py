from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="swiglu",
    impl_name="npu_fused_swiglu",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuFusedSwiGLU(OpImpl):
    def available(self) -> bool:
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        import torch_npu

        return torch_npu.npu_swiglu(x, dim=-1)
