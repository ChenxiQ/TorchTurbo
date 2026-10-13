from __future__ import annotations

import torch

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="rms_norm",
    impl_name="torch_rms_norm",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchRMSNorm(OpImpl):
    def __call__(self, hidden_states: torch.Tensor, weight: torch.Tensor, eps: float):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return weight * hidden_states.to(input_dtype)
