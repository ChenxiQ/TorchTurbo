"""Transient attention statistics consumed by the QK clipping controller."""

import torch
from torch import Tensor, nn


class QKClipState:
    """Accumulate detached maxima in global attention-head order.

    The optimizer-step controller reduces these buffers across workers and
    resets them after clipping. Forward recomputation is safe because maximum
    accumulation is idempotent. While the optimizer is suspended, ``enabled``
    is false and ``get_qk_clip_state`` hides the state from attention.
    """

    max_logits: Tensor
    observed: bool
    enabled: bool

    def __init__(self, max_logits: Tensor) -> None:
        if max_logits.ndim != 1 or max_logits.dtype != torch.float32:
            raise ValueError("QK clipping requires a 1D FP32 statistics tensor.")
        self.max_logits = max_logits
        self.observed = False
        self.enabled = True

    @torch.no_grad()
    def record(self, max_logits: Tensor, head_offset: int = 0) -> None:
        """Accumulate each finite head maximum at its global offset."""
        if max_logits.ndim != 1:
            raise ValueError("QK clipping expects one maximum per attention head.")
        end = head_offset + max_logits.numel()
        if head_offset < 0 or end > self.max_logits.numel():
            raise ValueError("QK clipping head range exceeds the global head count.")
        target = self.max_logits[head_offset:end]
        max_logits = max_logits.detach()
        # Invalid observations must not poison statistics across skipped AMP steps.
        valid_max_logits = torch.where(
            torch.isfinite(max_logits), max_logits, -torch.inf
        )
        torch.maximum(target, valid_max_logits, out=target)
        self.observed = True

    @torch.no_grad()
    def reset(self) -> None:
        """Start a new optimizer-step accumulation window."""
        self.max_logits.fill_(-torch.inf)
        self.observed = False


def get_qk_clip_state(module: nn.Module) -> QKClipState | None:
    """Return active statistics without wrapper attribute forwarding."""
    state = vars(module).get("qk_clip_state")
    return state if state is not None and state.enabled else None
