"""Cross-entropy loss from materialized language-model logits."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from torchturbo.ops import ops

__all__ = ["logits_loss"]


def logits_loss(
    logits: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
    *,
    normalizers: Sequence[float | torch.Tensor],
    loss_scales: Sequence[float],
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute logits CE while keeping zero-scaled streams diagnostics-only."""
    logits = tuple(logits)
    labels = tuple(labels)
    num_inputs = len(logits)
    if num_inputs == 0:
        raise ValueError("at least one LM loss input is required")
    if not (
        len(labels) == len(normalizers) == len(loss_scales) == num_inputs
    ):
        raise ValueError(
            "logits, labels, normalizers and loss_scales must have the same length"
        )

    for logits_i, label in zip(logits, labels):
        if logits_i.dim() != 3:
            raise ValueError(
                f"logits must be [B, S, V], got {tuple(logits_i.shape)}"
            )
        if label.shape != logits_i.shape[:2]:
            raise ValueError(
                f"labels {tuple(label.shape)} must match logits "
                f"{tuple(logits_i.shape[:2])}"
            )

    normalizer_scalars: list[torch.Tensor] = []
    for value in normalizers:
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError("normalizer values must be scalar tensors")
            scalar = value.detach().to(
                device=logits[0].device, dtype=torch.float32
            )
        else:
            scalar_value = float(value)
            if not math.isfinite(scalar_value):
                raise ValueError(
                    f"normalizer values must be finite, got {scalar_value}"
                )
            if scalar_value <= 0:
                raise ValueError(
                    f"normalizer values must be positive, got {scalar_value}"
                )
            scalar = torch.tensor(
                scalar_value, device=logits[0].device, dtype=torch.float32
            )
        normalizer_scalars.append(scalar.reshape(()))
    normalizer_tensor = torch.stack(normalizer_scalars)

    loss_scale_values: list[float] = []
    for value in loss_scales:
        if torch.is_tensor(value):
            raise TypeError(
                "loss scale values must be host scalars so zero-scaled "
                "objectives can be excluded from backward"
            )
        scalar_value = float(value)
        if not math.isfinite(scalar_value):
            raise ValueError(
                f"loss scale values must be finite, got {scalar_value}"
            )
        loss_scale_values.append(scalar_value)

    total_loss: torch.Tensor | None = None
    raw_losses: list[torch.Tensor] = []
    for logits_i, label, normalizer, loss_scale in zip(
        logits, labels, normalizer_tensor, loss_scale_values
    ):
        loss_logits = logits_i if loss_scale != 0.0 else logits_i.detach()
        raw_loss = (
            ops.cross_entropy_loss(
                loss_logits.reshape(-1, loss_logits.shape[-1]),
                label.reshape(-1).to(logits_i.device),
                reduction="sum",
                ignore_index=ignore_index,
            )
            / normalizer
        )
        if loss_scale != 0.0:
            scaled_loss = raw_loss * loss_scale
            total_loss = (
                scaled_loss if total_loss is None else total_loss + scaled_loss
            )
        raw_losses.append(raw_loss.detach())

    if total_loss is None:
        total_loss = torch.zeros(
            (),
            dtype=torch.float32,
            device=logits[0].device,
            requires_grad=torch.is_grad_enabled()
            and any(logits_i.requires_grad for logits_i in logits),
        )
    return total_loss, torch.stack(raw_losses)
