"""Memory-efficient linear projection plus language-model loss.

The implementation chunks flattened tokens, so ``chunk_size`` has the same
meaning for every batch size. Multiple prediction streams (for example a main
LM objective plus MTP objectives) share one fp32 weight-gradient accumulator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from torchturbo.ops import ops

__all__ = ["chunk_loss"]


class _ChunkedLinearCrossEntropy(torch.autograd.Function):
    """Compute one or more linear-CE objectives with bounded logits memory."""

    @staticmethod
    def forward(
        ctx,
        weight,
        normalizers,
        loss_scales,
        chunk_size,
        ignore_index,
        num_inputs,
        *hidden_and_labels,
    ):
        num_inputs = int(num_inputs)
        hidden_states = hidden_and_labels[:num_inputs]
        labels = hidden_and_labels[num_inputs:]
        needs = ctx.needs_input_grad
        # Zero-scaled streams retain raw-loss diagnostics without a backward edge.
        backward_mask = tuple(loss_scale != 0.0 for loss_scale in loss_scales)
        need_grad_weight = bool(needs[0]) and any(backward_mask)
        need_grad_hidden = [
            bool(needs[6 + i]) and backward_mask[i]
            for i in range(num_inputs)
        ]

        grad_weight = (
            torch.zeros_like(weight, dtype=torch.float32)
            if need_grad_weight
            else None
        )
        grad_inputs: list[torch.Tensor | None] = []
        raw_losses: list[torch.Tensor] = []
        total_loss = torch.zeros((), dtype=torch.float32, device=weight.device)

        for input_idx, (hidden, label) in enumerate(zip(hidden_states, labels)):
            hidden_size = hidden.shape[-1]
            hidden_flat = hidden.reshape(-1, hidden_size)
            labels_flat = label.reshape(-1)
            grad_input = (
                torch.empty_like(hidden) if need_grad_hidden[input_idx] else None
            )
            # view, not reshape: chunks must alias grad_input, and a silent
            # copy of a non-contiguous buffer would drop every chunk's write.
            grad_input_flat = (
                grad_input.view(-1, hidden_size)
                if grad_input is not None
                else None
            )
            raw_loss = torch.zeros((), dtype=torch.float32, device=hidden.device)

            hidden_chunks = torch.split(hidden_flat, chunk_size, dim=0)
            label_chunks = torch.split(labels_flat, chunk_size, dim=0)
            grad_chunks = (
                torch.split(grad_input_flat, chunk_size, dim=0)
                if grad_input_flat is not None
                else (None,) * len(hidden_chunks)
            )
            if len(hidden_chunks) != len(label_chunks):
                raise RuntimeError(
                    "chunk count mismatch: "
                    f"hidden={len(hidden_chunks)} labels={len(label_chunks)}"
                )

            normalizer = normalizers[input_idx]
            loss_scale = loss_scales[input_idx]
            compute_grads = backward_mask[input_idx] and (
                need_grad_hidden[input_idx] or need_grad_weight
            )
            # TODO: Pass CPU-side chunk-validity metadata so fully ignored
            # chunks can skip projection without synchronizing the NPU.
            for hidden_c, labels_c, grad_c in zip(
                hidden_chunks, label_chunks, grad_chunks
            ):
                # The custom function supplies linear's backward explicitly.
                # Detaching here keeps autograd scoped to the cast and CE only.
                linear_output = F.linear(hidden_c, weight).detach()
                linear_output.requires_grad_(compute_grads)

                with torch.enable_grad():
                    logits = linear_output.float()
                    loss_c = (
                        ops.cross_entropy_loss(
                            logits,
                            labels_c,
                            reduction="sum",
                            ignore_index=ignore_index,
                        )
                        / normalizer
                    )
                    scaled_loss_c = loss_c * loss_scale

                raw_loss.add_(loss_c.detach())
                total_loss.add_(scaled_loss_c.detach())

                if compute_grads:
                    # TODO: Use a fused CE backward that returns grad_logits
                    # directly, avoiding per-chunk autograd backward temporaries.
                    (grad_linear_output,) = torch.autograd.grad(
                        scaled_loss_c, linear_output
                    )
                    if need_grad_hidden[input_idx]:
                        torch.mm(
                            grad_linear_output,
                            weight,
                            out=grad_c,
                        )
                    if need_grad_weight:
                        ops.matmul_add(
                            grad_weight,
                            grad_linear_output,
                            hidden_c,
                            transpose_lhs=True,
                        )
                    del grad_linear_output

                del linear_output, logits, loss_c, scaled_loss_c

            grad_inputs.append(grad_input)
            raw_losses.append(raw_loss)

        raw_losses_tensor = torch.stack(raw_losses)
        ctx.mark_non_differentiable(raw_losses_tensor)
        ctx.grad_weight = grad_weight
        ctx.grad_inputs = grad_inputs
        ctx.weight_dtype = weight.dtype
        ctx.num_inputs = num_inputs
        return total_loss, raw_losses_tensor

    @staticmethod
    def backward(ctx, grad_total, _grad_raw_losses):
        grad_weight = ctx.grad_weight
        grad_inputs = ctx.grad_inputs
        ctx.grad_weight = ctx.grad_inputs = None

        # Rescale resident buffers without branching on the device scalar, which
        # would force a synchronization before the accumulated gradients move.
        if grad_total is not None:
            if grad_weight is not None:
                grad_weight.mul_(grad_total.to(grad_weight.dtype))
            for grad_input in grad_inputs:
                if grad_input is not None:
                    grad_input.mul_(grad_total.to(grad_input.dtype))

        if grad_weight is not None:
            grad_weight = grad_weight.to(ctx.weight_dtype)

        # weight, normalizers, scales, chunk_size, ignore_index, num_inputs,
        # then N hidden states and N labels.
        return (
            grad_weight,
            None,
            None,
            None,
            None,
            None,
            *grad_inputs,
            *((None,) * ctx.num_inputs),
        )


def chunk_loss(
    hidden_states: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
    *,
    lm_head: torch.nn.Module | torch.Tensor,
    normalizers: Sequence[float | torch.Tensor],
    loss_scales: Sequence[float],
    chunk_size: int = 1024,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute linear-CE while keeping zero-scaled streams diagnostics-only."""
    if isinstance(lm_head, torch.Tensor):
        weight = lm_head
    elif isinstance(lm_head, torch.nn.Linear):
        if lm_head.bias is not None:
            raise NotImplementedError(
                "chunked linear CE does not support lm_head bias"
            )
        weight = lm_head.weight
    else:
        raise TypeError(
            "chunked linear CE currently supports only nn.Linear lm_head "
            f"modules; got {type(lm_head).__name__}"
        )
    if weight.dim() != 2:
        raise ValueError(f"lm_head weight must be [V, H], got {tuple(weight.shape)}")

    hidden_states = tuple(hidden_states)
    labels = tuple(labels)
    num_inputs = len(hidden_states)
    if num_inputs == 0:
        raise ValueError("at least one LM loss input is required")
    if not (
        len(labels) == len(normalizers) == len(loss_scales) == num_inputs
    ):
        raise ValueError(
            "hidden_states, labels, normalizers and loss_scales must have "
            "the same length"
        )

    for hidden_state, label in zip(hidden_states, labels):
        if hidden_state.dim() != 3:
            raise ValueError(
                f"hidden states must be [B, S, H], got {tuple(hidden_state.shape)}"
            )
        if not hidden_state.is_contiguous():
            raise ValueError(
                "hidden states must be contiguous for chunked loss; "
                f"got shape={tuple(hidden_state.shape)}, "
                f"stride={hidden_state.stride()}"
            )
        if hidden_state.shape[-1] != weight.shape[-1]:
            raise RuntimeError(
                f"lm_head weight {tuple(weight.shape)} does not match hidden "
                f"size {hidden_state.shape[-1]}"
            )
        if label.shape != hidden_state.shape[:2]:
            raise ValueError(
                f"labels {tuple(label.shape)} must match hidden states "
                f"{tuple(hidden_state.shape[:2])}"
            )

    normalizer_scalars: list[torch.Tensor] = []
    for value in normalizers:
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError("normalizer values must be scalar tensors")
            scalar = value.detach().to(device=weight.device, dtype=torch.float32)
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
                scalar_value, device=weight.device, dtype=torch.float32
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
    loss_scale_tuple = tuple(loss_scale_values)

    # Detach zero-scaled streams so ``apply`` never builds a backward edge for
    # them. Keeping the edge would still traverse it once before the Function
    # returns None, which under FSDP unshards that stream's module for nothing.
    hidden_states = tuple(
        hidden_state if loss_scale != 0.0 else hidden_state.detach()
        for hidden_state, loss_scale in zip(hidden_states, loss_scale_tuple)
    )

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    return _ChunkedLinearCrossEntropy.apply(
        weight,
        normalizer_tensor,
        loss_scale_tuple,
        chunk_size,
        int(ignore_index),
        num_inputs,
        *hidden_states,
        *labels,
    )
