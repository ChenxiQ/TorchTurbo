"""Learning-rate scheduler construction for engine-managed optimizers."""

from __future__ import annotations

from functools import partial
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

# Load Transformers' scheduler utilities only when constructing a schedule,
# keeping their transitive dependencies out of this module's import path.


def create_lr_scheduler(
    name: str,
    optimizer: Optimizer,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    scheduler_specific_kwargs: dict[str, Any] | None = None,
) -> LRScheduler:
    """Create a named scheduler with an absolute cosine minimum for each group.

    For an absolute ``min_lr`` in ``cosine_with_min_lr`` or
    ``cosine_warmup_with_min_lr``, each parameter group uses its own
    minimum-to-peak LR ratio. Other scheduler configurations use Hugging
    Face's scheduler factory.
    """
    from transformers.optimization import get_scheduler

    scheduler_kwargs = scheduler_specific_kwargs or {}
    if scheduler_kwargs.get("min_lr") is not None:
        if name == "cosine_with_min_lr":
            return _get_cosine_with_min_lr_schedule(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **scheduler_kwargs,
            )
        if name == "cosine_warmup_with_min_lr":
            return _get_cosine_warmup_with_min_lr_schedule(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **scheduler_kwargs,
            )
    return get_scheduler(
        name,
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs=scheduler_kwargs,
    )


def _get_cosine_with_min_lr_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    *,
    min_lr: float,
    min_lr_rate: float | None = None,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    from transformers.optimization import _get_cosine_schedule_with_warmup_lr_lambda

    if min_lr_rate is not None:
        raise ValueError("Only one of min_lr or min_lr_rate should be set")

    lr_lambdas = [
        partial(
            _get_cosine_schedule_with_warmup_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            min_lr_rate=group_min_lr_rate,
        )
        for group_min_lr_rate in _get_group_min_lr_rates(optimizer, min_lr)
    ]
    return LambdaLR(optimizer, lr_lambdas, last_epoch=last_epoch)


def _get_cosine_warmup_with_min_lr_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    *,
    min_lr: float,
    min_lr_rate: float | None = None,
    num_cycles: float = 0.5,
    warmup_lr_rate: float | None = None,
    last_epoch: int = -1,
) -> LambdaLR:
    from transformers.optimization import (
        _get_cosine_with_min_lr_schedule_with_warmup_lr_rate_lambda,
    )

    if min_lr_rate is not None:
        raise ValueError("Only one of min_lr or min_lr_rate should be set")

    lr_lambdas = [
        partial(
            _get_cosine_with_min_lr_schedule_with_warmup_lr_rate_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            min_lr_rate=group_min_lr_rate,
            warmup_lr_rate=warmup_lr_rate,
        )
        for group_min_lr_rate in _get_group_min_lr_rates(optimizer, min_lr)
    ]
    return LambdaLR(optimizer, lr_lambdas, last_epoch=last_epoch)


def _get_group_min_lr_rates(optimizer: Optimizer, min_lr: float) -> list[float]:
    """Convert an absolute minimum LR into each group's minimum-to-peak ratio."""
    group_min_lr_rates = []
    for index, group in enumerate(optimizer.param_groups):
        # LambdaLR keeps an existing initial_lr as the group's peak.
        base_lr = group.get("initial_lr", group["lr"])
        if not 0 <= min_lr <= base_lr:
            raise ValueError(
                f"min_lr must be between 0 and the initial LR of each parameter "
                f"group; got min_lr={min_lr} for group {index} with initial "
                f"LR={base_lr}"
            )
        group_min_lr_rates.append(min_lr / base_lr if base_lr != 0 else 0.0)
    return group_min_lr_rates
