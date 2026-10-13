"""Loss building blocks shared by models, backends and adapters."""

from .chunk_loss import chunk_loss
from .logits_loss import logits_loss

__all__ = ["chunk_loss", "logits_loss"]
