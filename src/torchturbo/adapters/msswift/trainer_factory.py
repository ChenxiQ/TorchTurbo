"""Expose the turbo trainer classes at stable module-level names.

``TrainerFactory.get_cls`` resolves a trainer by importing the dotted-path
module from ``TRAINER_MAPPING`` and ``getattr``-ing the class name, so the
class must be reachable at module level by name. To keep ``import torchturbo``
(and adapter auto-discovery) independent of swift, ``TurboSeq2SeqTrainer`` is
built lazily on first attribute access: importing this module never imports
swift; only accessing ``TurboSeq2SeqTrainer`` (during a swift run, when swift
is installed) builds it. Contributes to swift ↔ llamafactory adapter isolation
— see :func:`torchturbo._discover_builtin_adapters`.
"""

from __future__ import annotations

from .trainer import get_turbo_dpo_trainer, get_turbo_seq2seq_trainer

__all__ = ["TurboSeq2SeqTrainer", "TurboDPOTrainer"]

_cached: dict = {}


def __getattr__(name):
    # PEP 562: build the trainer class on first access so importing this module
    # stays swift-free. ``get_turbo_seq2seq_trainer`` already caches the class;
    # the local cache makes the lazy attribute self-contained and guarantees a
    # single object per process.
    if name == "TurboSeq2SeqTrainer":
        cls = _cached.get(name)
        if cls is None:
            cls = get_turbo_seq2seq_trainer()
            _cached[name] = cls
        return cls
    if name == "TurboDPOTrainer":
        cls = _cached.get(name)
        if cls is None:
            cls = get_turbo_dpo_trainer()
            _cached[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
