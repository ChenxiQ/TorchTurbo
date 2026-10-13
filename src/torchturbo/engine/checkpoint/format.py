"""Checkpoint products selected by the engine, independent of backend storage."""

from __future__ import annotations

import enum


class CheckpointFormat(enum.Enum):
    """A model-state product written to or restored from a checkpoint.

    ``NATIVE`` preserves exact model state in the backend's native format.
    ``HF`` is an interoperable export. A resumable training checkpoint combines
    model state with optimizer, scheduler, and framework state.
    """

    NATIVE = "native"
    HF = "hf"


class ComponentStatus(enum.Enum):
    """Save state of one checkpoint component."""

    MISSING = "missing"
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"
