"""BasePatch — version-aware interface for monkey-patches.

Every patch **must** declare which framework versions it supports.
The adapter enforces this before applying — so version compat can't
be forgotten.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BasePatch(ABC):
    """Abstract base for a single monkey-patch.

    Subclasses **must** set ``supported_versions`` — a PEP 440 version
    specifier (e.g. ``">=0.9,<1.0"``).  The adapter will skip patches
    whose version range doesn't match the detected framework version.

    Example::

        class LoadModelPatch(BasePatch):
            name = "load_model"
            supported_versions = ">=0.9,<1.0"

            def patch(self, engine):
                ...
            def revert(self):
                ...
    """

    name: str = ""
    """Human-readable name for logging."""

    supported_versions: str
    """PEP 440 version specifier, e.g. ``">=0.9,<1.0"``.  **Required.**"""

    @abstractmethod
    def patch(self, engine: Any) -> None:
        """Apply this patch, using the given engine."""

    @abstractmethod
    def revert(self) -> None:
        """Revert this patch to restore original behavior."""

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}"
            f"(name={self.name!r}, versions={self.supported_versions!r})>"
        )
