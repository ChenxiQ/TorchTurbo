"""BaseAdapter — version-aware adapter with automatic patch filtering.

Adapters **must** implement ``get_framework_version()`` so the patch
system can check compatibility automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..logging import get_logger
from .patch import BasePatch


logger = get_logger(__name__)


def _version_matches(version: str, specifier: str) -> bool:
    """Check if ``version`` satisfies the PEP 440 ``specifier``.

    Uses ``packaging`` if available, else falls back to simple string
    prefix matching for common patterns like ``>=0.9,<1.0``.
    """
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        return Version(version) in SpecifierSet(specifier, prereleases=True)
    except ImportError:
        # Fallback: very basic check
        logger.warning_rank0(
            "packaging not installed — skipping strict version check "
            "for %s against %s",
            version,
            specifier,
        )
        return True  # optimistic: assume compatible


class BaseAdapter(ABC):
    """Abstract base for framework adapters.

    Subclass this and implement:
    - ``detect()`` — is the framework active?
    - ``activate()`` — register patches
    - ``get_framework_version()`` — current framework version string

    When ``apply_patches`` is called, each patch's ``supported_versions``
    is checked against ``get_framework_version()``.  Incompatible patches
    are skipped with a warning.
    """

    framework_name: str = ""

    def __init__(self) -> None:
        self._patches: list[BasePatch] = []

    @abstractmethod
    def detect(self) -> bool:
        """Return ``True`` if the target framework is active."""

    @abstractmethod
    def activate(self) -> None:
        """Register patches and apply top-level intercepts."""

    @abstractmethod
    def get_framework_version(self) -> str:
        """Return the current framework version string (e.g. ``"0.9.2"``)."""

    def deactivate(self) -> None:
        """Undo patches (calls ``revert_patches``)."""
        self.revert_patches()

    def add_patch(self, patch: BasePatch) -> None:
        """Register a patch."""
        self._patches.append(patch)

    def apply_patches(self, engine: Any) -> None:
        """Apply all registered patches, skipping version-incompatible ones."""
        version = self.get_framework_version()

        for p in self._patches:
            if not _version_matches(version, p.supported_versions):
                logger.warning_rank0(
                    "Skipping patch %r: needs %s, but framework is %s",
                    p.name or p.__class__.__name__,
                    p.supported_versions,
                    version,
                )
                continue

            logger.info_rank0(
                "Applying patch: %s (versions=%s, framework=%s)",
                p.name or p.__class__.__name__,
                p.supported_versions,
                version,
            )
            p.patch(engine)

    def revert_patches(self) -> None:
        """Revert all patches in reverse order."""
        for p in reversed(self._patches):
            logger.info_rank0("Reverting patch: %s", p.name or p.__class__.__name__)
            p.revert()
        self._patches.clear()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(framework={self.framework_name!r})>"
