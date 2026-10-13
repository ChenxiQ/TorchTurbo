"""Framework auto-detector for torchturbo.

Iterates registered adapters (sorted by priority) and calls ``detect()`` on
each one.  The first adapter that returns ``True`` wins.

Usage::

    from torchturbo.detector import FrameworkDetector

    name = FrameworkDetector.detect()          # -> "llamafactory_v0" or None
    all_names = FrameworkDetector.detect_all() # -> ["llamafactory_v0", ...]
"""

from __future__ import annotations

from .logging import get_logger
from .registry import BaseAdapter, get_adapter, get_adapters_by_priority


logger = get_logger(__name__)


class FrameworkDetector:
    """Stateless detector that queries registered adapters."""

    @staticmethod
    def detect() -> str | None:
        """Return the name of the first detected framework, or ``None``."""
        for name, adapter in get_adapters_by_priority():
            try:
                if adapter.detect():
                    logger.info_rank0("Detected framework: %s", name)
                    return name
            except Exception:
                logger.debug("Adapter %r raised during detect()", name, exc_info=True)
        return None

    @staticmethod
    def detect_all() -> list[str]:
        """Return names of **all** detected frameworks (priority order)."""
        detected: list[str] = []
        for name, adapter in get_adapters_by_priority():
            try:
                if adapter.detect():
                    detected.append(name)
            except Exception:
                logger.debug("Adapter %r raised during detect()", name, exc_info=True)
        return detected

    @staticmethod
    def get_adapter(name: str) -> BaseAdapter | None:
        """Convenience: retrieve an adapter by framework name."""
        return get_adapter(name)
