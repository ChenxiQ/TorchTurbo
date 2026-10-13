"""Adapter registry — simple dict + functions."""

from __future__ import annotations

from typing import Any

from ..logging import get_logger
from .adapter import BaseAdapter


logger = get_logger(__name__)


_adapters: dict[str, tuple[BaseAdapter, int]] = {}


def register_adapter(name: str, priority: int = 0):
    """Class decorator to register an adapter.

    Example::

        @register_adapter("llamafactory_v0", priority=10)
        class LFAdapter(BaseAdapter):
            ...
    """

    def decorator(adapter_cls: type[BaseAdapter]) -> type[BaseAdapter]:
        if not issubclass(adapter_cls, BaseAdapter):
            raise TypeError(f"{adapter_cls.__name__} must subclass BaseAdapter")

        instance = adapter_cls()
        instance.framework_name = name
        _adapters[name] = (instance, priority)
        logger.debug("Registered adapter %r (priority=%d)", name, priority)
        return adapter_cls

    return decorator


def get_adapter(name: str) -> BaseAdapter | None:
    """Look up a registered adapter by name."""
    entry = _adapters.get(name)
    return entry[0] if entry else None


def get_all_adapters() -> dict[str, BaseAdapter]:
    """Return all registered adapters as ``{name: adapter}``."""
    return {name: entry[0] for name, entry in _adapters.items()}


def get_adapters_by_priority() -> list[tuple[str, BaseAdapter]]:
    """Return adapters ordered by priority (highest first)."""
    return [
        (name, entry[0])
        for name, entry in sorted(
            _adapters.items(),
            key=lambda item: item[1][1],
            reverse=True,
        )
    ]


def clear_adapters() -> None:
    """Remove all registered adapters (for testing)."""
    _adapters.clear()
