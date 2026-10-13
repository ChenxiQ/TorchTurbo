"""Adapter registry package.

Provides:
- ``BasePatch`` — interface for monkey-patches
- ``BaseAdapter`` — interface for framework adapters
- ``register_adapter`` / ``get_adapter`` — registry functions
"""

from .adapter import BaseAdapter
from .patch import BasePatch
from .registry import (
    clear_adapters,
    get_adapter,
    get_adapters_by_priority,
    get_all_adapters,
    register_adapter,
)


__all__ = [
    "BaseAdapter",
    "BasePatch",
    "clear_adapters",
    "get_adapter",
    "get_adapters_by_priority",
    "get_all_adapters",
    "register_adapter",
]
