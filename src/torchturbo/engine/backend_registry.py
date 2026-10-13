"""Distributed backend registry — register and look up backends by name."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..logging import get_logger

if TYPE_CHECKING:
    from .backends.base import BaseBackend


logger = get_logger(__name__)

_BACKENDS: dict[str, type[BaseBackend]] = {}


def register_backend(name: str, backend_cls: type[BaseBackend]) -> None:
    """Register a backend class under the given name.

    Example::

        register_backend("megatron", MegatronBackend)
    """
    _BACKENDS[name] = backend_cls
    logger.debug("Registered distributed backend: %s -> %s", name, backend_cls.__name__)


def get_backend(name: str) -> type[BaseBackend] | None:
    """Look up a registered backend class by name."""
    return _BACKENDS.get(name)


def get_available_backends() -> list[str]:
    """Return names of all registered distributed backends."""
    return list(_BACKENDS.keys())
