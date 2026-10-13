"""Built-in distributed backends.

Importing this package auto-registers all built-in backends.
"""

from .fsdp2 import FSDP2Backend


__all__ = ["FSDP2Backend"]
