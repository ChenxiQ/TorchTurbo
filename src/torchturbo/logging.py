"""Logging utilities for torchturbo.

The log level is configured by the caller. Rank-0 helpers avoid duplicate
messages in distributed runs, and log output is compatible with tqdm progress
bars.

Usage::

    from torchturbo.logging import LogLevel, get_logger, logging_configure

    logging_configure(level=LogLevel.DEBUG)
    logger = get_logger(__name__)
    logger.info_rank0("hello")
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any, Callable

# Root logger name for torchturbo.
_LIBRARY_NAME = "torchturbo"

# Match the log format used by LlamaFactory and Transformers.
_FORMAT = "[%(levelname)s|%(asctime)s] %(name)s:%(lineno)s >> %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


class LogLevel(Enum):
    """Log levels supported by torchturbo."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


class _TqdmStream:
    """Write through ``tqdm.write`` without disrupting progress bars."""

    def __init__(self) -> None:
        self._fallback = __import__("sys").stdout

    def _writer(self) -> Callable[[str], Any]:
        try:
            from tqdm.auto import tqdm  # type: ignore[import-untyped]

            return lambda msg: tqdm.write(msg, end="")
        except Exception:
            return self._fallback.write

    def write(self, msg: str) -> int:
        written = self._writer()(msg)
        return len(msg) if written is None else written

    def flush(self) -> None:
        try:
            self._fallback.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        try:
            return self._fallback.isatty()
        except Exception:
            return False

    def reconfigure(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        getattr(self._fallback, "reconfigure", lambda *a, **k: None)(*args, **kwargs)


class TqdmStreamHandler(logging.StreamHandler):
    """Logging handler that writes through ``tqdm`` when available."""

    def __init__(self) -> None:
        super().__init__(_TqdmStream())


class TurboLogger:
    """Wrapper around :class:`logging.Logger` with distributed helpers.

    Plain methods log on every rank. ``*_rank0`` methods log only on rank 0,
    and ``*_rank0_once`` methods log once per logger instance on rank 0.
    """

    __slots__ = ("_name", "_log", "_rank0", "_once")

    @staticmethod
    def is_rank0() -> bool:
        """Return whether the current process is rank 0."""
        rank = os.getenv("LOCAL_RANK") or os.getenv("RANK")
        if rank is None:
            return True
        try:
            return int(rank) == 0
        except ValueError:
            return rank == "0"

    def __init__(self, name: str) -> None:
        self._name = name
        self._log = logging.getLogger(name)
        self._rank0 = self.is_rank0()
        self._once: set[tuple] = set()

    def _rank0_once(self, args: tuple, kwargs: dict) -> bool:
        """Return whether a message should be emitted once on rank 0."""
        if not self._rank0:
            return False
        key = (args, tuple(sorted(kwargs.items())))
        if key in self._once:
            return False
        self._once.add(key)
        return True

    def info(self, *args: Any, **kwargs: Any) -> None:
        self._log.info(*args, **kwargs)

    def info_rank0(self, *args: Any, **kwargs: Any) -> None:
        if self._rank0:
            kwargs.setdefault("stacklevel", 2)
            self._log.info(*args, **kwargs)

    def info_rank0_once(self, *args: Any, **kwargs: Any) -> None:
        if self._rank0_once(args, kwargs):
            kwargs.setdefault("stacklevel", 2)
            self._log.info(*args, **kwargs)

    def debug(self, *args: Any, **kwargs: Any) -> None:
        self._log.debug(*args, **kwargs)

    def debug_rank0(self, *args: Any, **kwargs: Any) -> None:
        if self._rank0:
            kwargs.setdefault("stacklevel", 2)
            self._log.debug(*args, **kwargs)

    def debug_rank0_once(self, *args: Any, **kwargs: Any) -> None:
        if self._rank0_once(args, kwargs):
            kwargs.setdefault("stacklevel", 2)
            self._log.debug(*args, **kwargs)

    def warning(self, *args: Any, **kwargs: Any) -> None:
        self._log.warning(*args, **kwargs)

    def warning_rank0(self, *args: Any, **kwargs: Any) -> None:
        if self._rank0:
            kwargs.setdefault("stacklevel", 2)
            self._log.warning(*args, **kwargs)

    def warning_rank0_once(self, *args: Any, **kwargs: Any) -> None:
        if self._rank0_once(args, kwargs):
            kwargs.setdefault("stacklevel", 2)
            self._log.warning(*args, **kwargs)

    def error(self, *args: Any, **kwargs: Any) -> None:
        self._log.error(*args, **kwargs)

    def error_rank0(self, *args: Any, **kwargs: Any) -> None:
        if self._rank0:
            kwargs.setdefault("stacklevel", 2)
            self._log.error(*args, **kwargs)

    def error_rank0_once(self, *args: Any, **kwargs: Any) -> None:
        if self._rank0_once(args, kwargs):
            kwargs.setdefault("stacklevel", 2)
            self._log.error(*args, **kwargs)


def logging_configure(level: LogLevel = LogLevel.INFO) -> None:
    """Set the torchturbo root logger level."""
    if not isinstance(level, LogLevel):
        raise TypeError(f"level must be a LogLevel, got {type(level).__name__}")
    logging.getLogger(_LIBRARY_NAME).setLevel(level.value)


def get_logger(name: str) -> TurboLogger:
    """Return a :class:`TurboLogger` for *name*."""
    return TurboLogger(name)


def _install() -> None:
    """Install the default handler for the torchturbo root logger."""
    root = logging.getLogger(_LIBRARY_NAME)
    handler = TqdmStreamHandler()
    handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT))
    root.addHandler(handler)
    root.propagate = False


_install()
