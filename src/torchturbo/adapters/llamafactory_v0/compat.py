"""Version compatibility checks for LlamaFactory."""

from __future__ import annotations

from ...logging import get_logger


logger = get_logger(__name__)


def get_llamafactory_version() -> str:
    """Return the LlamaFactory version string (e.g. ``"0.9.2"``)."""
    try:
        import llamafactory

        return getattr(llamafactory, "__version__", "0.0.0")
    except ImportError:
        return "0.0.0"


# Keep private alias for backward compat
_get_llamafactory_version = get_llamafactory_version


def is_llamafactory_v0() -> bool:
    """Return ``True`` if the installed LlamaFactory is v0.x.

    v0 versions look like ``"0.9.x"``; v1 versions start with ``"1."``.
    If version cannot be determined, we fall back to checking for v0-specific
    module paths.
    """
    version = _get_llamafactory_version()

    if version is not None:
        try:
            major = int(version.split(".")[0])
            return major == 0
        except (ValueError, IndexError):
            logger.debug("Could not parse LlamaFactory version: %s", version)

    # Fallback: v0 has `llamafactory.train.tuner`, v1 has `llamafactory.v1`
    try:
        import importlib.util

        has_tuner = importlib.util.find_spec("llamafactory.train.tuner") is not None
        has_v1 = importlib.util.find_spec("llamafactory.v1") is not None
        # If it has both, it's a codebase with v1 present — check which is primary
        # v0's entry point is `tuner.run_exp`; v1 uses `v1.trainers`
        if has_tuner and not has_v1:
            return True
        if has_tuner and has_v1:
            # Both present: treat as v0 (v1 is experimental in these builds)
            return True
        return False
    except Exception:
        return False


def is_llamafactory_v1() -> bool:
    """Return ``True`` if the installed LlamaFactory is v1.x."""
    version = _get_llamafactory_version()
    if version is not None:
        try:
            major = int(version.split(".")[0])
            return major >= 1
        except (ValueError, IndexError):
            pass
    return False
