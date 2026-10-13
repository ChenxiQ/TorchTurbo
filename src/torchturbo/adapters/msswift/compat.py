"""ms-swift presence / version detection."""

from __future__ import annotations

import importlib
import importlib.util


def is_swift_installed() -> bool:
    return importlib.util.find_spec("swift") is not None


def get_swift_version() -> str:
    """Installed ms-swift version (``"0.0.0"`` if unknown/uninstalled)."""
    if not is_swift_installed():
        return "0.0.0"
    try:
        import swift

        return getattr(swift, "__version__", "0.0.0")
    except Exception:
        return "0.0.0"


def is_swift_supported() -> bool:
    """True if ms-swift exposes the submodules this adapter hooks into.

    Pins to the post-refactor layout (``swift.pipelines`` / ``swift.model`` /
    ``swift.trainers``), which is what v4.1 ships.
    """
    if not is_swift_installed():
        return False
    for mod in ("swift.pipelines.train.sft", "swift.model.register", "swift.trainers.trainer_factory"):
        try:
            importlib.import_module(mod)
        except Exception:
            return False
    return True