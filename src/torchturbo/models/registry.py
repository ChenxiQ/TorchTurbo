"""Model registry — register_model / get_model."""

from __future__ import annotations

from ..logging import get_logger


logger = get_logger(__name__)

_models: dict[str, type] = {}


def register_model(
    name: str,
    config_cls: type | None = None,
    auto_class: type | None = None,
    exist_ok: bool = True,
):
    """Class decorator to register a model.

    Example::

        @register_model("turbo_llama")
        class TurboLlamaForCausalLM(PreTrainedModel):
            ...
    """

    def decorator(model_cls: type) -> type:
        _models[name] = model_cls
        logger.debug("Registered model %r", name)

        if config_cls is not None and auto_class is not None:
            try:
                from transformers import AutoConfig

                AutoConfig.register(name, config_cls, exist_ok=exist_ok)
                auto_class.register(config_cls, model_cls, exist_ok=exist_ok)
                logger.debug("Registered %r with HuggingFace AutoClass", name)
            except Exception:
                logger.debug(
                    "Failed to register %r with HuggingFace AutoClass",
                    name,
                    exc_info=True,
                )

        return model_cls

    return decorator


def get_model(name: str) -> type | None:
    """Look up a registered model class by name."""
    return _models.get(name)


def get_all_models() -> dict[str, type]:
    """Return all registered models."""
    return dict(_models)


def _register_builtins() -> None:
    """Auto-discover and register built-in models."""
    import importlib
    import pkgutil
    from pathlib import Path

    package_dir = Path(__file__).resolve().parent
    for _, module_name, is_pkg in pkgutil.iter_modules([str(package_dir)]):
        if is_pkg and module_name != "modules":
            model_file = package_dir / module_name / "model.py"
            if model_file.is_file():
                try:
                    importlib.import_module(
                        f".{module_name}.model", package=__package__
                    )
                except Exception as e:
                    logger.warning_rank0("Failed to load model %r: %s", module_name, e)
