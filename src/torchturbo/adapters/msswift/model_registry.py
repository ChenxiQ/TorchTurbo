"""Register turbo models and trainers into swift's native registries.

Populate swift's own extension dicts (no monkey-patching of swift internals):

* :data:`swift.model.MODEL_MAPPING` — one ``ModelMeta`` per turbo model, each
  using :class:`TurboModelLoader` (cross-stage; every stage reuses the same
  model loaders).
* :data:`swift.trainers.TrainerFactory.TRAINER_MAPPING` — the turbo trainer
  dotted paths contributed by each registered stage spec
  (:mod:`torchturbo.adapters.msswift.stages`).

Users pick a turbo model with ``--model_type turbo_<name>``; we deliberately do
not set ``architectures``, so swift's auto-detection never collides with its
per-vendor registrations.
"""

from __future__ import annotations

import importlib

from ...logging import get_logger
from ...models.base import TurboVLMModel
from ...models.registry import get_all_models

logger = get_logger(__name__)


def register_turbo_models_to_swift() -> None:
    from swift.model import Model, ModelGroup, ModelMeta, register_model
    from swift.model.model_meta import MODEL_MAPPING

    from .loader import get_turbo_model_loader

    for name, model_cls in get_all_models().items():
        model_type = f"turbo_{name}"
        # Inherit the *native* swift ModelMeta for this model where possible, so
        # turbo's ``turbo_<name>`` registration keeps swift's own per-model
        # conventions (template, requires, tags, is_multimodal, ...) AND — most
        # importantly — the native loader chain. The native loader's
        # ``get_processor`` carries side effects that matter for data
        # preprocessing (e.g. qwen2_vl/qwen3_vl ``patch_qwen_vl_utils`` reading
        # IMAGE_MAX_TOKEN_NUM / VIDEO_MAX_TOKEN_NUM env onto
        # ``qwen_vl_utils.vision_process``). Replacing the loader wholesale (as
        # turbo used to) severed that chain: the env was never read and every
        # image/video was tokenized at the large default caps, changing the
        # ``--max_length`` filter row count vs the native path. We instead hand
        # the native loader class to ``TurboModelLoader``, which delegates
        # ``get_processor`` (and the rest) to it and only overrides ``get_model``
        # with the turbo engine. See ``loader.py``.
        native_meta = MODEL_MAPPING.get(name)
        native_loader_cls = getattr(native_meta, "loader", None) if native_meta else None

        loader_cls = get_turbo_model_loader(native_loader_cls)
        meta = ModelMeta(
            model_type=model_type,
            model_groups=[ModelGroup([Model()], template=getattr(native_meta, "template", None))
                          if native_meta else ModelGroup([Model()])],
            loader=loader_cls,
            template=getattr(native_meta, "template", None),
            architectures=[],
            is_multimodal=isinstance(model_cls, type) and issubclass(model_cls, TurboVLMModel),
            requires=getattr(native_meta, "requires", []) if native_meta else [],
            tags=list(getattr(native_meta, "tags", [])) if native_meta else [],
            task_type=getattr(native_meta, "task_type", "causal_lm") if native_meta else "causal_lm",
        )
        register_model(meta, exist_ok=True)

        # Optional per-model swift template (a ``swift_template.py`` next to the
        # model).  Models without one just rely on the user-supplied --template.
        try:
            mod = importlib.import_module(f"torchturbo.models.{name}.swift_template")
            if hasattr(mod, "register"):
                mod.register()
        except ImportError:
            pass
        except Exception:
            logger.debug_rank0("swift_template registration failed for %r", name, exc_info=True)

    logger.info_rank0("Registered %d turbo models into swift MODEL_MAPPING", len(get_all_models()))


def register_turbo_trainers_to_swift() -> None:
    """Register the turbo trainers into swift (keyed by task/rlhf type)."""
    from swift.trainers import TrainerFactory

    from .stages import TURBO_TRAINERS

    registered = []
    for task_type, dotted in TURBO_TRAINERS.items():
        TrainerFactory.TRAINER_MAPPING[task_type] = dotted
        registered.append(task_type)
    logger.info_rank0("Registered turbo trainers into TrainerFactory: %s", registered)
