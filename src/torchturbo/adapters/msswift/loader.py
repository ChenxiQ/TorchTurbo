"""ms-swift model loader that delegates model construction to the turbo engine.

turbo replaces only ONE thing in swift's model-loading flow: how the ``nn.Module``
is built. swift's native loaders do ``AutoModel.from_pretrained`` in
:meth:`get_model`; turbo instead builds a meta module + FSDP2-shards it + loads
weights via the engine. Everything else — ``get_processor``, ``get_config``,
``load`` — is swift's own and turbo must NOT discard it, because per-model
loaders carry side effects there that matter for data preprocessing (NOT for
the sharded model):

* ``Qwen2VLLoader.get_processor`` runs ``patch_qwen_vl_utils(vision_process)``,
  which reads ``IMAGE_MAX_TOKEN_NUM`` / ``VIDEO_MAX_TOKEN_NUM`` / ``MAX_PIXELS``
  / ``FPS*`` env vars onto the ``qwen_vl_utils.vision_process`` module globals
  that swift's ``dataset.map`` later reads to size image/video tokens. Other
  vendors (glm/internlm/...) have analogous hooks in their own loaders.

So rather than subclass the plain ``ModelLoader`` base (which loses the native
chain), a turbo loader **inherits the model's native swift loader** and only
overrides :meth:`get_model` with the engine path. The relationship comes
straight from swift's own ``MODEL_MAPPING`` — turbo adds no per-model table.
Native ``get_processor`` and all its helpers (``_check_qwen_vl_utils``, ...) are
picked up via normal MRO inheritance.

Pickling: these subclass objects are produced at activation time. They are
defined at module scope by name (published into this module's globals) so
``dill`` pickles them *by reference* via ``(module, __qualname__)`` — a
by-value pickle would trip on ABCMeta's ``_abc._abc_data``. On a forked/spawned
worker, re-importing this module re-runs adapter activation which repopulates
the same globals before any worker resolves the reference. See ``pipeline.py`` /
``args.py`` for the same by-reference reasoning.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from swift.model.register import ModelLoader

from ...logging import get_logger
from . import engine_registry

logger = get_logger(__name__)


_DTYPE_MAP = {
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16, "float16": torch.float16,
    "fp32": torch.float32, "float32": torch.float32,
    "auto": torch.bfloat16,
}


def _resolve_dtype(raw: Any) -> torch.dtype:
    if isinstance(raw, torch.dtype):
        return raw
    if raw is None:
        return torch.bfloat16
    if isinstance(raw, str):
        try:
            return _DTYPE_MAP[raw.lower()]
        except KeyError as exc:
            supported = ", ".join(sorted(_DTYPE_MAP))
            raise ValueError(
                f"Unsupported torch dtype {raw!r}; expected one of: {supported}"
            ) from exc
    return torch.bfloat16


class TurboModelMixin:
    """Engine-backed ``get_model`` mixed ahead of a native swift loader.

    Place this first in a loader's bases so its :meth:`get_model` shadows the
    native one, while leaving every native method (``get_processor`` etc.) on
    the MRO intact. Pure behavior (no ``__init__``) so the native loader's
    construction — ultimately ``ModelLoader.__init__(model_info, model_meta,
    ...)`` — runs unchanged when swift does ``meta.loader(model_info,
    model_meta, ...)``.
    """

    def get_model(self, model_dir, config, processor, model_kwargs):
        engine = engine_registry.get_engine()
        peft_config = getattr(engine, "peft_config", None)
        torch_dtype = _resolve_dtype(getattr(self, "torch_dtype", None))

        if engine_registry.get_loading_role() == "reference":
            # Build the frozen reference through the same engine.
            logger.info_rank0(
                "TurboModelMixin: building reference via turbo engine "
                "(%s, dtype=%s, native_loader=%s)",
                model_dir, torch_dtype, type(self).__mro__[1].__name__,
            )
            return engine.build_model(
                model_name_or_path=model_dir,
                torch_dtype=torch_dtype,
                init_kwargs={"config": config},
            )

        logger.info_rank0(
            "TurboModelMixin: building via turbo engine (%s, dtype=%s, peft=%s, native_loader=%s)",
            model_dir, torch_dtype, "yes" if peft_config is not None else "no",
            type(self).__mro__[1].__name__,
        )
        return engine.prepare_model(
            model_name_or_path=model_dir,
            torch_dtype=torch_dtype,
            init_kwargs={"config": config},
            peft_config=peft_config,
        )

    def _postprocess_model(self, model_dir, model):
        from peft import PeftModel

        if isinstance(model, PeftModel):
            model = model.model
        super()._postprocess_model(model_dir, model)


class TurboModelLoader(TurboModelMixin, ModelLoader):
    """turbo loader for the text-only fallback (no native swift loader).

    Module-level so ``dill`` pickles it by reference. When a native swift
    loader exists for the model, :func:`get_turbo_model_loader` returns a
    per-native subclass instead (see below) — this base is only the fallback.
    """


_SUBCLASS_CACHE: dict = {}


def get_turbo_model_loader(native_loader_cls: Optional[type] = None) -> type:
    """Return a ``TurboModelMixin``-backed loader for the model.

    * ``native_loader_cls is None`` → the text-only :class:`TurboModelLoader`
      (base ``ModelLoader`` chain).
    * otherwise → a subclass ``(TurboModelMixin, native_loader_cls)`` whose
      ``get_model`` is turbo's engine path and whose ``get_processor`` /
      ``get_config`` / helpers are the native loader's (via MRO). The native
      loader class is the model's real swift loader, looked up from swift's
      ``MODEL_MAPPING`` at registration; turbo keeps no model table of its own.

    Subclasses are cached AND published into this module's globals under their
    qualname so ``dill``'s by-reference ``(module, qualname)`` resolves on
    workers (a by-value pickle of an ABCMeta subclass trips on
    ``_abc._abc_data``).
    """
    if native_loader_cls is None:
        return TurboModelLoader
    cached = _SUBCLASS_CACHE.get(native_loader_cls)
    if cached is not None:
        return cached

    qualname = f"TurboModelLoader_{native_loader_cls.__name__}"
    subclass = type(
        qualname,
        (TurboModelMixin, native_loader_cls),
        {},
    )
    subclass.__module__ = __name__
    # Publish under the qualname so dill's by-reference ``getattr(module,
    # qualname)`` resolves on workers (not just the cache dict).
    globals()[qualname] = subclass
    _SUBCLASS_CACHE[native_loader_cls] = subclass
    return subclass
