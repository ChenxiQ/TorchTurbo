"""LlamaFactory template registration for Qwen3.5-MoE.

Called by the LF adapter's ``_register_lf_templates()`` during activation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...logging import get_logger
from ...registry import BasePatch

if TYPE_CHECKING:
    from ...engine import TorchTurboEngine

logger = get_logger(__name__)


def register() -> None:
    """Register the qwen3_5 chat templates with LlamaFactory."""
    from llamafactory.data.template import (
        register_template,
        TEMPLATES,
        ReasoningTemplate,
    )
    from llamafactory.data.formatter import StringFormatter
    from llamafactory.data.mm_plugin import get_mm_plugin

    if "qwen3_5" not in TEMPLATES:
        register_template(
            name="qwen3_5",
            format_user=StringFormatter(
                slots=[
                    "<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"
                ]
            ),
            format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
            format_system=StringFormatter(
                slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]
            ),
            stop_words=["<|im_end|>"],
            replace_eos=True,
            template_class=ReasoningTemplate,
            mm_plugin=get_mm_plugin(
                name="qwen3_vl",
                image_token="<|image_pad|>",
                video_token="<|video_pad|>",
            ),
        )
        logger.debug("Registered LF template 'qwen3_5'")

    if "qwen3_5_nothink" not in TEMPLATES:
        register_template(
            name="qwen3_5_nothink",
            format_user=StringFormatter(
                slots=[
                    "<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"
                ]
            ),
            format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
            format_system=StringFormatter(
                slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]
            ),
            stop_words=["<|im_end|>"],
            replace_eos=True,
            mm_plugin=get_mm_plugin(
                name="qwen3_vl",
                image_token="<|image_pad|>",
                video_token="<|video_pad|>",
            ),
        )
        logger.debug("Registered LF template 'qwen3_5_nothink'")


class Qwen35DummyMediaPatch(BasePatch):
    """Skip fake image/audio injection for Qwen3.5 text-only batches."""

    name = "qwen3_5_dummy_media"
    supported_versions = "==0.9.5"

    _PATCHED_MODEL_TYPES = ("qwen3_5_moe", "qwen3_5")

    def __init__(self) -> None:
        self._original_call = None
        self._cls = None

    def patch(self, engine: "TorchTurboEngine | None" = None) -> None:
        from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq

        self._cls = MultiModalDataCollatorForSeq2Seq
        self._original_call = MultiModalDataCollatorForSeq2Seq.__call__
        patched_types = self._PATCHED_MODEL_TYPES
        original_call = self._original_call

        def _patched_call(collator_self, features):
            model_type = getattr(getattr(collator_self, "model", None), "config", None)
            model_type = getattr(model_type, "model_type", None)

            if model_type not in patched_types:
                return original_call(collator_self, features)

            plugin = collator_self.template.mm_plugin
            has_mm = any(feature.get("images") or feature.get("videos") or feature.get("audios") for feature in features)

            # Empty media lists still identify multimodal datasets.
            is_mllm = getattr(collator_self, "_torchturbo_is_mllm", None)
            if is_mllm is None:
                is_mllm = any(
                    isinstance(feature.get("images"), list) or
                    isinstance(feature.get("videos"), list) or
                    isinstance(feature.get("audios"), list)
                    for feature in features
                )
                collator_self._torchturbo_is_mllm = is_mllm

            saved_img = saved_aud = None
            if not has_mm and not is_mllm:
                saved_img, saved_aud = plugin.image_token, plugin.audio_token
                plugin.image_token = None
                plugin.audio_token = None

            try:
                result = original_call(collator_self, features)
            finally:
                if saved_img is not None:
                    plugin.image_token, plugin.audio_token = saved_img, saved_aud

            return result

        MultiModalDataCollatorForSeq2Seq.__call__ = _patched_call
        logger.info_rank0("Qwen35DummyMediaPatch applied for %s", patched_types)

    def revert(self) -> None:
        if self._cls and self._original_call:
            self._cls.__call__ = self._original_call
            self._original_call = None
            self._cls = None


def get_patches() -> list[BasePatch]:
    return [Qwen35DummyMediaPatch()]
