"""LlamaFactory template + plugin registration for Kimi K2.5.

Called by the LF adapter's ``_register_lf_templates()`` during activation.
All LlamaFactory imports are isolated here — the model directory stays
clean if LF isn't installed.
"""

from __future__ import annotations

from ...logging import get_logger

logger = get_logger(__name__)


def register() -> None:
    """Register the kimi_k25 mm_plugin and template with LlamaFactory."""
    from llamafactory.data.mm_plugin import register_mm_plugin, get_mm_plugin, PLUGINS
    from llamafactory.data.template import (
        register_template,
        TEMPLATES,
        ReasoningTemplate,
    )
    from llamafactory.data.formatter import StringFormatter

    from .mm_plugin import KimiK25Plugin

    # 1. Register the mm_plugin
    if "kimi_k25" not in PLUGINS:
        register_mm_plugin("kimi_k25", KimiK25Plugin)
        logger.debug("Registered LF mm_plugin 'kimi_k25'")

    # 2. Register the template
    if "kimi_k25" not in TEMPLATES:
        register_template(
            name="kimi_k25",
            format_user=StringFormatter(
                slots=[
                    "<|im_user|>user<|im_middle|>{{content}}<|im_end|><|im_assistant|>assistant<|im_middle|>"
                ]
            ),
            format_assistant=StringFormatter(slots=["{{content}}<|im_end|>"]),
            format_system=StringFormatter(
                slots=["<|im_system|>system<|im_middle|>{{content}}<|im_end|>"]
            ),
            default_system="You are a helpful assistant",
            stop_words=["<|im_end|>"],
            replace_eos=True,
            mm_plugin=get_mm_plugin("kimi_k25", image_token="<|media_pad|>"),
        )
        logger.debug("Registered LF template 'kimi_k25'")
