"""LlamaFactory template for Bailing MoE V3.

Called by the LF adapter's ``_register_lf_templates()`` during activation.
All LlamaFactory imports are isolated here so the model package stays usable
without LlamaFactory installed.
"""

from __future__ import annotations

import json
from typing import Any

from ...logging import get_logger


logger = get_logger(__name__)

LING_V3_TOOL_PROMPT = (
    "# Tools\nYou may call one or more functions to assist with the user query.\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{tool_text}"
    "\n</tools>\nIf none of the functions can be used, point it out. "
    "If the given question lacks the parameters required by the function, also point it out.\n"
    "If you need to use a function, for each function call, output the function name and arguments "
    "within the following XML format:\n"
    "<function=func_name>\n<parameter=arg_key>\narg_value\n</parameter>\n...\n</function>\n"
)


def _register_ling_v3_tool_format() -> None:
    """Register the Ling V3 tool formatter when LlamaFactory lacks it."""
    from llamafactory.data import tool_utils

    tools = getattr(tool_utils, "TOOLS", None)
    qwen_tool_utils = getattr(tool_utils, "QwenToolUtils", None)
    if not isinstance(tools, dict) or qwen_tool_utils is None:
        return
    if "ling_v3" in tools:
        return

    class LingV3ToolUtils(qwen_tool_utils):
        """Ling v3 tool using template."""

        @staticmethod
        def tool_formatter(tools: list[dict[str, Any]]) -> str:
            tool_text = ""
            for tool in tools:
                wrapped_tool = (
                    tool
                    if tool.get("type") == "function"
                    else {"type": "function", "function": tool}
                )
                tool_text += "\n" + json.dumps(wrapped_tool, ensure_ascii=False)
            return LING_V3_TOOL_PROMPT.format(tool_text=tool_text)

    tools["ling_v3"] = LingV3ToolUtils()
    logger.debug("Registered LF tool format 'ling_v3'")


def register() -> None:
    """Register the Bailing MoE V3 chat template with LlamaFactory."""
    from llamafactory.data.formatter import FunctionFormatter, StringFormatter
    from llamafactory.data.formatter import ToolFormatter
    from llamafactory.data.template import TEMPLATES, register_template

    _register_ling_v3_tool_format()

    if "bailing_v3" in TEMPLATES:
        return

    register_template(
        name="bailing_v3",
        format_user=StringFormatter(
            slots=[
                "<role>HUMAN</role>\n{{content}}<|role_end|>\n\n<role>ASSISTANT</role>"
            ]
        ),
        format_system=StringFormatter(
            slots=["<role>SYSTEM</role>\n{{content}}<|role_end|>\n\n"]
        ),
        format_assistant=StringFormatter(slots=["{{content}}<|role_end|>\n\n"]),
        format_observation=StringFormatter(
            slots=[
                "<role>OBSERVATION</role>\n\n```result\n{{content}}\n```\n<|role_end|>\n\n<role>ASSISTANT</role>"
            ]
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|role_end|>\n\n"], tool_format="ling_v3"
        ),
        format_tools=ToolFormatter(tool_format="ling_v3"),
        stop_words=[""],
        efficient_eos=True,
    )
    logger.debug("Registered LF template 'bailing_v3'")
