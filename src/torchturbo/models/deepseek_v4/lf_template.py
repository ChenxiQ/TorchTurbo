"""LlamaFactory template for DeepSeek V4.

Called by the LF adapter's ``_register_lf_templates()`` during activation.
All LlamaFactory imports are isolated here so the model package stays usable
without LlamaFactory installed.
"""

from __future__ import annotations

import json
import re
from typing import Any, Union

from ...logging import get_logger


logger = get_logger(__name__)

DEEPSEEK_V4_TOOL_PROMPT = """## Tools

You have access to a set of tools you can use to answer the user's question.
You can invoke functions by writing a "<｜DSML｜function_calls>" block like the following as part of your reply to the user:
<｜DSML｜function_calls>
<｜DSML｜invoke name="$FUNCTION_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
...
</｜DSML｜invoke>
<｜DSML｜invoke name="$FUNCTION_NAME2">
...
</｜DSML｜invoke>
</｜DSML｜function_calls>

String and scalar parameters should be specified as is without any escaping or quotes, while lists and objects should use JSON format. The "string" attribute should be set to "true" for string type parameters and "false" for other types (numbers, booleans, arrays, objects).

If the thinking_mode is enabled, then after function results you should strongly consider outputting a thinking block. Here is an example:

<｜DSML｜function_calls>
...
</｜DSML｜function_calls>

<function_results>
...
</function_results>

<think>...thinking about results</think>

Here are the functions available in JSONSchema format:
<functions>
{tool_text}
</functions>
"""


def _register_deepseek_v4_tool_format() -> None:
    """Register the DeepSeek V4 DSML tool formatter."""
    from llamafactory.data import tool_utils
    from llamafactory.data.tool_utils import FunctionCall, ToolUtils

    tools = getattr(tool_utils, "TOOLS", None)
    if not isinstance(tools, dict):
        return
    if "deepseek_v4" in tools:
        return

    class DeepSeekV4ToolUtils(ToolUtils):
        """DeepSeek V4 tool using DSML template."""

        DSML_TOKEN = "｜DSML｜"

        @staticmethod
        def tool_formatter(tools: list[dict[str, Any]]) -> str:
            tool_texts = []
            for tool in tools:
                tool = tool.get("function", "") if tool.get("type") == "function" else tool
                tool_texts.append(json.dumps(tool, ensure_ascii=False))

            return DEEPSEEK_V4_TOOL_PROMPT.format(tool_text="\n".join(tool_texts))

        @staticmethod
        def function_formatter(functions: list["FunctionCall"]) -> str:
            tool_calls = []
            for name, arguments in functions:
                args_dict = json.loads(arguments)
                params_str = "\n".join([
                    f'<｜DSML｜parameter name="{k}" string="{"true" if isinstance(v, str) else "false"}">{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}</｜DSML｜parameter>'
                    for k, v in args_dict.items()
                ])
                tool_call = f'<｜DSML｜invoke name="{name}">\n{params_str}\n</｜DSML｜invoke>'
                tool_calls.append(tool_call)

            newline = "\n"
            return f'<｜DSML｜function_calls>\n{newline.join(tool_calls)}\n</｜DSML｜function_calls>'

        @staticmethod
        def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
            function_calls_start = f"<{DeepSeekV4ToolUtils.DSML_TOKEN}function_calls>"
            function_calls_end = f"</{DeepSeekV4ToolUtils.DSML_TOKEN}function_calls>"
            invoke_start = f"<{DeepSeekV4ToolUtils.DSML_TOKEN}invoke"
            invoke_end = f"</{DeepSeekV4ToolUtils.DSML_TOKEN}invoke>"
            parameter_start = f"<{DeepSeekV4ToolUtils.DSML_TOKEN}parameter"
            parameter_end = f"</{DeepSeekV4ToolUtils.DSML_TOKEN}parameter>"

            # Check if content contains function calls
            if function_calls_start not in content:
                return content

            results = []

            # Find the function_calls block
            start_idx = content.find(function_calls_start)
            if start_idx == -1:
                return content

            start_idx += len(function_calls_start)
            end_idx = content.find(function_calls_end, start_idx)
            if end_idx == -1:
                return content

            calls_block = content[start_idx:end_idx].strip()

            # Parse individual invoke blocks
            idx = 0
            while idx < len(calls_block):
                invoke_pos = calls_block.find(invoke_start, idx)
                if invoke_pos == -1:
                    break

                # Extract function name
                name_match = re.search(r'name="([^"]+)"', calls_block[invoke_pos:invoke_pos + 200])
                if not name_match:
                    return content
                tool_name = name_match.group(1)

                # Find the end of this invoke block
                invoke_content_start = calls_block.find(">", invoke_pos) + 1
                invoke_content_end = calls_block.find(invoke_end, invoke_content_start)
                if invoke_content_end == -1:
                    return content

                invoke_content = calls_block[invoke_content_start:invoke_content_end].strip()

                # Parse parameters
                args_dict = {}
                param_idx = 0
                while param_idx < len(invoke_content):
                    param_pos = invoke_content.find(parameter_start, param_idx)
                    if param_pos == -1 or param_pos >= len(invoke_content):
                        break

                    # Extract parameter name and string attribute
                    param_tag_end = invoke_content.find(">", param_pos)
                    param_tag = invoke_content[param_pos:param_tag_end + 1]

                    name_attr_match = re.search(r'name="([^"]+)"', param_tag)
                    string_attr_match = re.search(r'string="([^"]+)"', param_tag)

                    if not name_attr_match:
                        return content

                    param_name = name_attr_match.group(1)
                    is_string = string_attr_match.group(1) == "true" if string_attr_match else False

                    # Extract parameter value
                    value_start = param_tag_end + 1
                    value_end = invoke_content.find(parameter_end, value_start)
                    if value_end == -1:
                        return content

                    param_value = invoke_content[value_start:value_end]

                    # Parse value based on type
                    if is_string:
                        args_dict[param_name] = param_value
                    else:
                        try:
                            args_dict[param_name] = json.loads(param_value)
                        except json.JSONDecodeError:
                            args_dict[param_name] = param_value

                    param_idx = value_end + len(parameter_end)

                results.append(FunctionCall(tool_name, json.dumps(args_dict, ensure_ascii=False)))
                idx = invoke_content_end + len(invoke_end)

            return results if results else content

    tools["deepseek_v4"] = DeepSeekV4ToolUtils()
    logger.debug("Registered LF tool format 'deepseek_v4'")


def register() -> None:
    """Register the DeepSeek V4 chat templates with LlamaFactory."""
    from llamafactory.data.formatter import EmptyFormatter, FunctionFormatter
    from llamafactory.data.formatter import StringFormatter, ToolFormatter
    from llamafactory.data.template import ReasoningTemplate, TEMPLATES, register_template

    _register_deepseek_v4_tool_format()

    if "deepseek_v4" not in TEMPLATES:
        register_template(
            name="deepseek_v4",
            format_user=StringFormatter(slots=["<｜User｜>{{content}}<｜Assistant｜>"]),
            format_assistant=StringFormatter(slots=["{{content}}<｜end▁of▁sentence｜>"]),
            format_system=StringFormatter(slots=["{{content}}\n\n"]),
            format_function=FunctionFormatter(slots=["{{content}}<｜end▁of▁sentence｜>"], tool_format="deepseek_v4"),
            format_observation=StringFormatter(slots=["\n<function_results>\n<result>{{content}}</result>\n</function_results>"]),
            format_tools=ToolFormatter(tool_format="deepseek_v4"),
            format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
            stop_words=["<｜end▁of▁sentence｜>"],
            thought_words=("<think>", "</think>"),
            enable_thinking=False,
        )
        logger.debug("Registered LF template 'deepseek_v4'")

    if "deepseek_v4_thinking" not in TEMPLATES:
        register_template(
            name="deepseek_v4_thinking",
            format_user=StringFormatter(slots=["<｜User｜>{{content}}<｜Assistant｜>"]),
            format_assistant=StringFormatter(slots=["{{content}}<｜end▁of▁sentence｜>"]),
            format_system=StringFormatter(slots=["{{content}}\n\n"]),
            format_function=FunctionFormatter(slots=["{{content}}<｜end▁of▁sentence｜>"], tool_format="deepseek_v4"),
            format_observation=StringFormatter(slots=["\n<function_results>\n<result>{{content}}</result>\n</function_results>"]),
            format_tools=ToolFormatter(tool_format="deepseek_v4"),
            format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
            stop_words=["<｜end▁of▁sentence｜>"],
            thought_words=("<think>", "</think>"),
            enable_thinking=True,
            template_class=ReasoningTemplate,
        )
        logger.debug("Registered LF template 'deepseek_v4_thinking'")
