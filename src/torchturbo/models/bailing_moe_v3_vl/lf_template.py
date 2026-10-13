"""LlamaFactory registration using the checkpoint's Bailing VL chat template."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from llamafactory.data.template import ReasoningTemplate, Template

from .processing import (
    get_last_query_index,
    normalize_thinking_content,
    read_auto_thinking_env,
    remove_thinking_content,
    resolve_sft_thinking,
)

_ASSISTANT_PREFIX = "<role>ASSISTANT</role>\n"
_EMPTY_THINKING = re.compile(r"^<think>\s*</think>\s*")


@dataclass
class BailingVLTemplate(ReasoningTemplate):
    """Render the checkpoint Jinja template and expose assistant-only targets.

    Subclassing ReasoningTemplate lets LlamaFactory apply ``--enable_thinking``
    and ``--preserve_thinking`` to this template. The checkpoint template
    renders the thinking delimiters itself, so encoding keeps the plain
    Template flow instead of the reasoning template's CoT rewriting.
    """

    auto_sft_thinking: bool = field(default_factory=read_auto_thinking_env)

    def encode_oneturn(
        self,
        tokenizer: Any,
        messages: list[dict[str, str]],
        system: str | None = None,
        tools: str | None = None,
    ) -> tuple[list[int], list[int]]:
        return Template.encode_oneturn(self, tokenizer, messages, system, tools)

    def encode_multiturn(
        self,
        tokenizer: Any,
        messages: list[dict[str, str]],
        system: str | None = None,
        tools: str | None = None,
        discarding_history_cot: bool = False,
    ) -> list[tuple[list[int], list[int]]]:
        # History reasoning is removed by _encode according to
        # preserve_thinking, which LlamaFactory also uses to derive
        # discarding_history_cot.
        encoded = self._encode(tokenizer, messages, system, tools, is_training=True)
        return [(encoded[i], encoded[i + 1]) for i in range(0, len(encoded), 2)]

    def _encode(
        self,
        tokenizer: Any,
        messages: list[dict[str, str]],
        system: str | None,
        tools: str | None,
        *,
        is_training: bool = False,
    ) -> list[list[int]]:
        tool_definitions = json.loads(tools) if tools else None
        normalized = deepcopy(messages)
        last_user = max(
            (
                index
                for index, message in enumerate(normalized)
                if message["role"] == "user"
            ),
            default=-1,
        )
        if is_training:
            last_user = get_last_query_index(normalized)
        for message in normalized:
            if message["role"] == "observation":
                message["role"] = "tool"
            elif message["role"] == "function":
                call_content = message["content"]
                thought_content = ""
                if call_content.lstrip().startswith(self.thought_words[0]):
                    thought_content, thought_end, call_content = call_content.partition(
                        self.thought_words[1]
                    )
                    thought_content += thought_end
                calls = json.loads(call_content)
                if isinstance(calls, dict):
                    calls = [calls]
                for call in calls:
                    if isinstance(call["arguments"], str):
                        call["arguments"] = json.loads(call["arguments"])
                message.update(
                    role="assistant", content=thought_content, tool_calls=calls
                )

        enable_thinking = bool(self.enable_thinking)
        if is_training and self.auto_sft_thinking:
            enable_thinking = resolve_sft_thinking(normalized, system)

        if (
            is_training
            and system
            and not tool_definitions
            and "detailed thinking on" not in system
            and "detailed thinking off" not in system
        ):
            thinking = "on" if enable_thinking else "off"
            system += f"\ndetailed thinking {thinking}"
        conversation: list[dict[str, Any]] = []
        if system:
            conversation.append({"role": "system", "content": system})

        for index, message in enumerate(normalized):
            if message["role"] != "assistant":
                continue
            if is_training:
                message["content"] = normalize_thinking_content(message["content"])
            if not self.preserve_thinking and index < last_user:
                if is_training:
                    message["content"] = remove_thinking_content(message["content"])
                else:
                    message["content"] = self.remove_thought(message["content"])
                message.pop("reasoning_content", None)

        encoded: list[list[int]] = []
        previous_text = ""
        for index, message in enumerate(normalized):
            conversation.append(message)
            rendered = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                preserve_thinking=True,
                tools=tool_definitions,
            )
            if index % 2 == 0:
                rendered += _ASSISTANT_PREFIX
            if not rendered.startswith(previous_text):
                raise ValueError(
                    "Bailing VL chat history must remain a prefix when encoding a response."
                )
            fragment = rendered[len(previous_text) :]
            if is_training and index % 2 == 1:
                # Empty thinking belongs to the masked prompt, as in Swift's
                # ignore_empty_think loss scale. Keep it in the input sequence.
                empty_thinking = _EMPTY_THINKING.match(fragment)
                if empty_thinking:
                    prefix = empty_thinking.group()
                    fragment = fragment[len(prefix) :]
                    if fragment == "<|role_end|>" and index < len(normalized) - 1:
                        # Swift's separator inherits an empty response's zero weight.
                        prefix += fragment
                        fragment = ""
                    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
                    encoded[-1].extend(prefix_ids)
            encoded.append(tokenizer.encode(fragment, add_special_tokens=False))
            previous_text = rendered
        return encoded


def register() -> None:
    """Register the Bailing VL template and its image/video plugin."""
    from llamafactory.data.formatter import StringFormatter
    from llamafactory.data.mm_plugin import PLUGINS, get_mm_plugin, register_mm_plugin
    from llamafactory.data.template import TEMPLATES, register_template

    from .mm_plugin import BailingVLPlugin

    name = "bailing_moe_v3_vl"
    if name not in PLUGINS:
        register_mm_plugin(name, BailingVLPlugin)
    if name not in TEMPLATES:
        register_template(
            name=name,
            format_user=StringFormatter(
                slots=["<role>HUMAN</role>{{content}}<|role_end|>" + _ASSISTANT_PREFIX]
            ),
            format_assistant=StringFormatter(slots=["{{content}}<|role_end|>"]),
            stop_words=["<|role_end|>"],
            thought_words=("<think>", "</think>"),
            enable_thinking=False,
            preserve_thinking=True,
            replace_jinja_template=False,
            template_class=BailingVLTemplate,
            mm_plugin=get_mm_plugin(
                name, image_token="<|image_pad|>", video_token="<|video_pad|>"
            ),
        )
