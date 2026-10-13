"""Bailing V3 VL training prompts and media preprocessing for ms-swift."""

import json
from itertools import groupby
from typing import Any, Literal

from PIL import Image
from swift.template import Template
from swift.template.template_inputs import StdTemplateInputs
from swift.template.template_meta import TemplateMeta
from swift.utils import to_device

from ..bailing_hybrid.swift_template import (
    THINK_END,
    LingV3AgentTemplate,
    _make_ling_v3_template_meta,
)
from .processing import (
    get_last_query_index,
    get_video_inputs,
    normalize_thinking_content,
    read_auto_thinking_env,
    remove_thinking_content,
    resolve_sft_thinking,
)

_TEMPLATE_NAME = "bailing_moe_v3_vl"


class BailingVLAgentTemplate(LingV3AgentTemplate):
    """Format tool calls exactly as the checkpoint chat template renders them."""

    def _format_tools(
        self,
        tools: list[dict[str, Any]],
        system: str | None = None,
        user_message: dict[str, Any] | None = None,
    ) -> str:
        descriptions = super()._format_tools(tools, user_message=user_message)
        system = system or ""
        if system and not system.endswith("\n"):
            system += "\n"
        return system + descriptions

    def _format_tool_calls(self, tool_call_messages: list[dict[str, Any]]) -> str:
        tool_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message["content"])
            name, arguments = tool_call["name"], tool_call["arguments"]
            if (
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(arguments, dict)
            ):
                raise ValueError(
                    "Ling tool calls require a non-empty name and an arguments object."
                )
            lines = [f"<tool_call>{name}"]
            for key, value in arguments.items():
                if not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)
                lines.extend([f"<arg_key>{key}</arg_key>", f"<arg_value>{value}</arg_value>"])
            lines.append("</tool_call>")
            tool_calls.append("\n".join(lines))
        return "\n".join(tool_calls)

    def _add_tool_call_prefix(
        self, tool_content: str, pre_message: dict[str, Any] | None = None
    ) -> str:
        # A newline separates the first call from visible assistant content.
        if pre_message is None or pre_message["role"] != "assistant":
            return tool_content
        content = pre_message.get("content") or ""
        if not isinstance(content, str):
            return tool_content
        answer = content.rsplit(THINK_END, 1)[-1].lstrip("\n")
        return "\n" + tool_content if answer else tool_content

    def _format_tool_responses(
        self, assistant_content: Any, tool_messages: list[dict[str, Any]]
    ) -> tuple[Any, list[str]]:
        prompt = self._format_standalone_tool_responses(tool_messages)
        prompt.append("<|role_end|><role>ASSISTANT</role>\n")
        return assistant_content, prompt

    def _format_standalone_tool_responses(
        self, tool_messages: list[dict[str, Any]]
    ) -> list[str]:
        if not tool_messages:
            raise ValueError("Ling observations require at least one tool response.")
        prompt = ["<|role_end|>", "<role>OBSERVATION</role>"]
        for message in tool_messages:
            prompt.extend(["\n<tool_response>\n", message["content"], "\n</tool_response>"])
        return prompt


class BailingVLTemplate(Template):
    """Encode Ling SFT prompts and leave real-media fusion to the model."""

    support_padding_free: bool = True
    auto_sft_thinking: bool

    def init_env_args(self) -> None:
        super().init_env_args()
        # Include prompt and image preprocessing semantics in Swift's cache key.
        self._version = "bailing-vl-auto-thinking-qwen-images-v1"
        self.auto_sft_thinking = read_auto_thinking_env()

    def _get_max_pixels(self, inputs: StdTemplateInputs | None = None) -> int | None:
        # Per-sample pixel limits are applied by fetch_image after Swift's resize.
        return self.max_pixels

    def _remove_thinking_content(
        self, content: str, thinking_suffix: str = THINK_END
    ) -> str:
        return remove_thinking_content(content, thinking_suffix)

    def _remove_history_thinking(self, inputs: StdTemplateInputs) -> None:
        messages = inputs.messages
        last_query = get_last_query_index(messages)
        for message in messages[:last_query]:
            if message["role"] != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, list) and content and isinstance(content[0], str):
                content[0] = self._remove_thinking_content(content[0])
            elif isinstance(content, str):
                message["content"] = self._remove_thinking_content(content)

    def _get_system(self, inputs: StdTemplateInputs) -> str:
        system = super()._get_system(inputs)
        raw_system = inputs.system
        if raw_system is None:
            raw_system = self.template_meta.default_system
        raw_system = raw_system or ""
        if "detailed thinking on" in raw_system or "detailed thinking off" in raw_system:
            return system
        thinking = "on" if self._get_enable_thinking(inputs) else "off"
        directive = f"detailed thinking {thinking}"
        if not system:
            return directive
        if inputs.tools:
            return system + directive
        return system.rstrip("\n") + "\n" + directive

    def _preprocess_tool_call(self, inputs: StdTemplateInputs) -> None:
        super()._preprocess_tool_call(inputs)
        if self.template_backend != "swift":
            return
        # Fold observations without an assistant call into the preceding query
        # before Swift pairs queries and answers.
        messages = []
        for role, group in groupby(inputs.messages, key=lambda message: message["role"]):
            group = list(group)
            if role != "tool" or (messages and messages[-1]["role"] == "assistant"):
                messages.extend(group)
                continue
            if not messages or messages[-1]["role"] != "user":
                raise ValueError("A standalone tool message must follow a user message.")
            query = messages[-1]
            content = query.get("content") or ""
            if not isinstance(content, str):
                raise ValueError("Standalone tool messages require text-only user content.")
            self.agent_template.template_meta = self.template_meta
            observations = self.agent_template._format_standalone_tool_responses(group)
            if not all(isinstance(part, str) for part in observations):
                raise ValueError("Standalone tool formatting must produce text contexts.")
            query["content"] = content + "".join(observations)
        inputs.messages[:] = messages

    def _swift_prepare_inputs(self, inputs: StdTemplateInputs) -> None:
        if (
            self.auto_sft_thinking
            and self.mode == "train"
            and self.task_type == "causal_lm"
            and self.use_chat_template
        ):
            system = inputs.system
            if system is None:
                system = self.template_meta.default_system
            inputs.chat_template_kwargs["enable_thinking"] = resolve_sft_thinking(
                inputs.messages, system, inputs.chat_template_kwargs.get("enable_thinking")
            )
        if self.use_chat_template and self.template_backend == "swift":
            # Normalize before Swift merges assistant content with tool calls.
            for message in inputs.messages:
                if message["role"] != "assistant":
                    continue
                message["content"] = normalize_thinking_content(message.get("content"))
        super()._swift_prepare_inputs(inputs)

    def replace_tag(
        self,
        media_type: Literal["image", "video", "audio"],
        index: int,
        inputs: StdTemplateInputs,
    ) -> list[str]:
        processor = self.processor
        if media_type == "image":
            from qwen_vl_utils import fetch_image

            inputs.images[index] = fetch_image(
                {"image": inputs.images[index], **inputs.chat_template_kwargs},
                image_patch_size=processor.image_processor.patch_size,
            )
            media_inputs = processor.image_processor(
                images=[inputs.images[index]], return_tensors="pt", do_resize=False
            )
            merge_size = processor.image_processor.merge_size
            num_tokens = int(media_inputs["image_grid_thw"].prod().item()) // merge_size**2
            text = (
                processor.vision_start_token
                + processor.image_token * num_tokens
                + processor.vision_end_token
            )
        elif media_type == "video":
            media_inputs = get_video_inputs([inputs.videos[index]], processor)
            text = (
                processor.video_start_token
                + processor.video_token
                + processor.video_end_token
            )
            text = processor._expand_video_tokens([text], media_inputs)[0]
        else:
            raise ValueError("Bailing V3 VL supports image and video inputs only.")
        inputs.mm_processor_kwargs["bailing_media_inputs"].append(media_inputs)
        return [text]

    def _encode(self, inputs: StdTemplateInputs) -> dict[str, Any]:
        inputs.mm_processor_kwargs["bailing_media_inputs"] = []
        encoded = super()._encode(inputs)
        media_inputs = inputs.mm_processor_kwargs.pop("bailing_media_inputs")
        encoded.update(self._data_collator_mm_data(media_inputs))
        return encoded

    def _post_encode(self, model, inputs: dict[str, Any]) -> dict[str, Any]:
        """Add zero-valued visual and projector dependencies to text-only batches."""
        if (
            not self.is_training
            or inputs.get("pixel_values") is not None
            or inputs.get("pixel_values_videos") is not None
        ):
            return inputs

        input_ids = inputs["input_ids"]
        base_model = self.get_base_model(model)
        inputs_embeds = base_model.get_input_embeddings()(input_ids)
        dummy_image = Image.new("RGB", (32, 32), (0, 0, 0))
        media_inputs = self.processor.image_processor(
            images=[dummy_image], return_tensors="pt"
        )
        media_inputs = to_device(media_inputs, input_ids.device)
        dummy_embeds = base_model.get_image_features(
            pixel_values=media_inputs["pixel_values"],
            image_grid_thw=media_inputs["image_grid_thw"],
        )
        dummy_dependency = dummy_embeds[0].mean().to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        inputs["inputs_embeds"] = inputs_embeds + dummy_dependency * 0
        inputs["mtp_input_ids"] = input_ids
        return inputs


def _register_agent_template() -> None:
    from swift.agent_template import agent_template_map

    agent_template_map[_TEMPLATE_NAME] = BailingVLAgentTemplate


def _make_template_meta() -> TemplateMeta:
    """Build the ling v3 template meta with the checkpoint tool-call format."""
    meta = _make_ling_v3_template_meta(_TEMPLATE_NAME, BailingVLTemplate)
    meta.agent_template = _TEMPLATE_NAME
    return meta


def register() -> None:
    """Attach the Bailing VL template to the existing Turbo Swift model entry."""
    from swift.model.model_meta import MODEL_MAPPING
    from swift.template.register import register_template

    _register_agent_template()
    register_template(_make_template_meta(), exist_ok=True)
    model_meta = MODEL_MAPPING[f"turbo_{_TEMPLATE_NAME}"]
    model_meta.template = _TEMPLATE_NAME
    model_meta.candidate_templates = [_TEMPLATE_NAME]
