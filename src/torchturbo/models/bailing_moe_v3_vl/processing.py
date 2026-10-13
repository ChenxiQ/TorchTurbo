"""Bailing VL preprocessing shared by training framework adapters."""

import os
from typing import Any

THINK_START = "<think>"
THINK_END = "</think>"


def read_auto_thinking_env() -> bool:
    """Resolve the mode once so worker copies and cache keys retain it."""
    value = os.environ.get("LING_SFT_AUTO_THINKING", "0").strip().lower()
    if value not in {"0", "1", "false", "true", "off", "on", "no", "yes"}:
        raise ValueError("LING_SFT_AUTO_THINKING must be a boolean (0/1 or false/true).")
    return value in {"1", "true", "on", "yes"}


def normalize_thinking_content(content: Any) -> Any:
    """Match the checkpoint's inline thinking and boundary-newline handling."""
    if not isinstance(content, str) or THINK_END not in content:
        return content
    parts = content.split(THINK_END)
    reasoning = parts[0].rsplit(THINK_START, 1)[-1].strip("\n")
    answer = parts[-1].lstrip("\n")
    return f"{THINK_START}{reasoning}{THINK_END}{answer}"


def remove_thinking_content(content: str, thinking_suffix: str = THINK_END) -> str:
    """Remove the leading normalized block without changing visible whitespace."""
    if content.startswith(THINK_START) and thinking_suffix in content:
        content = content.partition(thinking_suffix)[2]
    return THINK_START + THINK_END + content


def get_last_query_index(messages: list[dict[str, Any]]) -> int:
    """Find the last user question, excluding wrapped tool observations."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message["role"] != "user":
            continue
        content = message.get("content")
        if (
            isinstance(content, str)
            and content.startswith("<tool_response>")
            and content.endswith("</tool_response>")
        ):
            continue
        return index
    return len(messages) - 1


def _assistant_has_thinking(content: Any, index: int) -> bool:
    if content is None:
        return False
    if not isinstance(content, str):
        raise ValueError(
            f"Cannot infer thinking mode from non-text assistant message {index}; "
            "set chat_template_kwargs.enable_thinking explicitly."
        )
    content = content.lstrip()
    if content.startswith(THINK_END):
        raise ValueError(f"Unmatched </think> in assistant message {index}.")
    if not content.startswith(THINK_START):
        return False
    reasoning, closing, answer = content[len(THINK_START) :].partition(THINK_END)
    has_extra_tags = THINK_START in reasoning or THINK_START in answer or THINK_END in answer
    if not closing or has_extra_tags:
        raise ValueError(
            f"Incomplete or ambiguous thinking block in assistant message {index}."
        )
    return bool(reasoning.strip())


def resolve_sft_thinking(
    messages: list[dict[str, Any]],
    system: str | None = None,
    enable_thinking: bool | None = None,
) -> bool:
    """Infer a sample's mode before normalization or tool-call merging."""
    explicit = enable_thinking
    if explicit is not None and not isinstance(explicit, bool):
        raise ValueError("chat_template_kwargs.enable_thinking must be a JSON boolean.")
    if system is not None and not isinstance(system, str):
        raise ValueError("Automatic thinking mode requires a text system message.")
    system = system or ""
    on = "detailed thinking on" in system
    off = "detailed thinking off" in system
    if on and off:
        raise ValueError("Conflicting system instructions: thinking is both on and off.")
    system_mode = on if on or off else None
    if explicit is not None:
        if system_mode is not None and explicit != system_mode:
            raise ValueError("Conflicting thinking modes in chat_template_kwargs and system.")
        return explicit
    if system_mode is not None:
        return system_mode

    has_target = False
    thinking = False
    for index, message in enumerate(messages):
        if message["role"] == "tool_call":
            has_target = True
        if message["role"] != "assistant":
            continue
        has_target = True
        if _assistant_has_thinking(message.get("content"), index):
            thinking = True
    if not has_target:
        raise ValueError("Cannot infer SFT thinking mode without an assistant answer or tool call.")
    return thinking


def get_video_inputs(videos: list[Any], processor: Any) -> dict[str, Any]:
    """Decode videos with the native frame sampling and timestamp metadata."""
    conversation = [
        {
            "role": "user",
            "content": [{"type": "video", "video": video} for video in videos],
        }
    ]
    _, video_inputs, _ = processor.process_vision_info(conversation)
    frames = [video[0] for video in video_inputs]
    timestamps = [video[1]["resmp_ts"] for video in video_inputs]
    return processor.image_processor(
        images=None,
        videos=frames,
        do_resize=False,
        videos_timestamps_seconds=timestamps,
        return_tensors="pt",
    )
