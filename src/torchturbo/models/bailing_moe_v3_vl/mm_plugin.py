"""Bridge LlamaFactory media to the checkpoint's Bailing MM2 processor."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any

from llamafactory.data.mm_plugin import BasePlugin
from llamafactory.extras.constants import IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from PIL import Image

from .processing import get_video_inputs


class BailingVLPlugin(BasePlugin):
    """Match Ling image preprocessing and preserve native video metadata."""

    def _preprocess_image(
        self,
        image: Image.Image,
        image_max_pixels: int,
        image_min_pixels: int,
        **kwargs: Any,
    ) -> Image.Image:
        from torchvision.transforms import Resize

        image = image.convert("RGB")
        width, height = image.size
        if image_max_pixels > 0 and width * height > image_max_pixels:
            # Swift caps the area with bilinear resize before Qwen patch alignment.
            ratio = width / height
            height_scaled = math.sqrt(image_max_pixels / ratio)
            width_scaled = height_scaled * ratio
            image = Resize((int(height_scaled), int(width_scaled)))(image)
        return image

    def _get_mm_inputs(
        self,
        images: list[Any],
        videos: list[Any],
        audios: list[Any],
        processor: Any,
    ) -> dict[str, Any]:
        self._validate_input(processor, images, videos, audios)
        media_inputs: dict[str, Any] = {}
        if images:
            from qwen_vl_utils import fetch_image

            image_processor = processor.image_processor
            image_inputs = self._regularize_images(
                images,
                image_max_pixels=processor.image_max_pixels,
                image_min_pixels=processor.image_min_pixels,
            )["images"]
            image_inputs = [
                fetch_image(
                    {"image": image, "min_pixels": processor.image_min_pixels},
                    image_patch_size=image_processor.patch_size,
                )
                for image in image_inputs
            ]
            media_inputs.update(
                image_processor(images=image_inputs, return_tensors="pt", do_resize=False)
            )
        if videos:
            media_inputs.update(get_video_inputs(videos, processor))
        return media_inputs

    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[Any],
        videos: list[Any],
        audios: list[Any],
        processor: Any,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        texts = [message["content"] for message in messages]
        texts = [
            text.replace(IMAGE_PLACEHOLDER, processor.image_token) for text in texts
        ]
        texts = [
            text.replace(
                VIDEO_PLACEHOLDER,
                processor.video_start_token
                + processor.video_token
                + processor.video_end_token,
            )
            for text in texts
        ]
        if self.expand_mm_tokens:
            media_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if images:
                merge_size = processor.image_processor.merge_size
                token_counts = iter(
                    int(grid.prod().item()) // merge_size**2
                    for grid in media_inputs["image_grid_thw"]
                )
                wrapped_image = (
                    processor.vision_start_token
                    + processor.image_token
                    + processor.vision_end_token
                )
                image_pattern = (
                    re.escape(wrapped_image) + "|" + re.escape(processor.image_token)
                )

                def expand_image(_match: re.Match[str]) -> str:
                    return (
                        processor.vision_start_token
                        + processor.image_token * next(token_counts)
                        + processor.vision_end_token
                    )

                texts = [re.sub(image_pattern, expand_image, text) for text in texts]
            if videos:
                texts = processor._expand_video_tokens(texts, media_inputs)
        for message, text in zip(messages, texts):
            message["content"] = text
        return messages

    def get_mm_inputs(
        self,
        images: list[Any],
        videos: list[Any],
        audios: list[Any],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Any,
    ) -> dict[str, Any]:
        media_inputs = self._get_mm_inputs(images, videos, audios, processor)
        # LF slices grids by the number of original videos when packing samples.
        # Keep one grid per video; the model expands temporal grids for mRoPE.
        media_inputs.pop("video_timestamps_seconds", None)
        return media_inputs
