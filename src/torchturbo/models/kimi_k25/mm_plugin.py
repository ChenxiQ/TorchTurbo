"""LLamaFactory mm_plugin for Kimi K2.5
"""

from copy import deepcopy
from io import BytesIO
from typing import TYPE_CHECKING, BinaryIO, Literal, Optional, TypedDict, Union
from typing_extensions import override

from llamafactory.extras.constants import IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from llamafactory.data.mm_plugin import BasePlugin
from llamafactory.extras.packages import (
    is_librosa_available,
    is_pillow_available,
    is_pyav_available,
    is_transformers_version_greater_than,
)

if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject

if TYPE_CHECKING:
    import torch
    from av.stream import Stream
    from numpy.typing import NDArray
    from transformers import PreTrainedTokenizer, ProcessorMixin

    class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]

    ImageInput = Union[str, bytes, EncodedImage, BinaryIO, ImageObject]
    VideoInput = Union[str, BinaryIO, list[list[ImageInput]]]
    AudioInput = Union[str, BinaryIO, NDArray]


class KimiK25Plugin(BasePlugin):
    image_token: str = "<|media_pad|>"
    video_token: str = "<|media_pad|>"

    def process_messages(self, messages, images, videos, audios, processor):
        self._validate_input(processor, images, videos, audios)

        visual_processor = processor.media_processor

        messages = deepcopy(messages)
        img_idx = 0
        vid_idx = 0

        for message in messages:
            content = message["content"]
            if not isinstance(content, str):
                continue

            # --- 1. deal with navit ---
            if IMAGE_PLACEHOLDER in content:
                parts = content.split(IMAGE_PLACEHOLDER)
                new_content = ""
                for i in range(len(parts) - 1):
                    new_content += parts[i]
                    media_item = {"type": "image", "image": images[img_idx]}
                    seq_len = visual_processor.get_resize_config(media_item)[
                        "num_tokens"
                    ]

                    prompt = (
                        f"<|media_begin|>image<|media_content|>"
                        f"{self.image_token * seq_len}"
                        f"<|media_end|>"
                    )
                    new_content += prompt
                    img_idx += 1
                new_content += parts[-1]
                content = new_content

            # --- 2. deal with video chunks ---
            if VIDEO_PLACEHOLDER in content:
                parts = content.split(VIDEO_PLACEHOLDER)
                new_content = ""
                for i in range(len(parts) - 1):
                    new_content += parts[i]
                    video_source = videos[vid_idx]
                    video_chunks = visual_processor.split_video_chunks(video_source)

                    video_prompt = ""
                    for chunk in video_chunks:
                        chunk_seq_len = visual_processor.get_resize_config(chunk)[
                            "num_tokens"
                        ]

                        # chunk['prompt']: [timestamp]<|media_begin|>video<|media_content|><|media_pad|><|media_end|>
                        timestamp_part = chunk["prompt"].split("<|media_begin|>")[0]
                        video_prompt += (
                            f"{timestamp_part}<|media_begin|>video<|media_content|>"
                            f"{self.video_token * chunk_seq_len}"
                            f"<|media_end|>"
                        )

                    new_content += video_prompt
                    vid_idx += 1
                new_content += parts[-1]
                content = new_content

            message["content"] = content

        return messages

    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["ProcessorMixin"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        visual_processor = processor.media_processor

        all_medias = []

        for img in images:
            if not isinstance(img, Image.Image):
                img = self._regularize_images([img])["images"][0]
            all_medias.append({"type": "image", "image": img})

        for vid in videos:
            video_chunks = visual_processor.split_video_chunks(vid)
            all_medias.extend(video_chunks)

        inputs = visual_processor.preprocess(all_medias, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"],
            "grid_thws": inputs["grid_thws"],
        }
