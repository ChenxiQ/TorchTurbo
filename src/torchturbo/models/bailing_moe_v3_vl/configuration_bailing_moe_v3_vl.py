# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BailingMoE V3 VL model configuration"""

from __future__ import annotations

import os
from typing import Any, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class BailingMoeV3Config(PretrainedConfig):
    model_type = "bailing_moe_v3"

    # Conservative tensor-parallel plan for vLLM's Transformers backend.
    #
    # Keep the hybrid KDA/MLA attention blocks replicated: their non-Linear
    # convolution/recurrent parameters and state are head-shaped and are not
    # automatically sharded when vLLM replaces only nn.Linear modules.
    # MLP projections are safe to shard in the standard colwise -> rowwise
    # pattern, including the unfused per-expert representation used by this
    # remote-code model.
    base_model_tp_plan = {
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.mlp.experts.*.gate_proj": "colwise",
        "layers.*.mlp.experts.*.up_proj": "colwise",
        "layers.*.mlp.experts.*.down_proj": "rowwise",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
    }

    def __init__(
        self,
        vocab_size: int = 157184,
        hidden_size: int = 2048,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 20,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        hidden_act: str = "silu",
        use_qkv_bias: bool = False,  # bailing only
        use_bias: bool = False,  # bailing only
        rms_norm_eps: float = 1e-06,
        tie_word_embeddings: bool = False,  # PretrainedConfig key, here change default value.
        embedding_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        output_dropout: float = 0.0,
        initializer_range: float = 0.02,
        max_position_embeddings: int = 32768,
        rope_theta: float = 600000.0,
        use_cache: bool = True,
        max_window_layers: int = 20,
        rope_scaling: dict[str, Any] | None = None,
        pad_token_id: int = 156892,
        eos_token_id: int = 156892,
        num_experts: int = 256,
        num_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        n_group: int = 8,
        topk_group: int = 4,
        moe_intermediate_size: int = 512,
        moe_shared_expert_intermediate_size: int = 512,
        expert_swiglu_limit_list: list[float] | None = None,
        share_expert_swiglu_limit_list: list[float] | None = None,
        first_k_dense_replace: int = 1,
        head_dim: int = 128,
        output_router_logits: bool = False,
        use_qk_norm: bool = True,
        num_nextn_predict_layers: int = 0,
        mtp_loss_scaling_factor: float = 0.0,
        moe_router_enable_expert_bias: bool = True,
        routed_scaling_factor: float = 1.0,
        layer_group_size: int | list[int] = 5,
        kv_lora_rank: int = 512,
        q_lora_rank: int | None = None,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        qk_nope_head_dim: int = 128,
        rope_interleave: bool = True,
        score_function: str = "sigmoid",
        scoring_func: str = "sigmoid",
        seq_aux: bool = True,
        topk_method: str = "noaux_tc",
        router_dtype: str = "fp32",
        gated_attention_proj_granularity_type: str | None = None,
        no_kda_lora: bool = False,
        kda_safe_gate: bool = False,
        kda_lower_bound: float | None = None,
        short_conv_kernel_size: int = 4,
        image_patch_token: int = 157157,
        video_patch_token: int = 156909,
        image_start_token: int = 157158,
        video_start_token: int = 157160,
        use_interleaved_frame_timestamp: bool = True,
        **kwargs: Any,
    ) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.use_qkv_bias = use_qkv_bias
        self.use_bias = use_bias
        self.rms_norm_eps = rms_norm_eps
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_loss_scaling_factor = mtp_loss_scaling_factor
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.max_window_layers = max_window_layers
        self.head_dim = head_dim or self.hidden_size // self.num_attention_heads
        self.rope_scaling = rope_scaling
        self.use_qk_norm = use_qk_norm
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.routed_scaling_factor = routed_scaling_factor

        # MoE configs
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.expert_swiglu_limit_list = expert_swiglu_limit_list
        self.share_expert_swiglu_limit_list = share_expert_swiglu_limit_list
        self.first_k_dense_replace = first_k_dense_replace
        self.output_router_logits = output_router_logits

        # Linear configs
        self.layer_group_size = layer_group_size
        # mla
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

        self.score_function = score_function
        self.scoring_func = scoring_func
        self.seq_aux = seq_aux
        self.topk_method = topk_method
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.rope_interleave = rope_interleave
        self.router_dtype = router_dtype
        self.gated_attention_proj_granularity_type = (
            gated_attention_proj_granularity_type
        )
        self.no_kda_lora = no_kda_lora
        self.kda_safe_gate = kda_safe_gate
        self.kda_lower_bound = kda_lower_bound if kda_safe_gate else None
        self.short_conv_kernel_size = short_conv_kernel_size
        self.image_patch_token = image_patch_token
        self.video_patch_token = video_patch_token
        self.image_start_token = image_start_token
        self.video_start_token = video_start_token
        # config.json does not carry this flag. The default mirrors the
        # checkpoint processor (preprocessor_config.json), which renders every
        # video frame as its own timestamped vision segment.
        self.use_interleaved_frame_timestamp = use_interleaved_frame_timestamp

        if isinstance(layer_group_size, int):
            if layer_group_size <= 0:
                raise ValueError("layer_group_size must be positive")
            if num_hidden_layers % layer_group_size != 0:
                raise ValueError(
                    f"num_hidden_layers={num_hidden_layers} must be divisible by "
                    f"layer_group_size={layer_group_size}"
                )
        elif isinstance(layer_group_size, list):
            if len(layer_group_size) != num_hidden_layers:
                raise ValueError(
                    "When layer_group_size is a list, its length must equal num_hidden_layers"
                )
        else:
            raise TypeError("layer_group_size must be an int or a list")

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def layers_block_type(self) -> list[str]:
        """Return the SGLang-compatible attention type for every decoder layer."""
        if isinstance(self.layer_group_size, int):
            return [
                "attention"
                if (layer_idx + 1) % self.layer_group_size == 0
                else "linear_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]
        return [
            "linear_attention" if layer_type == 1 else "attention"
            for layer_type in self.layer_group_size
        ]

    @property
    def linear_layer_ids(self) -> list[int]:
        return [
            i
            for i, layer_type in enumerate(self.layers_block_type)
            if layer_type == "linear_attention"
        ]

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return [
            i
            for i, layer_type in enumerate(self.layers_block_type)
            if layer_type == "attention"
        ]


class Qwen3VLMoeVisionConfig(PretrainedConfig):
    model_type = "qwen3_moe_vit"

    def __init__(
        self,
        depth: int = 27,
        hidden_size: int = 1152,
        hidden_act: str = "gelu_pytorch_tanh",
        intermediate_size: int = 4304,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 3584,
        num_position_embeddings: int = 2304,
        deepstack_visual_indexes: list[int] = [8, 16, 24],
        initializer_range: float = 0.02,
        disable_merger_proj: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range
        self.deepstack_visual_indexes = deepstack_visual_indexes
        self.disable_merger_proj = disable_merger_proj

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs: Any
    ) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        if "vision_config" in config_dict:
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class BailingMoeV3VLConfig(PretrainedConfig):
    model_type = "bailing_moe_v3_vl"
    sub_configs = {
        "text_config": BailingMoeV3Config,
        "vision_config": Qwen3VLMoeVisionConfig,
    }

    def __init__(
        self,
        text_config: BailingMoeV3Config | dict[str, Any] | None = None,
        vision_config: Qwen3VLMoeVisionConfig | dict[str, Any] | None = None,
        image_token_id: int = 157157,
        video_token_id: int = 156909,
        vision_start_token_id: int = 157158,
        vision_end_token_id: int = 157159,
        tie_word_embeddings: bool = False,
        mrope_section: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        llm_config = kwargs.pop("llm_config", None)
        if text_config is None:
            text_config = llm_config

        if isinstance(vision_config, dict):
            vision_config = Qwen3VLMoeVisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = Qwen3VLMoeVisionConfig()

        if isinstance(text_config, dict):
            text_config = BailingMoeV3Config(**text_config)
        elif text_config is None:
            text_config = BailingMoeV3Config()

        self.vision_config = vision_config
        self.text_config = text_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        # M-RoPE section: split qk_rope_head_dim // 2 frequencies into [T, H, W].
        # Prefer an explicit top-level value; otherwise follow SGLang and read it
        # from the language rope config before falling back to the historical default.
        rope_parameters = text_config.rope_parameters
        if rope_parameters is None:
            rope_parameters = dict(text_config.rope_scaling or {})
        else:
            rope_parameters = dict(rope_parameters)

        if mrope_section is None:
            mrope_section = rope_parameters.get("mrope_section", [8, 12, 12])
        mrope_section = list(mrope_section)
        expected_mrope_dims = text_config.qk_rope_head_dim // 2
        if len(mrope_section) != 3 or sum(mrope_section) != expected_mrope_dims:
            raise ValueError(
                f"mrope_section must contain three entries summing to {expected_mrope_dims}, "
                f"got {mrope_section}"
            )
        self.mrope_section = mrope_section
        # The HF text rotary module reads mrope_section from the text config,
        # while SGLang reads it from llm_config.rope_parameters.  Set both.
        text_config.mrope_section = mrope_section

        # Keep both naming conventions in sync.  SGLang uses rope_parameters; the
        # HF modeling in this directory still understands rope_scaling/default RoPE.
        rope_parameters.update(
            rope_type=rope_parameters.get(
                "rope_type", rope_parameters.get("type", "default")
            ),
            mrope_section=mrope_section,
            video_rope=rope_parameters.get("video_rope", True),
        )
        text_config.rope_parameters = dict(rope_parameters)
        text_config.rope_scaling = dict(rope_parameters)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        # SGLang names the language config `llm_config`; keep it as an alias of
        # the Hugging Face `text_config` across serialization round trips.
        self.llm_config = self.text_config

    @classmethod
    def from_dict(
        cls, config_dict: dict[str, Any], **kwargs: Any
    ) -> BailingMoeV3VLConfig | tuple[BailingMoeV3VLConfig, dict[str, Any]]:
        # Transformers applies overrides after __init__, so normalize the alias
        # before the parent method updates the nested configuration.
        llm_config = kwargs.pop("llm_config", None)
        if llm_config is not None and "text_config" not in kwargs:
            kwargs["text_config"] = llm_config

        result = super().from_dict(config_dict, **kwargs)
        if isinstance(result, tuple):
            config, unused_kwargs = result
            config.llm_config = config.text_config
            return config, unused_kwargs

        result.llm_config = result.text_config
        return result

    @property
    def num_nextn_predict_layers(self) -> int:
        return self.text_config.num_nextn_predict_layers

    @num_nextn_predict_layers.setter
    def num_nextn_predict_layers(self, value: int) -> None:
        self.text_config.num_nextn_predict_layers = value
