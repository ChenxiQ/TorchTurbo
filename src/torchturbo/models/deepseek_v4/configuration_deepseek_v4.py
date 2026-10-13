# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V4 configuration for transformers library."""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from torchturbo.logging import get_logger

logger = get_logger(__name__)

DEEPSEEK_V4_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class DeepseekV4Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekV4Model`].
    It is used to instantiate a DeepSeek-V4 model according to the specified arguments,
    defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control
    model outputs. Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 129280):
            Vocabulary size of the DeepSeek-V4 model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimension of the MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 4096):
            Dimension of the MoE intermediate representations.
        num_hidden_layers (`int`, *optional*, defaults to 42):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 1):
            Number of key/value heads for GQA (1 for MQA).
        n_routed_experts (`int`, *optional*, defaults to 8):
            Number of routed experts in MoE layers.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts in MoE layers.
        num_experts_per_tok (`int`, *optional*, defaults to 2):
            Number of experts to route each token to.
        scoring_func (`str`, *optional*, defaults to "sqrtsoftplus"):
            Score function for routing ("softmax", "sigmoid", "sqrtsoftplus").
        route_scale (`float`, *optional*, defaults to 1.0):
            Scaling factor for routing weights.
        aux_loss_alpha (`float`, *optional*, defaults to 0.001):
            Auxiliary loss coefficient for load balancing.
        q_lora_rank (`int`, *optional*, defaults to 1024):
            Rank for Q projection LoRA.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank for KV projection (not used in current MLA).
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Dimension of rotary position embeddings.
        qk_nope_head_dim (`int`, *optional*, defaults to 448):
            Dimension of non-rotary query/key heads.
        v_head_dim (`int`, *optional*, defaults to 512):
            Dimension of value heads.
        o_groups (`int`, *optional*, defaults to 8):
            Number of groups for output projection.
        o_lora_rank (`int`, *optional*, defaults to 1024):
            Rank for output projection LoRA.
        window_size (`int`, *optional*, defaults to 128):
            Sliding window size for local attention.
        compress_ratios (`Tuple[int]`, *optional*):
            Compression ratio for each layer's KV cache.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base period of the RoPE embeddings.
        compress_rope_theta (`float`, *optional*, defaults to 40000.0):
            RoPE theta for compressed KV.
        rope_factor (`float`, *optional*, defaults to 40.0):
            YaRN scaling factor.
        original_seq_len (`int`, *optional*, defaults to 65536):
            Original sequence length for YaRN.
        beta_fast (`int`, *optional*, defaults to 32):
            Fast beta for YaRN correction.
        beta_slow (`int`, *optional*, defaults to 1):
            Slow beta for YaRN correction.
        index_n_heads (`int`, *optional*, defaults to 64):
            Number of heads for indexer.
        index_head_dim (`int`, *optional*, defaults to 128):
            Head dimension for indexer.
        index_topk (`int`, *optional*, defaults to 512):
            Top-k value for indexer.
        dsa_indexer_loss_coeff (`float`, *optional*, defaults to 0.0):
            Coefficient for the indexer auxiliary KL loss.
        hc_mult (`int`, *optional*, defaults to 4):
            Hyper-connection multiplier.
        hc_sinkhorn_iters (`int`, *optional*, defaults to 20):
            Sinkhorn iterations for HC.
        hc_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for HC computation.
        swiglu_limit (`float`, *optional*, defaults to 0.0):
            Clamping limit for SwiGLU.
        num_hash_layers (`int`, *optional*, defaults to 0):
            Number of layers using hash-based routing.
        dspark_block_size (`int`, *optional*, defaults to 0):
            Number of positions produced by one DSpark draft block. Zero disables
            DSpark and selects the standard MTP implementation.
        dspark_noise_token_id (`int`, *optional*, defaults to 0):
            Token ID used for non-anchor positions in a DSpark draft block.
        dspark_num_layers (`int`, *optional*, defaults to 0):
            Number of DSpark layers. Zero means the value is inferred
            from checkpoint tensor names.
        dspark_target_layer_ids (`list[int]`, *optional*):
            Backbone layers whose mean hidden states form the DSpark context.
        dspark_markov_rank (`int`, *optional*, defaults to 256):
            Rank of the low-rank DSpark Markov transition head.
        num_mtp_layers (`int`, *optional*, defaults to 0):
            Number of Multi-Token Prediction layers.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for RMS normalization.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use KV cache.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input/output embeddings.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention.
        hidden_act (`str`, *optional*, defaults to "silu"):
            Activation function for MLP.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            Maximum sequence length.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of sequence token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of sequence token id.
    """

    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=129280,
        hidden_size=4096,
        intermediate_size=4096,
        moe_intermediate_size=4096,
        num_hidden_layers=42,
        num_attention_heads=64,
        num_key_value_heads=1,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        scoring_func="sqrtsoftplus",
        route_scale=1.0,
        aux_loss_alpha=0.001,
        q_lora_rank=1024,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=448,
        v_head_dim=512,
        head_dim=512,
        o_groups=8,
        o_lora_rank=1024,
        sliding_window=128,
        compress_ratios=None,
        rope_theta=10000.0,
        compress_rope_theta=40000.0,
        rope_factor=40.0,
        original_max_position_embeddings=65536,
        beta_fast=32,
        beta_slow=1,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        swiglu_limit=0.0,
        num_hash_layers=0,
        num_nextn_predict_layers=0,
        dspark_block_size=0,
        dspark_noise_token_id=0,
        dspark_num_layers=0,
        dspark_target_layer_ids=(),
        dspark_markov_rank=256,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        hidden_act="silu",
        initializer_range=0.02,
        max_position_embeddings=4096,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        dsa_indexer_loss_coeff: float = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.route_scale = route_scale
        self.aux_loss_alpha = aux_loss_alpha
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.head_dim = head_dim
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.sliding_window = sliding_window
        self.compress_ratios = compress_ratios or [1] * num_hidden_layers
        self.rope_theta = rope_theta
        self.compress_rope_theta = compress_rope_theta
        self.rope_factor = rope_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.dsa_indexer_loss_coeff = dsa_indexer_loss_coeff
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.swiglu_limit = swiglu_limit
        self.num_hash_layers = num_hash_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.dspark_block_size = dspark_block_size
        self.dspark_noise_token_id = dspark_noise_token_id
        self.dspark_num_layers = dspark_num_layers
        self.dspark_target_layer_ids = dspark_target_layer_ids
        self.dspark_markov_rank = dspark_markov_rank
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
