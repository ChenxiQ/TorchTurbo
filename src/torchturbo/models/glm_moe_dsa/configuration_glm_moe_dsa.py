# coding=utf-8

from transformers.configuration_utils import PretrainedConfig


class GlmMoeDsaConfig(PretrainedConfig):
    model_type = "glm_moe_dsa"
    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    def __init__(
        self,
        vocab_size=154880,
        hidden_size=6144,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        num_hidden_layers=78,
        num_attention_heads=64,
        num_key_value_heads=64,
        n_shared_experts=1,
        n_routed_experts=256,
        routed_scaling_factor=2.5,
        kv_lora_rank=512,
        q_lora_rank=2048,
        qk_rope_head_dim=64,
        v_head_dim=256,
        qk_nope_head_dim=192,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=8,
        norm_topk_prob=True,
        hidden_act="silu",
        max_position_embeddings=202752,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        rope_parameters=None,
        mlp_layer_types=None,
        attention_bias=False,
        attention_dropout=0.0,
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=32,
        indexer_types=None,
        index_topk_pattern=None,
        index_topk_freq=1,
        index_skip_topk_offset=2,
        index_share_for_mtp_iteration=False,
        **kwargs,
    ):
        # ===== 基础字段 =====
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        # ===== MoE =====
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        # 兼容 modeling 中使用的字段名
        self.num_local_experts = n_routed_experts

        self.routed_scaling_factor = routed_scaling_factor
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.n_group = n_group
        self.topk_group = topk_group

        # ===== MLA / attention =====
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim

        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.qk_rope_head_dim

        # ===== MLP / 激活 =====
        self.hidden_act = hidden_act

        # ===== 位置编码 =====
        self.max_position_embeddings = max_position_embeddings
        if rope_parameters is None:
            rope_parameters = {
                "rope_type": "default",
                "rope_theta": 1000000.0,
            }
        self.rope_parameters = rope_parameters

        # ===== 归一化/初始化 =====
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps

        # ===== attention 其他 =====
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # ===== cache =====
        self.use_cache = use_cache

        # ===== DSA Indexer =====
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads

        # ===== DSA Indexer 类型（index share）=====
        # `"full"` 该层自己跑 indexer 产出 top-k；`"shared"` 复用上一个 full 层的 top-k。
        # 与 GLM 5.2 的 index share 技术对应。
        if indexer_types is None:
            if index_topk_pattern is not None:
                indexer_types = [{"F": "full", "S": "shared"}[c] for c in index_topk_pattern]
            else:
                freq = max(index_topk_freq, 1)
                offset = index_skip_topk_offset
                indexer_types = [
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
                    for i in range(num_hidden_layers)
                ]

        if len(indexer_types) != num_hidden_layers:
            raise ValueError(
                "`indexer_types` length must equal `num_hidden_layers`, "
                f"but got {len(indexer_types)} and {num_hidden_layers}"
            )

        for item in indexer_types:
            if item not in ("full", "shared"):
                raise ValueError(
                    "`indexer_types` items must be either 'full' or 'shared', "
                    f"but got {item}"
                )
        self.indexer_types = indexer_types
        self.index_share_for_mtp_iteration = index_share_for_mtp_iteration

        # ===== MLP layer types =====
        # 官方逻辑：first 3 dense, rest sparse
        if mlp_layer_types is None:
            mlp_layer_types = ["dense"] * min(3, num_hidden_layers) + ["sparse"] * (
                num_hidden_layers - 3
            )

        if len(mlp_layer_types) != num_hidden_layers:
            raise ValueError(
                "`mlp_layer_types` length must equal `num_hidden_layers`, "
                f"but got {len(mlp_layer_types)} and {num_hidden_layers}"
            )

        for item in mlp_layer_types:
            if item not in ["dense", "sparse"]:
                raise ValueError(
                    "`mlp_layer_types` items must be either 'dense' or 'sparse', "
                    f"but got {item}"
                )
        self.mlp_layer_types = mlp_layer_types

        # ===== 合法性检查 =====
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                "`num_attention_heads` must be divisible by `num_key_value_heads`, "
                f"got {num_attention_heads} and {num_key_value_heads}"
            )

        if n_routed_experts <= 0:
            raise ValueError("`n_routed_experts` must be > 0")

        if num_experts_per_tok <= 0:
            raise ValueError("`num_experts_per_tok` must be > 0")

        if num_experts_per_tok > n_routed_experts:
            raise ValueError(
                "`num_experts_per_tok` must be <= `n_routed_experts`, "
                f"got {num_experts_per_tok} and {n_routed_experts}"
            )

        if n_group <= 0:
            raise ValueError("`n_group` must be > 0")

        if n_routed_experts % n_group != 0:
            raise ValueError(
                "`n_routed_experts` must be divisible by `n_group`, "
                f"got {n_routed_experts} and {n_group}"
            )

        if topk_group <= 0 or topk_group > n_group:
            raise ValueError(
                "`topk_group` must satisfy 0 < topk_group <= n_group, "
                f"got topk_group={topk_group}, n_group={n_group}"
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
