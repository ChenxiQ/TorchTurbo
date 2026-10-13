"""Mask-creation tests for the qwen3_moe registry migration."""

from __future__ import annotations

import torch


def _sliding_window_factory():
    import transformers.models.qwen3_moe.modeling_qwen3_moe as qwen3_moe_model

    # The registered mask entry maps "turbo_ulysses" to the flash-attention
    # mask function; the factory must not take the unrecognized-name
    # early-exit that drops the mask.
    return qwen3_moe_model.create_sliding_window_causal_mask


def _config():
    from transformers import Qwen3MoeConfig

    config = Qwen3MoeConfig(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=16,
        use_sliding_window=True,
        sliding_window=4,
        max_position_embeddings=16,
    )
    config._attn_implementation = "turbo_ulysses"
    return config


def _call(factory, attention_mask):
    embeds = torch.zeros(1, attention_mask.size(-1), 8)
    cache_position = torch.arange(attention_mask.size(-1))
    return factory(
        config=_config(),
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=None,
    )


def test_sliding_window_factory_keeps_padding_mask():
    """Unpacked padded batches keep a 2D bool mask so flash attention unpads."""
    from torchturbo.models.modules.attention.ulysses import (
        register_turbo_ulysses_attention,
    )

    register_turbo_ulysses_attention(_config(), object(), 2)
    padding = torch.tensor([[1, 1, 1, 0, 0]])

    out = _call(_sliding_window_factory(), padding)

    torch.testing.assert_close(out, padding.bool())


def test_sliding_window_factory_returns_none_without_padding():
    """All-valid masks become None; flash attention runs causal without unpad."""
    from torchturbo.models.modules.attention.ulysses import (
        register_turbo_ulysses_attention,
    )

    register_turbo_ulysses_attention(_config(), object(), 2)
    all_valid = torch.ones(1, 5, dtype=torch.long)

    out = _call(_sliding_window_factory(), all_valid)

    assert out is None
