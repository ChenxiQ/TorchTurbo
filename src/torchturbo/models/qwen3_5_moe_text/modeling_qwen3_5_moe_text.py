from __future__ import annotations

import torch
from transformers.cache_utils import Cache
from transformers.utils import TransformersKwargs, can_return_tuple
from typing import Unpack

from ..qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeCausalLMOutputWithPast,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeMultiTokenPredictor,
    Qwen3_5MoeTextModel,
    load_balancing_loss_func,
)
from .configuration_qwen3_5_moe_text import Qwen3_5MoeTextConfig

__all__ = [
    "Qwen3_5MoeTextForCausalLM",
    "Qwen3_5MoeTextModel",
    "Qwen3_5MoeGatedDeltaNet",
    "Qwen3_5MoeMultiTokenPredictor",
    "Qwen3_5MoeCausalLMOutputWithPast",
]


class Qwen3_5MoeTextForCausalLM(Qwen3_5MoeForCausalLM):
    config: Qwen3_5MoeTextConfig

    def __init__(self, config: Qwen3_5MoeTextConfig) -> None:
        super().__init__(config)
        mtp_num = config.mtp_num_hidden_layers
        self.mtp = Qwen3_5MoeMultiTokenPredictor(config) if mtp_num > 0 else None

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Qwen3_5MoeCausalLMOutputWithPast:
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )

        seq_input = input_ids if input_ids is not None else inputs_embeds
        if position_ids is None:
            idx = torch.arange(seq_input.shape[1], device=seq_input.device)
            position_ids = idx.view(1, 1, -1).expand(4, seq_input.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, *position_ids.shape)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        mtp_logits = None
        mtp_hidden_states = None
        if self.mtp is not None and self.training and input_ids is not None:
            mtp_hidden_states = self.mtp(
                input_ids=input_ids,
                hidden_states=hidden_states,
                embed_tokens=self.model.embed_tokens,
                position_embeddings=self.model.rotary_emb(hidden_states, position_ids[1:]),
                attention_mask=attention_mask,
                position_ids=position_ids[0],
                pad_token_id=int(getattr(self.config, "pad_token_id", 0) or 0),
                **kwargs,
            )
            mtp_logits = [self.lm_head(h) for h in mtp_hidden_states]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        return Qwen3_5MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            mtp_logits=mtp_logits,
            mtp_hidden_states=mtp_hidden_states,
        )
