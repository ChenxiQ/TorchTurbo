from __future__ import annotations

import torch

from torchturbo.ops import ops


def apply_rotary_pos_emb_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = ops.rotary_position_embedding(q, cos, sin, 0)
    k_embed = ops.rotary_position_embedding(k, cos, sin, 0)
    return q_embed, k_embed


def rms_norm_forward_npu(self, hidden_states: torch.Tensor):
    return ops.rms_norm(hidden_states, self.weight, self.variance_epsilon)
