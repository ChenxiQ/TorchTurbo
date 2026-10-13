from __future__ import annotations

from torchturbo.ops import ops


def apply_rotary_pos_emb_kimi_moe(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    q_embed = ops.rotary_position_embedding(q, cos, sin, 0)
    k_embed = ops.rotary_position_embedding(k, cos, sin, 0)
    return q_embed, k_embed
