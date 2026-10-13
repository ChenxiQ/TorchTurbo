"""Pure forward-FLOP primitives (2 ops per MAC; stdlib-only).

Forward→training conversion (model_flops=fwd*3, hardware_flops=fwd*4 with
activation checkpointing) lives in ``estimators``.
"""

from __future__ import annotations

def linear_flops(batch_elems: int, in_features: int, out_features: int) -> int:
    """Linear over ``batch_elems`` rows (batch_elems = batch*seq)."""
    return 2 * batch_elems * in_features * out_features

# Convolution

def conv2d_flops(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    kernel_size: int,
    stride: int,
) -> int:
    """FLOPs of a Conv2d over a ``(B, C, H, W)`` input (square kernel/stride)."""
    out_h = (height - kernel_size) // stride + 1
    out_w = (width - kernel_size) // stride + 1
    return 2 * in_channels * kernel_size * kernel_size * out_h * out_w * out_channels * batch_size

def depthwise_conv1d_flops(
    batch_size: int,
    seq_length: int,
    channels: int,
    kernel_size: int,
) -> int:
    """Depthwise causal Conv1d (``groups == D``) used in linear-attention layers.

    Cost: ``2 * B * T * D * W`` — the conv only; elementwise silu/bias are
    negligible.
    """
    return 2 * batch_size * seq_length * channels * kernel_size

def conv3d_flops(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    depth: int,
    height: int,
    width: int,
    kernel_d: int,
    kernel_h: int,
    kernel_w: int,
) -> int:
    """FLOPs of a Conv3d (cube stride == kernel assumed) over a ``(B,C,D,H,W)`` input.

    ``D`` is the temporal extent (frames). Used by VLM patch-embedding towers.
    """
    out_d = depth // kernel_d
    out_h = height // kernel_h
    out_w = width // kernel_w
    kvol = kernel_d * kernel_h * kernel_w
    return 2 * in_channels * out_channels * kvol * out_d * out_h * out_w * batch_size

def vit_attention_flops(
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    num_heads: int,
) -> int:
    """Forward FLOPs of one **non-causal** ViT self-attention block.

    Uses a fused QKV projection (``hidden -> 3*hidden``) and a separate output
    projection (``hidden -> hidden``); ``head_dim = hidden / num_heads``.
    Scores/softmax-weighting are O(S^2) per head over ``hidden`` total.
    """
    del num_heads  # full attention cost collapses to hidden_size (num_heads*head_dim)
    bs = batch_size * seq_length
    qkv = linear_flops(bs, hidden_size, hidden_size * 3)
    qk_product = 2 * batch_size * seq_length * seq_length * hidden_size
    pv_product = 2 * batch_size * seq_length * seq_length * hidden_size
    o_proj = linear_flops(bs, hidden_size, hidden_size)
    return qkv + qk_product + pv_product + o_proj

# Attention variants

def mha_attention_flops(
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int | None = None,
    q_dim: int | None = None,
    k_dim: int | None = None,
    v_dim: int | None = None,
    o_dim: int | None = None,
) -> int:
    """Forward FLOPs of one MHA/GQA attention layer (theoretical).

    Notation: ``B = batch_size``, ``S = seq_length`` (per-step length of the
    O(S^2) qk/pv matmuls; pass the real packed per-segment length when summing
    over segments), ``nh = num_heads``, ``nkv = num_kv_heads`` (GQA shrinks k/v
    heads; equals nh in MHA), ``hd = head_dim`` (the contraction dim of qk/pv).
    ``q_dim``/``k_dim``/``v_dim``/``o_dim`` are *per-head* projection output
    dims, default ``hd``; the projection output total is ``heads * per_head_dim``.

    Shapes per layer:

        q_proj : [B, S, hidden] -> [B, S, nh*q_dim]  -> [B, S, nh,  q_dim]   q_dim = hd
        k_proj : [B, S, hidden] -> [B, S, nkv*k_dim] -> [B, S, nkv, k_dim]   k_dim = hd
        v_proj : [B, S, hidden] -> [B, S, nkv*v_dim] -> [B, S, nkv, v_dim]   v_dim = hd
        q      : [B, nh,  S, hd]      k, v : [B, nkv, S, hd]   (GQA broadcasts nkv -> nh)
        scores : [B, nh, S, S]  = q @ k^T   (contract over hd)  -> 2 * B * nh * S^2 * hd
        ctx    : [B, nh, S, hd] = scores @ v (contract over S)  -> 2 * B * nh * S^2 * hd
        o_proj : [B, S, nh*o_dim] -> [B, S, hidden]            o_dim = hd

    qk/pv contract over ``hd``. ``q_dim`` must equal ``k_dim``; extra channels
    a fused q_proj emits (e.g. a gate) must be cost separately, not via ``q_dim``.
    """
    if num_kv_heads is None:
        num_kv_heads = num_heads
    if q_dim is None:
        q_dim = head_dim
    if k_dim is None:
        k_dim = head_dim
    if v_dim is None:
        v_dim = head_dim
    if o_dim is None:
        o_dim = head_dim
    assert q_dim == k_dim, f"q_dim ({q_dim}) must equal k_dim ({k_dim})"

    bs = batch_size * seq_length
    q_proj = linear_flops(bs, hidden_size, num_heads * q_dim)
    k_proj = linear_flops(bs, hidden_size, num_kv_heads * k_dim)
    v_proj = linear_flops(bs, hidden_size, num_kv_heads * v_dim)
    qk_product = 2 * batch_size * seq_length * seq_length * num_heads * head_dim
    pv_product = 2 * batch_size * seq_length * seq_length * num_heads * head_dim
    o_proj = linear_flops(bs, num_heads * o_dim, hidden_size)
    return q_proj + k_proj + v_proj + qk_product + pv_product + o_proj

def mla_attention_flops(
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    num_heads: int,
    q_lora_rank: int | None,
    kv_lora_rank: int | None,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int = 0,
    v_head_dim: int | None = None,
    has_q_down_proj: bool = True,
    has_kv_down_proj: bool = True,
) -> int:
    """Forward FLOPs of one Multi-head Latent Attention layer (theoretical).

    MLA (DeepSeek-V2/V3) compresses q and kv into low-rank latent vectors and
    only stores the latent kv cache; the up-projections reconstruct per-head
    q/k/v on the fly.

    Notation: ``B = batch_size``, ``S = seq_length`` (per-step length of the
    O(S^2) qk/pv matmuls), ``nh = num_heads``, ``H = hidden_size``,
    ``dnope = qk_nope_head_dim``, ``drope = qk_rope_head_dim``,
    ``dhead = dnope + drope`` (q/k contraction dim), ``dv = v_head_dim``
    (defaults to ``dnope``). ``ql = q_lora_rank``, ``kvl = kv_lora_rank``.

    Shapes per layer (down/up are the latent compress/expand matmuls; pass
    ``has_*_down_proj=False`` when fused away, e.g. some inference paths):

        q_down : [B, S, H]    -> [B, S, ql]            (latent compress)
        q_up   : [B, S, ql]   -> [B, S, nh*dhead] -> [B, nh, S, dhead]
        kv_a   : [B, S, H]    -> [B, S, kvl + drope]   (latent + 1 shared rope k)
                                                     -> k_pe [B, 1, S, drope] (broadcast to nh)
        kv_up  : [B, S, kvl]  -> [B, S, nh*(dnope+dv)] -> k_nope [B, nh, S, dnope]
                                                     -> v      [B, nh, S, dv]
        q, k_nope, k_pe : [B, nh, S, dnope] / [B, nh, S, drope]   (k = concat(k_nope, k_pe))
        scores : [B, nh, S, S] = q @ k^T  (contract over dhead = dnope+drope)  -> 2 * B * nh * S^2 * dhead
        ctx    : [B, nh, S, dv] = scores @ v  (contract over S)               -> 2 * B * nh * S^2 * dv
        o_proj : [B, S, nh*dv] -> [B, S, H]

    qk contracts over the full ``dhead`` (nope + rope; the rope part comes from
    the shared ``k_pe`` broadcast across heads). pv contracts over ``dv``.
    """
    if v_head_dim is None:
        v_head_dim = qk_nope_head_dim
    bs = batch_size * seq_length
    dhead = qk_nope_head_dim + qk_rope_head_dim

    # q path: latent compress then expand to nh * (nope + rope).
    if has_q_down_proj and q_lora_rank is not None:
        q_down = linear_flops(bs, hidden_size, q_lora_rank)
        q_up = linear_flops(bs, q_lora_rank, num_heads * dhead)
    else:
        q_down = 0
        q_up = linear_flops(bs, hidden_size, num_heads * dhead)

    # kv path: kv_a compresses to (latent + shared rope k); kv_up expands the
    # latent to per-head k_nope + v.
    if has_kv_down_proj and kv_lora_rank is not None:
        kv_down = linear_flops(bs, hidden_size, kv_lora_rank + qk_rope_head_dim)
        kv_up = linear_flops(bs, kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim))
    else:
        kv_down = 0
        kv_up = linear_flops(bs, hidden_size, num_heads * (qk_nope_head_dim + v_head_dim))

    qk_product = 2 * batch_size * seq_length * seq_length * num_heads * dhead
    pv_product = 2 * batch_size * seq_length * seq_length * num_heads * v_head_dim
    # output projection (full-rank back to hidden).
    o_proj = linear_flops(bs, num_heads * v_head_dim, hidden_size)
    return q_down + q_up + kv_down + kv_up + qk_product + pv_product + o_proj

def linear_attention_flops(
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    conv_kernel_size: int = 1,
) -> int:
    """Forward FLOPs of one linear-attention layer (Qwen3.5 GatedDeltaNet).

    Mirrors ``Qwen3_5MoeGatedDeltaNet``: linear attention is O(S) (not O(S^2)) —
    per token each value head keeps a ``[head_k_dim, head_v_dim]`` state and the
    recurrent step (see :func:`_recurrence_flops`, ``torch_recurrent_gated_delta_rule``)
    runs over it. Costed: fused ``in_proj_qkv`` (hidden -> 2*key_dim + value_dim),
    the per-head gates ``in_proj_z`` (value_dim), ``in_proj_b`` / ``in_proj_a``
    (num_v_heads), the depthwise-1D conv, the recurrence, and ``out_proj``
    (value_dim -> hidden). q/k are expanded (``repeat_interleave``) to
    ``num_v_heads`` before the recurrence, so the recurrence uses v-heads.
    """
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    bs = batch_size * seq_length

    # Fused QKV: the model emits (2*key_dim + value_dim) channels — q, k, v.
    qkv_proj = linear_flops(bs, hidden_size, 2 * key_dim + value_dim)
    # Gating / per-head modulation projections (z=value_dim gate, b/a=v_heads).
    gate_projs = (
        linear_flops(bs, hidden_size, value_dim)      # z (output gate)
        + linear_flops(bs, hidden_size, num_v_heads)  # b
        + linear_flops(bs, hidden_size, num_v_heads)  # a
    )
    o_proj = linear_flops(bs, value_dim, hidden_size)
    conv = depthwise_conv1d_flops(batch_size, seq_length, 2 * key_dim + value_dim, conv_kernel_size)
    recurrence = recurrence_flops(batch_size, seq_length, num_v_heads, head_k_dim, head_v_dim)
    return qkv_proj + gate_projs + o_proj + conv + recurrence

def recurrence_flops(
    batch_size: int,
    seq_length: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> int:
    """FLOPs of the GatedDeltaNet recurrent step ``torch_recurrent_gated_delta_rule``.

    Notation: ``B`` = batch_size, ``S`` = seq_length, ``H`` = num_v_heads,
    ``K`` = head_k_dim, ``V`` = head_v_dim. The loop runs over ``S`` tokens;
    each iteration's tensors (per token ``i``) are:

        state   : [B, H, K, V]   (recurrent, carried across tokens)
        q_t     : [B, H, K]      (query for token i)
        k_t     : [B, H, K]      (key   for token i)
        v_t     : [B, H, V]      (value for token i)
        g_t     : [B, H, 1, 1]   (decay gate, scalar per head)
        beta_t  : [B, H, 1]      (delta-rule gate, scalar per head)

    Per-token step (shapes shown):

        # 1. decay gate:  state = state * g_t                [B,H,K,V] * [B,H,1,1] -> [B,H,K,V]
        # 2. pre-read:     kv_mem = (state * k_t[...,None]).sum(K)   [B,H,K,V]*[B,H,K,1] -> sum(K) -> [B,H,V]
        # 3. delta rule:   delta = (v_t - kv_mem) * beta_t           [B,H,V] - [B,H,V] -> [B,H,V]
        # 4. outer update: state = state + k_t[...,None] * delta[...,None,:]  outer [B,H,K,V] -> [B,H,K,V]
        # 5. readout:      o = (state * q_t[...,None]).sum(K)        [B,H,K,V]*[B,H,K,1] -> sum(K) -> [B,H,V]

    FLOPs per token per head (1 per scalar op; ``K`` over head_k_dim):

        1. state * g           : K*V    (K*V mul by scalar; no add)
        2. (state*k)...sum(K)  : 2*K*V  (K*V mul + ~K*V add)
        3. (v - kv_mem)*beta   : 2*V    (V sub + V mul)
        4. k * delta  (outer)  : 2*K*V  (K*V mul + K*V add)
        5. (state*q)...sum(K)  : 2*K*V  (K*V mul + ~K*V add)

    Total: ``7*K*V + 2*V`` per (token, head). Times ``H * B * S``.

    (Step 1 is mul-only — the decay gate ``g`` is a per-head scalar, so it adds
    no additions unlike the MAC steps 2/4/5.)
    """
    kv = head_k_dim * head_v_dim
    per_token_per_head = 7 * kv + 2 * head_v_dim
    return per_token_per_head * num_v_heads * batch_size * seq_length

# Feed-forward / MLP

def gated_mlp_flops(batch_size: int, seq_length: int, hidden_size: int, intermediate_size: int) -> int:
    """Gated MLP (SwiGLU): gate, up, down — three matmuls."""
    bs = batch_size * seq_length
    return linear_flops(bs, hidden_size, intermediate_size) * 3

def plain_mlp_flops(batch_size: int, seq_length: int, hidden_size: int, intermediate_size: int) -> int:
    """Non-gated MLP: up + down — two matmuls."""
    bs = batch_size * seq_length
    return linear_flops(bs, hidden_size, intermediate_size) * 2

def moe_flops(
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    expert_intermediate_size: int,
    num_experts: int,
    top_k: int,
    shared_expert_intermediate_size: int = 0,
    use_gate: bool = True,
    activation_checkpoint: bool = False,
) -> int:
    """Forward FLOPs of one MoE FFN layer (activated-expert convention).

    Counts only the **activated** routed experts (``top_k``) plus the shared
    expert (if any) plus the router gating linear. This is the standard MFU
    accounting — the matmul cost a token actually incurs, not ``num_experts``
    experts fully evaluated. The router scans all experts so its cost is
    ``2 * B * S * H * num_experts``.

    ``use_gate`` toggles whether experts are gated (SwiGLU, 3 matmuls) or plain
    (2 matmuls). ``activation_checkpoint`` is accepted for symmetry but has no
    effect on forward FLOPs.
    """
    del activation_checkpoint  # forward only; AC handled in core
    bs = batch_size * seq_length
    per_expert = gated_mlp_flops(batch_size, seq_length, hidden_size, expert_intermediate_size) if use_gate else \
        plain_mlp_flops(batch_size, seq_length, hidden_size, expert_intermediate_size)
    routed = per_expert * top_k

    shared = 0
    if shared_expert_intermediate_size > 0:
        shared = gated_mlp_flops(batch_size, seq_length, hidden_size, shared_expert_intermediate_size) \
            if use_gate else plain_mlp_flops(batch_size, seq_length, hidden_size, shared_expert_intermediate_size)

    router = linear_flops(bs, hidden_size, num_experts)
    return routed + shared + router

# Head

def logits_flops(batch_size: int, seq_length: int, hidden_size: int, vocab_size: int) -> int:
    """Final classification head: ``hidden -> vocab`` once per token."""
    return linear_flops(batch_size * seq_length, hidden_size, vocab_size)

def embedding_flops(batch_size: int, seq_length: int, vocab_size: int, hidden_size: int) -> int:
    """Token embedding lookup cost — a gather, so this is a conservative upper bound.

    Most MFU estimates omit it.
    """
    del vocab_size
    return 2 * batch_size * seq_length * hidden_size

# Vision encoder (Qwen-VL style ViT) — per-image, variable resolution

def vision_encoder_flops(
    grid_thw_list,
    vision_config,
    *,
    include_merger: bool = True,
) -> tuple[int, int]:
    """Forward FLOPs of a Qwen-VL vision tower over a list of images/videos.

    Parameters
    ----------
    grid_thw_list:
        Iterable of ``(T, H, W)`` patch-grid triples — one per image/video in
        the batch, exactly what a model's ``image_grid_thw`` /
        ``video_grid_thw`` tensors carry (each row is ``(t, h, w)`` in patch
        units). Resolutions may differ per item, so FLOPs are summed per item.
    vision_config:
        The model's ``vision_config`` ``PretrainedConfig`` (depth / hidden_size /
        intermediate_size / in_channels / patch_size / temporal_patch_size /
        spatial_merge_size / out_hidden_size / num_heads), accessed by attribute.
    include_merger:
        Count the PatchMerger's fc1/fc2 projections. Checkpoints with
        ``disable_merger_proj=True`` keep only the merger's norm, so those
        recipes pass False.

    Returns
    ``(total_forward_flops, num_merged_tokens)`` where ``num_merged_tokens`` is
    the total visual tokens merged into the LLM sequence (each ``spatial_merge²``
    patches collapse to one). The merged tokens are already counted by the
    decoder FLOPs, so this returns only the vision encoder + merger cost.

    Attention is **non-causal, per-image** (varlen), so it is O(P_i^2) per item;
    the caller sums per-item FLOPs.
    """
    depth = int(getattr(vision_config, "depth", getattr(vision_config, "num_hidden_layers", 0)))
    hidden = int(getattr(vision_config, "hidden_size", 0))
    intermediate = int(getattr(vision_config, "intermediate_size", 0))
    num_heads = int(getattr(vision_config, "num_heads", getattr(vision_config, "num_attention_heads", 1)))
    in_channels = int(getattr(vision_config, "in_channels", 3))
    patch = int(getattr(vision_config, "patch_size", 16))
    tpatch = int(getattr(vision_config, "temporal_patch_size", 1))
    sm = int(getattr(vision_config, "spatial_merge_size", getattr(vision_config, "spatial_merge", 1)) or 1)
    out_hidden = int(getattr(vision_config, "out_hidden_size", getattr(vision_config, "out_hidden", hidden)))
    if hidden <= 0 or patch <= 0 or depth <= 0:
        return 0, 0

    merge_in = hidden * sm * sm
    total_flops = 0
    merged_tokens = 0
    for t, h, w in grid_thw_list:
        t, h, w = int(t), int(h), int(w)
        if t <= 0 or h <= 0 or w <= 0:
            continue
        P = t * h * w  # patches for this item
        # 1) Conv3d patch embed over (t*tp, h*patch, w*patch) pixels.
        conv = conv3d_flops(1, in_channels, hidden, t * tpatch, h * patch, w * patch, tpatch, patch, patch)
        # 2) depth ViT blocks: non-causal attention O(P^2) + plain MLP.
        attn = vit_attention_flops(1, P, hidden, num_heads)
        mlp = plain_mlp_flops(1, P, hidden, intermediate) if intermediate > 0 else 0
        # 3) PatchMerger: P patches -> P/sm^2 tokens; fc1 then fc2.
        out_t = P // (sm * sm) if sm > 0 else P
        merger = (
            linear_flops(out_t, merge_in, merge_in) + linear_flops(out_t, merge_in, out_hidden)
            if include_merger else 0
        )
        total_flops += conv + depth * (attn + mlp) + merger
        merged_tokens += out_t
    return total_flops, merged_tokens