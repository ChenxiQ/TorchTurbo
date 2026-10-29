"""AscendC KDA (chunk_kda) backend for the bailing_hybrid linear-attention layer.

This package vendors the full-AscendC training adapter for the ``fla_npu``
KDA operators (see ``ascendc_chunk_kda_function.py``): both the forward and
the backward pass run the AscendC kernels. The updated A2/A3 interface supports
fused Q/K L2 normalization and beta sigmoid in both passes. With sigmoid
fusion enabled, callers pass raw beta logits; forward saves FP32 effective
beta for backward.

:func:`ascendc_chunk_kda` keeps fla-org's ``chunk_kda`` calling convention
(``g``/``beta`` are ``[B, T, H, ...]``) and returns ``(o, final_state)``, with
``o`` always time-first ``[B, T, HV, V]``.  ``q``/``k``/``v`` may be either
time-first ``[B, T, H, D]`` or, with ``head_major_qkv=True``, the head-first
``[B, H, T, D]`` spelling the fused short convolution produces -- the latter
keeps the whole Q/K/V path free of layout transposes.  Both packed varlen
(``cu_seqlens`` given) and dense (``cu_seqlens=None``) inputs are supported.

AllGather CP accepts a KDAContextParallel context with local packed boundaries.
Unsupported configurations (inference intermediate states,
V-first state layout, GVA,
initial_state, chunk_size != 64, disable_recompute=False) are rejected before
any AscendC kernel is launched.  The ops dispatcher selects an implementation
once via ``available()`` and does NOT retry on exception, so such a raise
aborts the run -- it is not a fallback.
"""

from __future__ import annotations

import torch

from .ascendc_chunk_kda_function import AscendCChunkKDAFunction
from .cp import KDAContextParallel

__all__ = [
    "AscendCChunkKDAFunction",
    "KDAContextParallel",
    "ascendc_chunk_kda",
]


def ascendc_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    chunk_size: int = 64,
    disable_recompute: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    head_major_qkv: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    cp_context: KDAContextParallel | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Drop-in full-AscendC replacement for ``fla.ops.kda.chunk_kda``.

    Only the subset exercised by bailing_hybrid KDA layers (packed varlen or
    dense training) is supported; unsupported keyword combinations raise
    ``ValueError`` **before any kernel launch**.  Note the ops dispatcher picks
    an implementation once via ``available()`` and never retries on exception,
    so a raise here aborts the run rather than falling back to triton -- keep
    the checks below in sync with ``AscendCChunkKDA.available()``.

    ``head_major_qkv=True`` accepts q/k/v in the ``[B, H, T, D]`` spelling the
    fused short convolution already produces; the operator then runs BNSD/NTD
    and the backward returns ``dq``/``dk``/``dv`` head-first, so the whole
    Q/K/V path is free of layout transposes.  ``g``/``beta`` stay
    sequence-major (projection outputs) and are converted here.

    ``disable_recompute`` must be ``True``: the fused AscendC backward only
    consumes forward-saved caches (see the check below).  Run the triton
    provider with the same value so both paths stay in one memory regime.
    Without CP, ``output_final_state`` is honoured, but gradients w.r.t.
    final_state are NOT supported. CP uses internal boundary states and
    requires ``output_final_state=False``.
    """
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("allow_neg_eigval=True requires beta sigmoid fusion.")
    for unsupported in (
        "return_intermediate_states",
    ):
        if kwargs.pop(unsupported, False):
            raise ValueError(
                f"ascendc_chunk_kda does not support {unsupported}=True; use the "
                "fla-org triton provider for this configuration."
            )
    if kwargs:
        raise ValueError(f"ascendc_chunk_kda got unexpected arguments: {sorted(kwargs)}")

    if state_v_first:
        raise ValueError("ascendc_chunk_kda does not support state_v_first=True.")
    if q.ndim != 4:
        raise ValueError(
            f"ascendc_chunk_kda expects 4-D q/k/v, got {tuple(q.shape)}."
        )
    # Head axis: [B,T,H,D] normally, [B,H,T,D] in the head-first spelling.
    head_axis = 1 if head_major_qkv else 2
    if v.shape[head_axis] != q.shape[head_axis]:
        raise ValueError(
            "ascendc_chunk_kda does not support GVA (HV > H); "
            f"got H={q.shape[head_axis]} vs HV={v.shape[head_axis]}."
        )
    if chunk_size != 64:
        # The fused AscendC backward hard-requires chunk_size=64; accepting 128
        # here would let the forward pass and the backward raise mid-training.
        raise ValueError(
            f"ascendc_chunk_kda requires chunk_size=64, got {chunk_size}."
        )
    if not disable_recompute:
        # The fused AscendC backward only consumes forward-saved caches:
        # disable_recompute=False routes to the Ascend950-only optimized path
        # (measured on Ascend910_93: "optimized KDA backward requires
        # Ascend950/A5"), and that path's recompute mode additionally needs
        # H%8==0 plus every sequence length divisible by 64 -- neither holds
        # for the production SP8 shape (H=2, sample-length packing).
        raise ValueError(
            "ascendc_chunk_kda requires disable_recompute=True (the fused "
            "AscendC backward has no recompute path on A2/A3); set it on the "
            "triton provider as well so both paths stay comparable."
        )
    if initial_state is not None:
        raise ValueError(
            "ascendc_chunk_kda does not support user-supplied initial_state "
            "in this training adapter."
        )
    if q.shape[0] != 1:
        # Production always runs B=1 (one packed sequence per micro-batch).
        raise ValueError(
            "ascendc_chunk_kda requires batch size 1; flatten "
            "variable-length inputs into a single sequence with cu_seqlens, "
            "or use the fla-org triton provider."
        )
    if use_gate_in_kernel and A_log is None and lower_bound is None:
        raise ValueError(
            "`A_log` must be provided when `use_gate_in_kernel=True` and `lower_bound` is not set."
        )
    if safe_gate and use_gate_in_kernel:
        if lower_bound is None:
            raise ValueError("`lower_bound` must be specified when `safe_gate=True` and `use_gate_in_kernel=True`.")
        if not (-5 <= lower_bound < 0):
            raise ValueError(f"`lower_bound` must be in the safe range [-5, 0), got {lower_bound}.")

    if scale is None:
        scale = q.shape[-1] ** -0.5

    if cp_context is not None:
        if not isinstance(cp_context, KDAContextParallel):
            raise ValueError("cp_context must be a KDAContextParallel instance.")
        if not head_major_qkv or not all((use_qk_l2norm_in_kernel,
                                         use_beta_sigmoid_in_kernel, use_gate_in_kernel)):
            raise ValueError("KDA CP requires head-major Q/K/V and fused L2, sigmoid and gate.")
        if q.dtype != torch.bfloat16 or q.shape[-1] != 128 or v.shape[-1] != 128:
            raise ValueError("KDA CP currently supports BF16 and key/value dimensions 128.")
        if output_final_state:
            raise ValueError("KDA CP does not expose global final states; set output_final_state=False.")
        if cu_seqlens is None or q.shape[2] != cp_context.cu_seqlens[-1]:
            raise ValueError("KDA CP requires local packed offsets covering the local Q/K/V tokens.")
        src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
        if tuple(src.detach().cpu().tolist()) != cp_context.cu_seqlens:
            raise ValueError("KDA CP cu_seqlens must match the context's local packed offsets.")

    # ``disable_recompute=True`` means the fused backward consumes the
    # forward's saved caches, so they must be exported whenever a backward can
    # run.  NOTE: ``torch.is_grad_enabled()`` does NOT distinguish the two
    # activation-checkpointing passes -- torch 2.10's non-reentrant checkpoint
    # keeps grad enabled in the first pass as well
    # (``_checkpoint_without_reentrant_generator`` has no ``no_grad()``), so
    # BOTH passes export here.  That is harmless: the first pass's saved
    # tensors are dropped by the checkpoint's pack hook, and this only skips
    # the export for genuinely grad-free callers (inference / eval).
    export_intermediates = torch.is_grad_enabled()

    # Work around the slow BF16 gate path in AscendC Prepare.
    # Keep the cast outside Function.apply so autograd restores the input dtype.
    g = g.float()

    return AscendCChunkKDAFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        bool(use_beta_sigmoid_in_kernel),
        bool(allow_neg_eigval),
        state_v_first,
        cu_seqlens,
        cu_seqlens_cpu,
        safe_gate,
        lower_bound,
        chunk_size,
        disable_recompute,
        False,  # return_intermediate_states
        cp_context,
        bool(head_major_qkv),
        export_intermediates,
    )
