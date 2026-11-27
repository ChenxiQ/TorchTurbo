"""AscendC implementation of chunked Kimi Delta Attention."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import torch

from torchturbo.runtime import TurboDevice
from torchturbo.utils.functools import cache

from ...base import OpImpl, OpImplKind
from ...registry import register_op


def _chunk_indices(cu_seqlens):
    indices = []
    for seq, (left, right) in enumerate(pairwise(cu_seqlens)):
        for chunk in range((right - left + 63) // 64):
            indices.extend((seq, chunk))
    return indices


@cache
def _prepare_kda_metadata(cu_seqlens, _version):
    """Cache host boundaries and 64-token chunk indices across layers."""
    cu_host = cu_seqlens.detach().cpu().tolist()
    return cu_host, _chunk_indices(cu_host)


def _cp_forward(
    ctx: Any,
    cp_context: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    scale: float,
    lower_bound: float | None,
    cu_seqlens: torch.Tensor | None,
    save_backward: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Head-major three-stage forward for context parallelism.

    Deliberately separate from the fused single-rank path: the CP kernels
    exchange recurrent states across ranks, and the fused backward rejects
    ``dht``, so the whole CP leg runs on the three-stage operators.
    """
    from .kda_cp import cp_forward

    # CP always runs packed, and the boundaries must be this rank's shard --
    # not whatever global cu_seqlens the caller happens to pass. Using the
    # global list here would make the three-stage backward size its kernels
    # for the full sequence against local-shaped caches.
    cu_host = list(cp_context.cu_seqlens)
    ctx.g_dtype, ctx.beta_dtype = g.dtype, beta.dtype

    # The dispatcher squeezes the batch dim off q/k/v only, so g/beta still
    # carry it and must drop it here before going head-major.
    q_h = q.transpose(0, 1).contiguous()
    k_h = k.transpose(0, 1).contiguous()
    v_h = v.transpose(0, 1).contiguous()
    g_h = g.squeeze(0).transpose(0, 1).contiguous()
    beta_h = beta.squeeze(0).transpose(0, 1).contiguous()

    outputs, q_hat, k_hat, q_rstd, k_rstd, beta_eff = cp_forward(
        cp_context, q_h, k_h, v_h, g_h, beta_h, scale,
        safe_gate=True, lower_bound=lower_bound,
        A_log=A_log, dt_bias=dt_bias,
    )
    output, _final_state, gk, aqk, akk, w, u, qg, kg, v_new, h, _h0 = outputs

    ctx.cp_context = cp_context
    ctx.scale = scale
    ctx.lower_bound = lower_bound
    ctx.use_gate_in_kernel = True
    ctx.use_beta_sigmoid_in_kernel = True
    ctx.cu_seqlens = cu_host
    ctx.set_materialize_grads(False)
    if save_backward:
        # The three-stage backward needs the full head-major cache set.
        ctx.save_for_backward(
            q_hat, k_hat, v_h, g_h, beta_h, A_log, dt_bias,
            gk, aqk, akk, w, u, qg, kg, v_new, h,
            q_rstd, k_rstd, beta_eff,
        )

    if output.dim() == 3:
        output = output.unsqueeze(0)
    return output, None


def _cp_backward(
    ctx: Any,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, ...]:
    """Three-stage backward with cross-rank dht exchange.

    Mirrors the reference CP leg: prepare -> boundary dht -> dhu -> finalize,
    then restores the model's token-major spelling for the q/k/v gradients.
    """
    from fla_npu.ops.ascendc import (
        npu_chunk_gated_delta_rule_bwd_dhu,
        npu_chunk_kda_bwd_finalize,
        npu_chunk_kda_bwd_prepare,
    )

    from .kda_cp import cp_backward_boundary

    (
        q, k, v, g, beta, A_log, dt_bias,
        gk, aqk, akk, w, u, qg, kg, v_new, h,
        q_rstd, k_rstd, beta_eff,
    ) = ctx.saved_tensors
    packed = ctx.cu_seqlens is not None
    layout = "NTD" if packed else "BNSD"

    # cp_forward runs the BNSD spelling, so every cache comes back with a
    # leading batch dim; the NTD three-stage path wants rank-3 head-major.
    # v/g/beta were already squeezed on the way in.
    if packed:
        q, k, gk, aqk, akk, w, u, qg, kg, v_new, h = (
            tensor.squeeze(0).contiguous()
            for tensor in (q, k, gk, aqk, akk, w, u, qg, kg, v_new, h)
        )
        q_rstd, k_rstd, beta_eff = (
            tensor.squeeze(0).contiguous()
            for tensor in (q_rstd, k_rstd, beta_eff)
        )

    # grad_output arrives token-major; the CP kernels want head-major.
    do_h = grad_output.transpose(1, 2).contiguous()
    if packed:
        do_h = do_h.squeeze(0)

    # fwd_prepare derives this itself; dhu and finalize demand it alongside
    # cu_seqlens.
    from fla_npu.ops.ascendc._stable import _canonical_chunk_indices
    chunk_indices = _canonical_chunk_indices(ctx.cu_seqlens, 64)

    d_aqk, dv_local, dq_raw = npu_chunk_kda_bwd_prepare(
        aqk, v_new, do_h, h, scale=ctx.scale, layout=layout,
        chunk_size=64, cu_seqlens=ctx.cu_seqlens,
    )

    # Dhu takes rank-4 token tensors; adding B=1 is a view.
    scan = (qg, kg, w, do_h, dv_local, gk)
    if packed:
        scan = tuple(tensor.unsqueeze(0) for tensor in scan)
    qg_scan, kg_scan, w_scan, do_scan, dv_scan_in, gk_scan = scan

    dht = cp_backward_boundary(
        ctx.cp_context, qg_scan, kg_scan, w_scan, do_scan, dv_scan_in,
        gk_scan, ctx.scale,
    )
    dh, _dh0, dv_scan = npu_chunk_gated_delta_rule_bwd_dhu(
        qg_scan, kg_scan, w_scan, do_scan, dv_scan_in, ctx.scale, 64,
        gK=gk_scan, cu_seqlens=ctx.cu_seqlens, chunk_indices=chunk_indices,
        use_exp2=True, dht=dht,
    )
    if packed:
        dh, dv_scan = dh.squeeze(0), dv_scan.squeeze(0)

    # Finalize's fused-sigmoid dtype row is raw_g=BF16/FP32, beta=FP32,
    # d_beta=BF16 (chunk_kda_bwd_finalize_def.cpp). Keep raw_g in the model's
    # own dtype so no FP32 copy is allocated, but beta must be FP32.
    bias = dt_bias
    if bias is None:
        bias = torch.zeros(
            (q.shape[0], q.shape[-1]), dtype=torch.float32, device=q.device,
        )
    dq_h, dk_h, dv_h, dbeta, dg, dA, dbias = npu_chunk_kda_bwd_finalize(
        q, k, v, gk, g, beta.float(), A_log, bias, akk, v_new, h, dh,
        dv_scan, d_aqk, dq_raw,
        scale=ctx.scale, lower_bound=ctx.lower_bound, chunk_size=64,
        safe_gate=True, use_gate_in_kernel=ctx.use_gate_in_kernel,
        q_rstd=q_rstd, k_rstd=k_rstd,
        cu_seqlens=ctx.cu_seqlens, chunk_indices=chunk_indices,
        output_layout=layout,
        use_beta_sigmoid_in_kernel=ctx.use_beta_sigmoid_in_kernel,
        use_exp2=True,
    )

    def to_token_major(tensor: torch.Tensor) -> torch.Tensor:
        if packed:
            return tensor.unsqueeze(0).transpose(1, 2).contiguous()
        return tensor.transpose(1, 2).contiguous()

    dq = to_token_major(dq_h).to(q.dtype)
    dk = to_token_major(dk_h).to(k.dtype)
    dv = to_token_major(dv_h).to(v.dtype)
    dg = to_token_major(dg).to(ctx.g_dtype)
    dbeta = to_token_major(dbeta).to(ctx.beta_dtype)

    dA = None if A_log is None else dA.to(A_log)
    if dbias is not None and dt_bias is not None:
        if dbias.dim() == 2:
            dbias = dbias.reshape(-1)
        dbias = dbias.to(dt_bias)
    dbias = None if dt_bias is None else dbias

    return (
        dq, dk, dv, dg, dbeta, dA, dbias,
        None, None, None, None, None, None, None, None, None,
    )


class ChunkKDAFunction(torch.autograd.Function):
    """Connect token-major model tensors to the AscendC KDA operators."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor | None,
        dt_bias: torch.Tensor | None,
        scale: float,
        output_final_state: bool,
        lower_bound: float | None,
        use_gate_in_kernel: bool,
        cu_seqlens: torch.Tensor | None,
        use_qk_l2norm_in_kernel: bool,
        use_beta_sigmoid_in_kernel: bool,
        save_backward: bool,
        cp_context: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if cp_context is not None:
            return _cp_forward(
                ctx, cp_context, q, k, v, g, beta, A_log, dt_bias,
                scale, lower_bound, cu_seqlens, save_backward,
            )

        from fla_npu.ops.ascendc import chunk_kda_fwd

        cu_host = indices = None
        if cu_seqlens is not None:
            cu_host, indices = _prepare_kda_metadata(cu_seqlens, cu_seqlens._version)
        packed = cu_host is not None
        ctx.g_dtype, ctx.beta_dtype = g.dtype, beta.dtype
        if packed:
            g, beta = g.squeeze(0), beta.squeeze(0)
        g, beta = g.contiguous(), beta.contiguous()
        q_hat = k_hat = q_rstd = k_rstd = None
        # Normalization caches are head-major regardless of the input layout.
        token_axis = 0 if packed else 1
        head_shape = q.transpose(token_axis, token_axis + 1).shape
        if use_qk_l2norm_in_kernel and save_backward:
            q_hat = torch.empty(head_shape, dtype=q.dtype, device=q.device)
            k_hat = torch.empty(head_shape, dtype=k.dtype, device=k.device)
            q_rstd = torch.empty(head_shape[:-1], dtype=torch.float32, device=q.device)
            k_rstd = torch.empty(head_shape[:-1], dtype=torch.float32, device=k.device)
        beta_eff = None
        if use_beta_sigmoid_in_kernel and save_backward:
            beta_eff = torch.empty(head_shape[:-1], dtype=torch.float32, device=beta.device)
        outputs = chunk_kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            64,
            layout="TND" if packed else "BSND",
            output_final_state=output_final_state,
            cu_seqlens=cu_host,
            chunk_indices=indices,
            safe_gate=True,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            disable_recompute=False,
            use_exp2=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            q_hat_out=q_hat,
            k_hat_out=k_hat,
            q_rstd_out=q_rstd,
            k_rstd_out=k_rstd,
            beta_eff_out=beta_eff,
        )
        output, final_state, _gk, aqk, akk, _w, _u, _qg, _kg, _v_new, _h, _state = outputs
        if q_hat is not None:
            q, k = q_hat, k_hat
        if save_backward:
            ctx.save_for_backward(
                q, k, v, g, beta, A_log, dt_bias, aqk, akk,
                q_rstd, k_rstd, beta_eff,
            )
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        ctx.cu_seqlens = cu_host
        ctx.chunk_indices = indices
        # An unused final state must arrive as None, not a materialized zero
        # gradient: this kernel does not implement final-state backward.
        ctx.set_materialize_grads(False)
        if packed:
            output = output.unsqueeze(0)
        return output, final_state

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
        grad_final_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        # A zero grad is the discarded output's placeholder; a non-zero
        # one means the loss depends on final_state, which has no backward.
        if grad_final_state is not None and bool(grad_final_state.any()):
            raise NotImplementedError(
                "AscendC KDA does not support gradients through final_state."
            )
        if getattr(ctx, "cp_context", None) is not None:
            return _cp_backward(ctx, grad_output)

        from fla_npu.ops.ascendc import npu_chunk_kda_bwd
        (
            q, k, v, g, beta, A_log, dt_bias, aqk, akk,
            q_rstd, k_rstd, beta_eff,
        ) = ctx.saved_tensors
        packed = ctx.cu_seqlens is not None
        if packed:
            grad_output = grad_output.squeeze(0)
        # Model tensors and output gradients stay token-major. The upstream
        # backward rebuilds the omitted forward caches.
        dq, dk, dv, dbeta, dg, _dh0, dA, dbias = npu_chunk_kda_bwd(
            q, k, v, beta, None, aqk, akk, None, None, None, None, None,
            grad_output.contiguous(), ctx.scale,
            raw_g=g, A_log=A_log, dt_bias=dt_bias,
            lower_bound=ctx.lower_bound, chunk_size=64,
            safe_gate=True, use_gate_in_kernel=ctx.use_gate_in_kernel,
            disable_recompute=False, use_exp2=True,
            q_rstd=q_rstd, k_rstd=k_rstd,
            cu_seqlens=ctx.cu_seqlens, chunk_indices=ctx.chunk_indices,
            layout="TND" if packed else "BSND",
            use_beta_sigmoid_in_kernel=ctx.use_beta_sigmoid_in_kernel,
            beta_eff=beta_eff,
        )
        dbias = None if dt_bias is None else dbias.reshape_as(dt_bias)
        # Gate and beta gradients have the original token-major layout.
        if packed:
            dg, dbeta = dg.unsqueeze(0), dbeta.unsqueeze(0)
        return (
            dq,
            dk,
            dv,
            dg.to(ctx.g_dtype),
            dbeta.to(ctx.beta_dtype),
            dA,
            dbias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # cp_context
        )


def ascendc_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = True,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cp_context: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run KDA on [B,T,C] Q/K/V inputs and return time-major output.

    A2/A3 training requires BF16 Q/K/V, safe gate computed in the kernel,
    equal Q/K/V head counts, K=V=128, and no state gradients.

    ``cp_context`` selects the context-parallel leg, which runs the
    three-stage operators because the fused backward rejects ``dht``.
    """
    if not safe_gate:
        raise NotImplementedError(
            "AscendC KDA training requires safe_gate=True. "
            "Select fla_kda for other gate configurations."
        )
    if initial_state is not None:
        raise NotImplementedError(
            "AscendC KDA training does not support initial_state."
        )
    if not use_gate_in_kernel:
        raise NotImplementedError(
            "AscendC KDA training requires use_gate_in_kernel=True."
        )
    if cu_seqlens is not None and q.size(0) != 1:
        raise ValueError("Packed KDA requires B=1.")

    qk_head_dim = g.shape[-1]
    v_head_dim = v.shape[-1] // beta.shape[-1]
    q = q.unflatten(-1, (-1, qk_head_dim))
    k = k.unflatten(-1, (-1, qk_head_dim))
    v = v.unflatten(-1, (-1, v_head_dim))

    # AscendC consumes FP32 gate parameters; these casts retain gradients
    # to the model's mixed-precision parameters and their local head slices.
    if A_log is not None:
        A_log = A_log.float()
    if dt_bias is not None:
        dt_bias = dt_bias.float()
    if scale is None:
        scale = q.size(-1) ** -0.5
    if lower_bound is None:
        lower_bound = -5.0

    inputs = []
    for tensor in (q, k, v):
        if cu_seqlens is not None:
            tensor = tensor.squeeze(0)
        inputs.append(tensor.contiguous())
    output, final_state = ChunkKDAFunction.apply(
        *inputs,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        output_final_state,
        lower_bound,
        use_gate_in_kernel,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel,
        torch.is_grad_enabled(),
        cp_context,
    )
    return output, final_state


@register_op(
    op_name="kda",
    impl_name="ascendc_kda",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class AscendCKDA(OpImpl):
    def available(self) -> bool:
        try:
            from fla_npu.ops.ascendc import (  # noqa: F401
                chunk_kda_fwd,
                npu_chunk_kda_bwd,
            )
        except ImportError:
            return False
        return True

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = True,
        use_gate_in_kernel: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cp_context: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return ascendc_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
        )
