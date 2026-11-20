"""Native NPU operator-level context parallelism (CP) for KDA training.

Unlike the Ulysses sequence parallelism (per-layer all-to-all on full
activations), CP keeps the sequence sharded across SP ranks and exchanges
only the tiny recurrent boundary states:

* forward: the KDA/conv state at the left boundary arrives from the previous
  rank; the state at the right boundary is sent to the next rank;
* backward: the state gradient ``dht`` arrives from the next rank; the
  gradient w.r.t. the left-boundary state ``dh0`` is sent back.

The communication volume is O(B * H * K * V) per direction per layer
independent of the sequence length, versus O(T * H * V) for Ulysses.
All kernels are fla_npu AscendC operators (no Triton dependency).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from torchturbo.ops.packed_params import get_packed_params


def _shard_state_flags(
    cu_seqlens_host: list[int] | None,
    sp_rank: int,
    sp_size: int,
    local_tokens: int,
) -> tuple[bool, bool]:
    """Whether this shard starts/ends strictly inside a sequence.

    A state only crosses a rank boundary when the global sequence actually
    continues there; exact boundary alignment needs no exchange.
    """
    if sp_size == 1:
        return False, False
    start = sp_rank * local_tokens
    end = start + local_tokens
    if cu_seqlens_host is None:
        return sp_rank > 0, sp_rank < sp_size - 1
    boundaries = set(cu_seqlens_host)
    return start not in boundaries, end not in boundaries


def _recv_state(group: dist.ProcessGroup, src: int, shape, dtype, device):
    buf = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(buf, src=src, group=group)
    return buf


def _send_state(group: dist.ProcessGroup, dst: int, tensor: torch.Tensor, dtype=None):
    payload = tensor.detach()
    if dtype is not None:
        payload = payload.to(dtype)
    dist.send(payload.contiguous(), dst=dst, group=group)


class CausalConv1dCPFunction(torch.autograd.Function):
    """Causal conv1d with CP boundary-state exchange.

    Forward receives the previous rank's trailing ``kernel_width - 1`` tokens
    as the conv state and ships its own trailing window downstream. Backward
    replays the local conv under autograd so the boundary-state gradients
    (``dht`` in, ``dh0`` out) ride the same p2p chain in reverse.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str | None,
        need_state_in: bool,
        need_state_out: bool,
        sp_group: dist.ProcessGroup,
        sp_rank: int,
    ) -> torch.Tensor:
        from torchturbo.ops import ops

        width = weight.shape[-1]
        state_in = None
        if need_state_in:
            state_in = _recv_state(
                sp_group, sp_rank - 1,
                (x.shape[0], x.shape[-1], width), x.dtype, x.device,
            )
        with torch.no_grad():
            y, final_state = ops.causal_conv1d(
                x=x,
                weight=weight,
                bias=bias,
                activation=activation,
                initial_state=state_in,
                output_final_state=need_state_out,
            )
        if need_state_out:
            _send_state(sp_group, sp_rank + 1, final_state, x.dtype)
        ctx.save_for_backward(x, weight, bias, state_in)
        ctx.activation = activation
        ctx.flags = (need_state_in, need_state_out)
        ctx.comm = (sp_group, sp_rank)
        return y

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor):
        from torchturbo.ops import ops

        x, weight, bias, state_in = ctx.saved_tensors
        need_state_in, need_state_out = ctx.flags
        sp_group, sp_rank = ctx.comm
        dht = None
        if need_state_out:
            dht = _recv_state(
                sp_group, sp_rank + 1,
                (x.shape[0], x.shape[-1], weight.shape[-1]), x.dtype, x.device,
            )
        with torch.enable_grad():
            xg = x.detach().requires_grad_()
            wg = weight.detach().requires_grad_()
            bg = bias.detach().requires_grad_() if bias is not None else None
            sg = state_in.detach().requires_grad_() if need_state_in else None
            y, fs = ops.causal_conv1d(
                x=xg,
                weight=wg,
                bias=bg,
                activation=ctx.activation,
                initial_state=sg,
                output_final_state=need_state_out,
            )
            outputs = [y] + ([fs] if need_state_out else [])
            grad_outputs = [dy] + ([dht] if need_state_out else [])
            inputs = [xg, wg] + ([bg] if bg is not None else []) + ([sg] if sg is not None else [])
            grads = torch.autograd.grad(outputs, inputs, grad_outputs, allow_unused=True)
        dx, dw = grads[0], grads[1]
        db = grads[2] if bg is not None else None
        ds = grads[-1] if sg is not None else None
        if need_state_in:
            _send_state(sp_group, sp_rank - 1, ds, x.dtype)
        return dx, dw, db, None, None, None, None, None


class ChunkKDACPFunction(torch.autograd.Function):
    """Chunked KDA with CP recurrent-state exchange (native AscendC chain).

    Mirrors ``ChunkKDAFunction`` in ``torchturbo.ops.providers.npu.kda`` but
    threads the recurrent state across ranks: ``initial_state`` in forward,
    and ``dht``/``dh0`` through ``npu_chunk_gated_delta_rule_bwd_dhu`` in
    backward. Token tensors arrive head-major ([B,H,T,D]); gate/beta arrive
    token-major and are transposed inside, matching the stock provider.
    """

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
        lower_bound: float | None,
        cu_seqlens_host: list[int] | None,
        chunk_indices: list[int] | None,
        need_state_in: bool,
        need_state_out: bool,
        sp_group: dist.ProcessGroup,
        sp_rank: int,
    ) -> torch.Tensor:
        from fla_npu.ops.ascendc import (
            npu_chunk_gated_delta_rule_fwd_h,
            npu_chunk_kda_fwd_finalize,
            npu_chunk_kda_fwd_prepare,
        )

        packed = cu_seqlens_host is not None
        layout = "NTD" if packed else "BNSD"
        output_layout = "TND" if packed else "BSND"
        ctx.g_dtype, ctx.beta_dtype = g.dtype, beta.dtype
        g = g.transpose(1, 2)
        beta = beta.transpose(1, 2)
        if packed:
            g, beta = g.squeeze(0), beta.squeeze(0)
        g, beta = g.contiguous(), beta.contiguous()

        num_heads = q.shape[1] if not packed else q.shape[0]
        key_dim, value_dim = q.shape[-1], v.shape[-1]
        num_seq = q.shape[0] if not packed else 1
        state_in = None
        if need_state_in:
            state_in = _recv_state(
                sp_group, sp_rank - 1,
                (num_seq, num_heads, key_dim, value_dim), torch.float32, q.device,
            )
        prep = npu_chunk_kda_fwd_prepare(
            q, k, v, g, beta, scale, 64,
            layout=layout, cu_seqlens=cu_seqlens_host, chunk_indices=chunk_indices,
            safe_gate=True, lower_bound=lower_bound, use_gate_in_kernel=True,
            A_log=A_log, dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
            use_exp2=True, backward_mode="save",
        )
        gk, aqk, akk, w, u, qg, kg, qg_scaled = prep[:8]
        q_hat, k_hat, q_rstd, k_rstd, beta_eff = prep[8:13]
        h, v_new, final_state = npu_chunk_gated_delta_rule_fwd_h(
            kg, w, u, gk=gk, initial_state=state_in,
            output_final_state=need_state_out,
            cu_seqlens=cu_seqlens_host, chunk_indices=chunk_indices,
        )
        output = npu_chunk_kda_fwd_finalize(
            qg_scaled, aqk, v_new, h, output_layout=output_layout,
            cu_seqlens=cu_seqlens_host, chunk_indices=chunk_indices,
        )
        if need_state_out:
            _send_state(sp_group, sp_rank + 1, final_state, torch.float32)
        ctx.save_for_backward(
            q_hat, k_hat, v, g, beta_eff, A_log, dt_bias, gk, aqk, akk, w, qg,
            kg, v_new, h, q_rstd, k_rstd, state_in,
        )
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.cu_seqlens = cu_seqlens_host
        ctx.chunk_indices = chunk_indices
        ctx.flags = (need_state_in, need_state_out)
        ctx.comm = (sp_group, sp_rank)
        ctx.set_materialize_grads(False)
        if packed:
            output = output.unsqueeze(0)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        from itertools import pairwise

        from fla_npu.ops.ascendc import (
            npu_chunk_gated_delta_rule_bwd_dhu,
            npu_chunk_kda_bwd_finalize,
            npu_chunk_kda_bwd_prepare,
        )

        (
            q, k, v, g, beta, A_log, dt_bias, gk, aqk, akk, w, qg, kg, v_new,
            h, q_rstd, k_rstd, state_in,
        ) = ctx.saved_tensors
        need_state_in, need_state_out = ctx.flags
        sp_group, sp_rank = ctx.comm
        packed = ctx.cu_seqlens is not None
        if packed:
            grad_output = grad_output.squeeze(0)
        token_axis = 0 if packed else 1
        grad_output = grad_output.transpose(token_axis, token_axis + 1).contiguous()

        cu, indices = ctx.cu_seqlens, ctx.chunk_indices
        if packed and any(left == right for left, right in pairwise(cu)):
            cu = [cu[0]] + [right for left, right in pairwise(cu) if right > left]
            from torchturbo.ops.providers.npu.kda import _chunk_indices

            indices = _chunk_indices(cu)
        layout = "NTD" if packed else "BNSD"

        dht = None
        if need_state_out:
            num_seq = 1 if packed else q.shape[0]
            dht = _recv_state(
                sp_group, sp_rank + 1,
                (num_seq, q.shape[-3], q.shape[-1], v.shape[-1]),
                q.dtype, q.device,
            )

        d_aqk, dv_local, dq_raw = npu_chunk_kda_bwd_prepare(
            aqk, v_new, grad_output, h, scale=ctx.scale, layout=layout,
            chunk_size=64, cu_seqlens=cu, chunk_indices=indices,
        )
        # Dhu requires rank-4 token tensors even for a packed sequence.
        scan_inputs = (qg, kg, w, grad_output, dv_local, gk)
        if packed:
            scan_inputs = tuple(t.unsqueeze(0) for t in scan_inputs)
        qg_scan, kg_scan, w_scan, do_scan, dv_local_scan, gk_scan = scan_inputs
        h0 = state_in.to(q.dtype) if state_in is not None else None
        dh, dh0, dv_scan = npu_chunk_gated_delta_rule_bwd_dhu(
            qg_scan, kg_scan, w_scan, do_scan, dv_local_scan, ctx.scale, 64,
            gK=gk_scan, h0=h0, dht=dht,
            cu_seqlens=cu, chunk_indices=indices, use_exp2=True,
        )
        if packed:
            dh, dv_scan = dh.squeeze(0), dv_scan.squeeze(0)
        if need_state_in:
            _send_state(sp_group, sp_rank - 1, dh0, q.dtype)

        if dt_bias is None:
            bias = torch.zeros((q.size(-3), q.size(-1)), dtype=torch.float32, device=q.device)
        else:
            bias = dt_bias.reshape(q.size(-3), q.size(-1))
        dq, dk, dv, dbeta, dg, dA, dbias = npu_chunk_kda_bwd_finalize(
            q, k, v, gk, g, beta.float(), A_log, bias, akk, v_new, h, dh,
            dv_scan, d_aqk, dq_raw,
            scale=ctx.scale, lower_bound=ctx.lower_bound, chunk_size=64,
            safe_gate=True, use_gate_in_kernel=True,
            use_exp2=True, q_rstd=q_rstd, k_rstd=k_rstd,
            cu_seqlens=cu, chunk_indices=indices, output_layout=layout,
            use_beta_sigmoid_in_kernel=True,
        )
        dbias = None if dt_bias is None else dbias.reshape_as(dt_bias)
        # Finalize returns head-major gradients; gate and beta are token-major.
        dg = dg.transpose(token_axis, token_axis + 1)
        dbeta = dbeta.transpose(token_axis, token_axis + 1)
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
        )


def kda_cp_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    scale: float | None,
    lower_bound: float | None,
    cu_seqlens: torch.Tensor | None,
    sp_group: dist.ProcessGroup,
    sp_rank: int,
    sp_size: int,
) -> torch.Tensor:
    """CP KDA on local [B,T,C] shards; returns the local head-major output."""
    packed = cu_seqlens is not None
    cu_host = indices = None
    if packed:
        from torchturbo.ops.providers.npu.kda import _prepare_kda_metadata

        cu_host, indices = _prepare_kda_metadata(cu_seqlens, cu_seqlens._version)
    local_tokens = q.shape[1]
    need_in, need_out = _shard_state_flags(cu_host, sp_rank, sp_size, local_tokens)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if lower_bound is None:
        lower_bound = -5.0
    inputs = []
    for tensor in (q, k, v):
        tensor = tensor.transpose(1, 2)
        if packed:
            tensor = tensor.squeeze(0)
        inputs.append(tensor.contiguous())
    if A_log is not None:
        A_log = A_log.float()
    if dt_bias is not None:
        dt_bias = dt_bias.float()
    return ChunkKDACPFunction.apply(
        *inputs,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        lower_bound,
        cu_host,
        indices,
        need_in,
        need_out,
        sp_group,
        sp_rank,
    )
