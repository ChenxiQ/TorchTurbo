"""AllGather state exchange for equal, contiguous KDA sequence shards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class KDAContextParallel:
    """Local packed boundaries and ownership within an explicit CP group."""

    group: dist.ProcessGroup
    rank: int
    size: int
    cu_seqlens: tuple[int, ...]
    prefix: int
    suffix: int

    @classmethod
    def from_global_offsets(
        cls,
        offsets: list[int],
        local_tokens: int,
        group: dist.ProcessGroup,
    ) -> KDAContextParallel:
        """Intersect global packed sequences with this group's token shard."""
        if group is None:
            raise ValueError("KDA CP requires an explicit sequence-parallel group.")
        rank, size = dist.get_rank(group), dist.get_world_size(group)
        if local_tokens <= 0 or offsets[0] != 0 or offsets[-1] != local_tokens * size:
            raise ValueError("KDA CP offsets must cover all equal, nonempty token shards.")
        if any(b < a for a, b in zip(offsets, offsets[1:])):
            raise ValueError("KDA CP offsets must be nondecreasing.")
        lo, hi = rank * local_tokens, (rank + 1) * local_tokens
        segments = [(a, b) for a, b in zip(offsets, offsets[1:])
                    if a < hi and b > lo and b > a]
        local_cu = (0, *(min(b, hi) - lo for _, b in segments))
        first_rank = segments[0][0] // local_tokens
        last_rank = (segments[-1][1] - 1) // local_tokens
        return cls(group, rank, size, local_cu, rank - first_rank, last_rank - rank)

    def exchange(self, summary: torch.Tensor, *, forward: bool, dtype: torch.dtype) -> torch.Tensor:
        """Merge only same-sequence predecessors or successors into a boundary."""
        from fla_npu.ops.ascendc import merge_fwd_bwd_kernel

        gathered = [torch.empty_like(summary) for _ in range(self.size)]
        dist.all_gather(gathered, summary.contiguous(), group=self.group)
        states = summary.new_zeros((len(self.cu_seqlens) - 1, summary.shape[0], 128, 128))
        count = self.prefix if forward else self.suffix
        if count:
            merge_fwd_bwd_kernel(
                states[0 if forward else -1], torch.stack(gathered), count,
                self.rank, forward=forward, state_v_first=False,
            )
        return states.to(dtype=dtype)


def cp_forward(
    context: KDAContextParallel,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    **gate_kwargs: Any,
) -> tuple:
    """Run fused Prepare, boundary exchange, FwdH and BSND Finalize.

    Inputs are packed head-major rank-3 tensors. Rank-4 views bridge the CP
    kernels; the caller reuses the single-rank adapter's saved-tensor handling.
    """
    from fla_npu.ops.ascendc import (
        chunk_fwd_h,
        chunk_kda_fwd_finalize,
        chunk_kda_fwd_prepare,
        pre_process_fwd_kernel_merged,
    )

    cu = list(context.cu_seqlens)
    q, k, v, g, beta = (t.unsqueeze(0) for t in (q, k, v, g, beta))
    saved = chunk_kda_fwd_prepare(
        q, k, v, g, beta, scale, layout="BNSD", cu_seqlens=cu,
        use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
        use_gate_in_kernel=True, epsilon=1e-6, use_exp2=True,
        backward_mode="save", **gate_kwargs,
    )
    gk, aqk, akk, w, u, qg, kg, qg_scaled, q_hat, k_hat, qr, kr, beta_eff = saved
    if context.suffix:
        summary = pre_process_fwd_kernel_merged(
            kg, w, u, gk=gk, cu_seqlens=cu[-2:], chunk_size=64,
        )[0]
    else:
        summary = q.new_zeros((q.shape[1], 128, 256), dtype=torch.float32)
    # FwdH selects its recurrent-state precision from initial_state.dtype.
    # Preserve FP32, matching the no-initial-state path, even for zero h0.
    h0 = context.exchange(summary, forward=True, dtype=torch.float32)
    h, v_new, _ = chunk_fwd_h(
        kg, w, u, gk=gk, initial_state=h0, cu_seqlens=cu,
        save_new_value=True, output_final_state=False, use_exp2=True,
    )
    o = chunk_kda_fwd_finalize(
        qg_scaled, aqk, v_new, h, cu_seqlens=cu, output_layout="BSND",
    )
    outputs = (o, None, gk, aqk, akk, w, u, qg, kg, v_new, h, h0)
    return outputs, q_hat, k_hat, qr, kr, beta_eff


def cp_backward_boundary(
    context: KDAContextParallel,
    qg: torch.Tensor,
    kg: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    gk: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Build the last local sequence's dht from same-sequence successors."""
    from fla_npu.ops.ascendc import pre_process_bwd_kernel_merged

    if context.prefix:
        # CP backward consumes unscaled qg and a low-precision gate prefix.
        summary = pre_process_bwd_kernel_merged(
            qg, kg, w, do, dv, scale, 64, gk=gk.to(qg.dtype),
            cu_seqlens=list(context.cu_seqlens[:2]),
        )
    else:
        summary = qg.new_zeros((qg.shape[1], 128, 256), dtype=torch.float32)
    # Unlike FwdH's state, the Dhu ABI requires dht to have q's low-precision dtype.
    return context.exchange(summary, forward=False, dtype=qg.dtype)
