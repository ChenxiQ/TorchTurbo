"""NPU provider for the CANN TND sparse_flash_mla op."""

from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op

_ORI_KV_TAIL_GUARD = 128


def _load_ops():
    """Load CANN ops lazily."""
    import cann_ops_transformer

    return cann_ops_transformer


def _has_required_cann_ops() -> bool:
    try:
        ops = _load_ops()
        return (
            callable(ops.sparse_flash_mla)
            and callable(ops.sparse_flash_mla_metadata)
            and callable(ops.sparse_flash_mla_grad)
            and callable(ops.sparse_flash_mla_grad_metadata)
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return False


def _max_seqlen(cu_seqlens: torch.Tensor | None) -> int:
    if cu_seqlens is None or cu_seqlens.numel() <= 1:
        return 0
    return int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())


# CANN exposes separate forward, metadata, and gradient kernels. Keep their
# coordination in one autograd bridge so the dispatcher exposes one operation.
class SparseFlashMla(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        sinks,
        softmax_scale,
        cmp_ratio,
        cu_seq_lens_q,
        cu_seq_lens_ori_kv,
        cu_seq_lens_cmp_kv,
        cmp_residual_kv,
        ori_win_left,
        ori_win_right,
    ):
        ops = _load_ops()
        has_cmp = cmp_kv is not None
        runtime = {
            "cu_seqlens_q": cu_seq_lens_q,
            "cu_seqlens_ori_kv": cu_seq_lens_ori_kv,
            "cu_seqlens_cmp_kv": cu_seq_lens_cmp_kv if has_cmp else None,
            "cmp_residual_kv": cmp_residual_kv if has_cmp else None,
            "cmp_ratio": cmp_ratio if has_cmp else 1,
            "ori_mask_mode": 4,
            # CANN 9.1 requires cmp_mask_mode=3 on A2/A3.
            "cmp_mask_mode": 3,
            "ori_win_left": ori_win_left,
            "ori_win_right": ori_win_right,
            "layout_q": "TND",
            "layout_kv": "TND",
        }
        meta_only = {
            "max_seqlen_q": _max_seqlen(cu_seq_lens_q),
            "max_seqlen_ori_kv": _max_seqlen(cu_seq_lens_ori_kv),
            "max_seqlen_cmp_kv": _max_seqlen(cu_seq_lens_cmp_kv) if has_cmp else 0,
            "ori_topk": 0,
            "cmp_topk": cmp_sparse_indices.size(-1)
            if cmp_sparse_indices is not None
            else 0,
            "has_ori_kv": True,
            "has_cmp_kv": has_cmp,
        }
        # TND batch size comes from cu_seqlens.
        metadata = ops.sparse_flash_mla_metadata(
            query.size(1), 1, query.size(2), **meta_only, **runtime
        )
        out, lse = ops.sparse_flash_mla(
            query,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            cmp_sparse_indices=cmp_sparse_indices,
            sinks=sinks,
            metadata=metadata,
            softmax_scale=softmax_scale,
            return_softmax_lse=True,
            **runtime,
        )
        ctx.save_for_backward(
            query,
            ori_kv,
            cmp_kv,
            out,
            lse,
            cmp_sparse_indices,
            cu_seq_lens_q,
            cu_seq_lens_ori_kv,
            cu_seq_lens_cmp_kv,
            cmp_residual_kv,
            sinks,
        )
        ctx.softmax_scale = softmax_scale
        ctx.runtime = runtime
        ctx.meta_only = meta_only
        return out

    @staticmethod
    def backward(ctx, grad_output):
        ops = _load_ops()
        (
            query,
            ori_kv,
            cmp_kv,
            out,
            lse,
            cmp_sparse_indices,
            _cu_seq_lens_q,
            _cu_seq_lens_ori_kv,
            _cu_seq_lens_cmp_kv,
            _cmp_residual_kv,
            sinks,
        ) = ctx.saved_tensors
        ori_kv_rows = ori_kv.size(0)
        # Pad a zero 128-row tail for the grad kernel without changing cu_seqlens.
        if ori_kv_rows:
            ori_kv_for_grad = torch.cat(
                [ori_kv, ori_kv.new_zeros((_ORI_KV_TAIL_GUARD, *ori_kv.shape[1:]))],
                dim=0,
            )
        else:
            ori_kv_for_grad = ori_kv
        grad_meta = ops.sparse_flash_mla_grad_metadata(
            query.size(1), 1, query.size(2), **ctx.meta_only, **ctx.runtime
        )
        dq, dori_kv, dcmp_kv, dsinks, _, _ = ops.sparse_flash_mla_grad(
            query,
            grad_output,
            out,
            lse,
            ori_kv=ori_kv_for_grad,
            cmp_kv=cmp_kv,
            cmp_sparse_indices=cmp_sparse_indices,
            sinks=sinks,
            metadata=grad_meta,
            softmax_scale=ctx.softmax_scale,
            **ctx.runtime,
        )
        dori_kv = dori_kv[:ori_kv_rows]
        # Autograd needs None when compressed KV is absent.
        if cmp_kv is None:
            dcmp_kv = None
        return (
            dq,
            dori_kv,
            dcmp_kv,
            None,
            dsinks,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _run_sparse_flash_mla(
    query: torch.Tensor,
    ori_kv: torch.Tensor | None,
    cmp_kv: torch.Tensor | None,
    cmp_sparse_indices: torch.Tensor | None,
    cmp_residual_kv: torch.Tensor | None,
    sinks: torch.Tensor,
    softmax_scale: float,
    cmp_ratio: int,
    ori_win_left: int,
    ori_win_right: int,
    *,
    cu_seq_lens_q: torch.Tensor | None = None,
    cu_seq_lens_ori_kv: torch.Tensor | None = None,
    cu_seq_lens_cmp_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the CANN TND sparse_flash_mla op with autograd."""
    query = query.contiguous()
    ori_kv = ori_kv.contiguous() if ori_kv is not None else None
    cmp_kv = cmp_kv.contiguous() if cmp_kv is not None else None
    if cmp_sparse_indices is not None:
        cmp_sparse_indices = cmp_sparse_indices.contiguous()
    if cmp_residual_kv is not None:
        cmp_residual_kv = cmp_residual_kv.to(torch.int32).contiguous()
    return SparseFlashMla.apply(
        query,
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        sinks,
        softmax_scale,
        cmp_ratio,
        cu_seq_lens_q,
        cu_seq_lens_ori_kv,
        cu_seq_lens_cmp_kv,
        cmp_residual_kv,
        ori_win_left,
        ori_win_right,
    )


@register_op(
    op_name="sparse_flash_mla",
    impl_name="npu_sparse_flash_mla",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuSparseFlashMla(OpImpl):
    def available(self) -> bool:
        return _has_required_cann_ops()

    def __call__(
        self,
        query: torch.Tensor,
        ori_kv: torch.Tensor | None,
        cmp_kv: torch.Tensor | None,
        cmp_sparse_indices: torch.Tensor | None,
        cmp_residual_kv: torch.Tensor | None,
        sinks: torch.Tensor,
        softmax_scale: float,
        cmp_ratio: int,
        ori_win_left: int,
        ori_win_right: int,
        *,
        cu_seq_lens_q: torch.Tensor | None = None,
        cu_seq_lens_ori_kv: torch.Tensor | None = None,
        cu_seq_lens_cmp_kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _run_sparse_flash_mla(
            query,
            ori_kv,
            cmp_kv,
            cmp_sparse_indices,
            cmp_residual_kv,
            sinks,
            softmax_scale,
            cmp_ratio,
            ori_win_left,
            ori_win_right,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_ori_kv=cu_seq_lens_ori_kv,
            cu_seq_lens_cmp_kv=cu_seq_lens_cmp_kv,
        )
