"""NPU (AscendC) provider for the kda op.

Delegates to the vendored adapters in ``_fla_npu_kda`` -- L2 norm and the
beta sigmoid (with its backward derivative) fused into the AscendC
kernels, backward as the prepare/dhu/finalize composition, built from
fla-npu main @ d77c3738.  priority=100 so `auto` on NPU picks it over the
fla native (priority=0); wheels that miss the fused backward symbols
fail the availability probe and fall back to the native triton provider.
"""

from __future__ import annotations

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


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
            import fla_npu  # noqa: F401
            # Probes the fused backward composition; older wheels ship
            # fwd/bwd only and fail here.
            from fla_npu.ops.ascendc import npu_chunk_kda_bwd  # noqa: F401
            from torchturbo.ops.providers.npu._fla_npu_kda import (  # noqa: F401
                ascendc_chunk_kda,
            )
        except Exception as exc:  # pragma: no cover - diagnostic
            import traceback

            from torchturbo.logging import get_logger

            get_logger(__name__).warning_rank0(
                "ascendc_kda unavailable (explicit selection will "
                "raise; auto falls back to fla triton): %r\n%s",
                exc,
                traceback.format_exc(),
            )
            return False
        return True

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from torchturbo.ops.providers.npu._fla_npu_kda import ascendc_chunk_kda

        # The adapter takes the raw cu_seqlens tensor, and rejects unknown
        # kwargs outright.
        packed_params = kwargs.pop("packed_params", None)
        if packed_params is not None and kwargs.get("cu_seqlens") is None:
            kwargs["cu_seqlens"] = packed_params.cu_seqlens

        return ascendc_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            **kwargs,
        )
