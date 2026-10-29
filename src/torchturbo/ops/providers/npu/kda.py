"""NPU (AscendC) provider for the chunk_kda op.

Full AscendC training mode with fused L2 normalization, beta sigmoid and gate.
Targets FLAN d77c37382aa8cf53e25005615c8c3f761fc137d2 (PR #998 merge).
The vendored adapter uses the fused forward without CP, split forward stages
with AllGather CP, and the shared Prepare/Dhu/Finalize backward.

Opt-in by design: ``available()`` stays False even when fla_npu is
importable, so the default `auto` selection on NPU keeps the fla-org triton
provider. ``TORCHTURBO_KDA_ASCENDC=1`` opts in. Explicit selection and CP
also require this existing switch, because dispatch checks availability.
"""

from __future__ import annotations

import os

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="chunk_kda",
    impl_name="ascendc_chunk_kda",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class AscendCChunkKDA(OpImpl):
    def available(self) -> bool:
        if os.environ.get("TORCHTURBO_KDA_ASCENDC", "0").strip() not in (
            "1", "true", "True",
        ):
            return False
        try:
            import fla_npu  # noqa: F401
            # The full-AscendC path needs BOTH fused ops; older wheels
            # (fwd-only) fail here and fall back to the native fla triton
            # provider under `auto`.
            from fla_npu.ops.ascendc import npu_chunk_kda_bwd  # noqa: F401
            from torchturbo.ops.providers.npu._fla_npu_kda import (  # noqa: F401
                ascendc_chunk_kda,
            )
        except Exception as exc:  # pragma: no cover - diagnostic
            import traceback

            from torchturbo.logging import get_logger

            get_logger(__name__).warning_rank0(
                "ascendc_chunk_kda unavailable (explicit selection will "
                "raise; auto falls back to fla triton): %r\n%s",
                exc,
                traceback.format_exc(),
            )
            return False
        return True

    def __call__(
        self,
        q,
        k,
        v,
        g,
        beta,
        **kwargs,
    ):
        from torchturbo.ops.providers.npu._fla_npu_kda import ascendc_chunk_kda

        return ascendc_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            **kwargs,
        )
