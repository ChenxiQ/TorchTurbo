"""Native (fla) provider for the chunk gated-delta-rule op.

This is the reference implementation dense Qwen3.5 has always used on non-NPU
hosts (and as the fp32-correctness baseline). ``devices=()`` makes it a candidate
on every runtime; on NPU the AscendC custom provider (priority=100) is preferred
over this one (priority=0), so `auto` policy still routes NPU → AscendC.
"""

from __future__ import annotations

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="gated_delta_rule",
    impl_name="fla_gated_delta_rule",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class FlaGatedDeltaRule(OpImpl):
    def available(self) -> bool:
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(
        self,
        q,
        k,
        v,
        g,
        beta,
        *,
        initial_state=None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        packed_params=None,
        **kwargs,
    ):
        # fla's chunk_gated_delta_rule is an autograd function (head_first=False
        # convention); signature mirrors ascendc_chunk_gated_delta_rule.
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        # ``packed_params`` is the provider API. Keep accepting the legacy
        # dispatcher keyword during the transition.
        cu_seqlens = kwargs.pop("cu_seqlens", None)
        if packed_params is not None:
            if cu_seqlens is not None:
                raise ValueError("pass either cu_seqlens or packed_params, not both")
            cu_seqlens = packed_params.cu_seqlens
        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
        )
