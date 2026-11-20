"""Shared training attention adapters for Bailing text and vision-language models."""

from typing import Any

import torch
from torch import nn
from torch.distributed import ProcessGroup

from torchturbo.runtime import get_device_runtime

def patch_rms_norm(norm_cls: type[nn.Module]) -> None:
    """Route the supplied RMSNorm class through the device operator dispatcher."""
    from torchturbo.ops import ops

    def rms_norm_forward(
        module: nn.Module, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return ops.rms_norm(hidden_states, module.weight, module.variance_epsilon)

    norm_cls.forward = rms_norm_forward


_KDA_SP_MODE = "ulysses"
_KDA_CP_BACKEND = "auto"


def set_kda_sp_mode(mode: str) -> None:
    """Select the KDA sequence-parallel strategy: 'ulysses' (default) or 'cp'."""
    global _KDA_SP_MODE
    if mode not in ("ulysses", "cp"):
        raise ValueError(f"Unknown kda_sp_mode {mode!r}; expected 'ulysses' or 'cp'.")
    _KDA_SP_MODE = mode


def set_kda_cp_backend(backend: str) -> None:
    """Select the CP operator backend: 'auto', 'ascendc' (fla_npu) or 'fla'."""
    global _KDA_CP_BACKEND
    if backend not in ("auto", "ascendc", "fla"):
        raise ValueError(
            f"Unknown kda_cp_backend {backend!r}; expected 'auto', 'ascendc' or 'fla'."
        )
    _KDA_CP_BACKEND = backend


def _select_cp_backend() -> str:
    """Resolve 'auto': native fla_npu AscendC operators when importable, else fla."""
    if _KDA_CP_BACKEND != "auto":
        return _KDA_CP_BACKEND
    try:
        from fla_npu.ops.ascendc import (  # noqa: F401
            chunk_kda_fwd,
            npu_chunk_gated_delta_rule_bwd_dhu,
            npu_chunk_kda_bwd_finalize,
            npu_chunk_kda_bwd_prepare,
        )
        from torchturbo.ops.providers.npu._fla_gdn import (  # noqa: F401
            ascendc_causal_conv1d,
        )
    except Exception:
        return "fla"
    return "ascendc"


def patch_kda_forward(
    attention_cls: type[nn.Module],
    sp_group: ProcessGroup | None,
    sp_size: int,
    sp_rank: int,
) -> None:
    """Install packed KDA training with sequence gathering and head sharding.

    The wrapped forward receives global packed boundaries as the
    ``cu_seqlens`` keyword; convolutions and KDA share them while their
    parameters are sliced by head for sequence parallelism. With sequence
    parallelism all exchanges share one side stream: each QKV gather
    launches right after its projection and records an event on that
    stream, so its convolution waits only on that gather and overlaps the
    exchanges still in flight.
    """
    from fla.modules.fused_norm_gate import rms_norm_gated

    from torchturbo.models.modules.attention.ulysses.ulysses import (
        gather_heads_scatter_seq,
        gather_seq_scatter_heads,
    )
    from torchturbo.ops import ops
    from torchturbo.ops.packed_params import get_packed_params

    a2a_stream = None
    qkv_events = None
    if sp_size > 1:
        device_runtime = get_device_runtime()
        a2a_stream = device_runtime.new_stream()
        qkv_events = [device_runtime.new_event() for _ in range(3)]

    def get_parameter_local_cp(weight: torch.Tensor, dim: int) -> torch.Tensor:
        if sp_size == 1:
            return weight

        slices = [slice(None)] * weight.dim()
        dim_size = weight.size(dim=dim)
        slices[dim] = slice(
            sp_rank * dim_size // sp_size, (sp_rank + 1) * dim_size // sp_size
        )
        return weight[tuple(slices)]

    def norm_forward(
        module: nn.Module,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return rms_norm_gated(
            x,
            g,
            module.weight,
            None,
            module.activation,
            residual=residual,
            eps=module.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )

    def cp_forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, None]:
        """KDA forward with operator-level context parallelism (fla cp_context).

        The sequence stays sharded across SP ranks; only tiny recurrent states
        are exchanged instead of the Ulysses all-to-all activation gather/scatter.
        """
        from fla.modules.causal_conv1d import causal_conv1d as fla_causal_conv1d
        from fla.ops.cp import build_cp_context
        from fla.ops.kda import chunk_kda as fla_chunk_kda

        assert past_key_value is None, "KDA training does not accept a KV cache."
        assert self.mode == "chunk", "Only chunk mode is supported in training."

        batch_size, q_len, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens")
        packed_params = get_packed_params(cu_seqlens, q_len * sp_size)
        if packed_params is not None:
            cu_seqlens = packed_params.cu_seqlens
        if cu_seqlens is None:
            cu_seqlens = torch.tensor(
                [0, q_len * sp_size], dtype=torch.long, device=hidden_states.device
            )
        cp_ctx = build_cp_context(
            cu_seqlens, group=sp_group, conv1d_kernel_size=self.conv_size
        )

        def conv(x: torch.Tensor, conv_mod: nn.Module) -> torch.Tensor:
            return fla_causal_conv1d(
                x=x,
                weight=conv_mod.weight.squeeze(1),
                bias=conv_mod.bias,
                activation=conv_mod.activation,
                cp_context=cp_ctx,
            )[0]

        q = conv(self.q_proj(hidden_states), self.q_conv1d)
        k = conv(self.k_proj(hidden_states), self.k_conv1d)
        v = conv(self.v_proj(hidden_states), self.v_conv1d)
        if self.no_kda_lora:
            g = self.f_proj(hidden_states)
        else:
            g = self.f_b_proj(self.f_a_proj(hidden_states))
        beta = self.b_proj(hidden_states).float().sigmoid()

        q = q.unflatten(-1, (-1, self.head_k_dim))
        k = k.unflatten(-1, (-1, self.head_k_dim))
        v = v.unflatten(-1, (-1, self.head_dim))
        g = g.unflatten(-1, (-1, self.head_dim))

        o, _ = fla_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=self.safe_gate,
            lower_bound=self.lower_bound,
            cp_context=cp_ctx,
        )

        # Output gating uses the original local sequence and all of its heads.
        if self.no_kda_lora:
            g_out = self.g_proj(hidden_states)
        else:
            g_out = self.g_b_proj(self.g_a_proj(hidden_states))
        g_out = g_out.unflatten(-1, (-1, self.head_dim))
        self.o_norm.forward = norm_forward.__get__(self.o_norm)
        o = self.o_norm(o, g_out)
        o = self.o_proj(o.flatten(-2))
        return o, None, None

    def cp_forward_native(
        self: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, None]:
        """KDA forward with operator-level context parallelism (fla_npu AscendC).

        Same CP contract as ``cp_forward`` but runs entirely on native NPU
        operators: the sequence stays sharded, and only the recurrent
        boundary states ([N,H,K,V]) plus the conv1d boundary windows cross
        ranks, replacing the Ulysses all-to-all activation exchanges.
        """
        from torchturbo.models.bailing_hybrid.kda_cp_npu import (
            CausalConv1dCPFunction,
            _shard_state_flags,
            kda_cp_core,
        )

        assert past_key_value is None, "KDA training does not accept a KV cache."
        assert self.mode == "chunk", "Only chunk mode is supported in training."

        batch_size, q_len, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens")
        packed_params = get_packed_params(cu_seqlens, q_len * sp_size)
        if packed_params is not None:
            cu_seqlens = packed_params.cu_seqlens
        cu_host = cu_seqlens.detach().cpu().tolist() if cu_seqlens is not None else None
        need_in, need_out = _shard_state_flags(cu_host, sp_rank, sp_size, q_len)

        def conv(x: torch.Tensor, conv_mod: nn.Module) -> torch.Tensor:
            return CausalConv1dCPFunction.apply(
                x,
                conv_mod.weight.squeeze(1),
                conv_mod.bias,
                conv_mod.activation,
                need_in,
                need_out,
                sp_group,
                sp_rank,
            )

        q = conv(self.q_proj(hidden_states), self.q_conv1d)
        k = conv(self.k_proj(hidden_states), self.k_conv1d)
        v = conv(self.v_proj(hidden_states), self.v_conv1d)
        if self.no_kda_lora:
            g = self.f_proj(hidden_states)
        else:
            g = self.f_b_proj(self.f_a_proj(hidden_states))
        beta = self.b_proj(hidden_states).float()

        q = q.unflatten(-1, (-1, self.head_k_dim))
        k = k.unflatten(-1, (-1, self.head_k_dim))
        v = v.unflatten(-1, (-1, self.head_dim))
        g = g.unflatten(-1, (-1, self.head_dim))

        o = kda_cp_core(
            q,
            k,
            v,
            g,
            beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=None,
            lower_bound=self.lower_bound,
            cu_seqlens=cu_seqlens,
            sp_group=sp_group,
            sp_rank=sp_rank,
            sp_size=sp_size,
        )

        # Output gating uses the original local sequence and all of its heads.
        if self.no_kda_lora:
            g_out = self.g_proj(hidden_states)
        else:
            g_out = self.g_b_proj(self.g_a_proj(hidden_states))
        g_out = g_out.unflatten(-1, (-1, self.head_dim))
        self.o_norm.forward = norm_forward.__get__(self.o_norm)
        o = self.o_norm(o, g_out)
        o = self.o_proj(o.flatten(-2))
        return o, None, None

    def sp_forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, None]:
        assert past_key_value is None, "KDA training does not accept a KV cache."
        assert self.mode == "chunk", "Only chunk mode is supported in training."

        # Global packed boundaries ride with the forward kwargs; the a2a
        # gathers below rebuild the full sequence on each rank. Convolutions
        # consume cached metadata with the padded tail as a final sequence.
        batch_size, q_len, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens")
        packed_params = get_packed_params(cu_seqlens, q_len * sp_size)
        if packed_params is not None:
            cu_seqlens = packed_params.cu_seqlens

        # Each gather launches right after its projection so the exchanges
        # pipeline against the remaining projections and convolutions; the
        # event recorded behind a gather marks its completion, and only its
        # convolution waits on it.
        q = self.q_proj(hidden_states)
        if sp_size > 1:
            q = gather_seq_scatter_heads(
                q, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size,
                a2a_stream=a2a_stream,
            )
            qkv_events[0].record(a2a_stream)
        k = self.k_proj(hidden_states)
        if sp_size > 1:
            k = gather_seq_scatter_heads(
                k, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size,
                a2a_stream=a2a_stream,
            )
            qkv_events[1].record(a2a_stream)
        v = self.v_proj(hidden_states)
        if sp_size > 1:
            v = gather_seq_scatter_heads(
                v, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size,
                a2a_stream=a2a_stream,
            )
            qkv_events[2].record(a2a_stream)

        # Compute g, beta from padded hidden_states (always [B, T/sp, H]
        # format), before the convolutions so they overlap the exchanges
        # still in flight.
        if self.no_kda_lora:
            g = self.f_proj(hidden_states)
        else:
            g = self.f_b_proj(self.f_a_proj(hidden_states))
        beta = self.b_proj(hidden_states).float()
        if sp_size > 1:
            # Launched before the convolutions so these small exchanges
            # finish while the convolutions run; the wait at the chunk
            # consumer below covers them.
            g = gather_seq_scatter_heads(
                g, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size,
                a2a_stream=a2a_stream,
            )
            beta = gather_seq_scatter_heads(
                beta, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size,
                a2a_stream=a2a_stream,
            )

        # Every convolution must reset at the same boundaries as recurrent KDA.
        if sp_size > 1:
            get_device_runtime().current_stream().wait_event(qkv_events[0])
        q, _ = ops.causal_conv1d(
            x=q,
            weight=get_parameter_local_cp(self.q_conv1d.weight, dim=0).squeeze(1),
            bias=get_parameter_local_cp(self.q_conv1d.bias, dim=0)
            if self.q_conv1d.bias is not None
            else None,
            activation=self.q_conv1d.activation,
            initial_state=None,
            output_final_state=False,
            packed_params=packed_params,
        )
        if sp_size > 1:
            get_device_runtime().current_stream().wait_event(qkv_events[1])
        k, _ = ops.causal_conv1d(
            x=k,
            weight=get_parameter_local_cp(self.k_conv1d.weight, dim=0).squeeze(1),
            bias=get_parameter_local_cp(self.k_conv1d.bias, dim=0)
            if self.k_conv1d.bias is not None
            else None,
            activation=self.k_conv1d.activation,
            initial_state=None,
            output_final_state=False,
            packed_params=packed_params,
        )
        if sp_size > 1:
            get_device_runtime().current_stream().wait_event(qkv_events[2])
        v, _ = ops.causal_conv1d(
            x=v,
            weight=get_parameter_local_cp(self.v_conv1d.weight, dim=0).squeeze(1),
            bias=get_parameter_local_cp(self.v_conv1d.bias, dim=0)
            if self.v_conv1d.bias is not None
            else None,
            activation=self.v_conv1d.activation,
            initial_state=None,
            output_final_state=False,
            packed_params=packed_params,
        )

        g = g.unflatten(-1, (-1, self.head_dim))

        if sp_size > 1:
            # Covers the gate/beta gathers trailing the QKV ones on the
            # shared stream.
            get_device_runtime().current_stream().wait_stream(a2a_stream)
            heads_per_rank = self.num_heads // sp_size
            start_head = sp_rank * heads_per_rank
            end_head = start_head + heads_per_rank
            A_log_local = self.A_log[start_head:end_head]
            start_ch = start_head * self.head_dim
            end_ch = end_head * self.head_dim
            dt_bias_local = self.dt_bias[start_ch:end_ch]
        else:
            A_log_local = self.A_log
            dt_bias_local = self.dt_bias

        o, _ = ops.kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log_local,
            dt_bias=dt_bias_local,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=self.safe_gate,
            lower_bound=self.lower_bound,
            cu_seqlens=cu_seqlens,
        )

        if sp_size > 1:
            # o: [B, T, H/sp, V] -> [B, T/sp, H, V]. The scatter runs on
            # the shared a2a stream; the gather-side exchanges return
            # contiguous views, but this scattered layout needs an extra
            # copy that _all_to_all_single keeps on the a2a stream.
            o = gather_heads_scatter_seq(
                o, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size,
                a2a_stream=a2a_stream,
            )

        # Output gating uses the original local sequence and all of its heads.
        if self.no_kda_lora:
            g = self.g_proj(hidden_states)
        else:
            g = self.g_b_proj(self.g_a_proj(hidden_states))
        g = g.unflatten(-1, (-1, self.head_dim))
        if sp_size > 1:
            get_device_runtime().current_stream().wait_stream(a2a_stream)
        self.o_norm.forward = norm_forward.__get__(self.o_norm)
        o = self.o_norm(o, g)
        o = self.o_proj(o.flatten(-2))
        return o, None, None

    if _KDA_SP_MODE == "cp":
        attention_cls.forward = (
            cp_forward_native if _select_cp_backend() == "ascendc" else cp_forward
        )
    else:
        attention_cls.forward = sp_forward
