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


def patch_kda_forward(
    attention_cls: type[nn.Module],
    sp_group: ProcessGroup | None,
    sp_size: int,
    sp_rank: int,
    use_fla_kcp: bool = False,
    use_fla_kcp_ascend: bool = False,
) -> None:
    """Install packed KDA training with Ulysses or FLA context parallelism.

    The wrapped forward receives global packed boundaries as the
    ``cu_seqlens`` keyword; convolutions and KDA share them while their
    parameters are sliced by head for sequence parallelism. With sequence
    parallelism all exchanges share one side stream: each QKV gather
    launches right after its projection and records an event on that
    stream, so its convolution waits only on that gather and overlaps the
    exchanges still in flight.

    With use_fla_kcp_ascend=True, convolution keeps this layout while
    KDA runs on local tokens and all heads after an inverse exchange.
    use_fla_kcp selects FLA's local convolution and Triton KDA CP path.
    """
    from fla.modules.fused_norm_gate import rms_norm_gated

    from torchturbo.models.modules.attention.ulysses.ulysses import (
        gather_heads_scatter_seq,
        gather_seq_scatter_heads,
    )
    from torchturbo.ops import ops
    from torchturbo.ops.packed_params import get_packed_params

    if use_fla_kcp and use_fla_kcp_ascend:
        raise ValueError(
            "use_fla_kcp and use_fla_kcp_ascend select different "
            "CP backends; enable only one."
        )
    use_ascendc_cp = use_fla_kcp_ascend and sp_size > 1
    if use_ascendc_cp:
        from torchturbo.ops.providers.npu._fla_npu_kda import KDAContextParallel

    a2a_stream = None
    qkv_events = None
    if sp_size > 1 and not use_fla_kcp:
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

        kda_kwargs = {}
        if use_ascendc_cp:
            if self.num_heads % sp_size:
                raise ValueError("KDA CP's convolution bridge requires heads divisible by SP size.")
            offsets = (packed_params.cu_seqlens_list if packed_params is not None
                       else [0, q_len * sp_size])
            cp_context = KDAContextParallel.from_global_offsets(offsets, q_len, sp_group)
            cu_seqlens = torch.tensor(cp_context.cu_seqlens, dtype=torch.int64)
            kda_kwargs.update(
                impl="ascendc_kda", cp_context=cp_context, cu_seqlens_cpu=cu_seqlens,
            )

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
        # KDA owns the sigmoid (in-kernel on AscendC, in the native adapter
        # for triton), so hand it raw logits.
        beta = self.b_proj(hidden_states).float()
        if sp_size > 1 and not use_ascendc_cp:
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
        # ``head_num`` asks for the head-first [B, H, T, Dh] output the fused
        # KDA operator consumes; the depthwise conv makes the regrouping exact.
        q_heads = q.shape[-1] // self.head_k_dim
        k_heads = k.shape[-1] // self.head_k_dim
        v_heads = v.shape[-1] // self.head_dim
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
            head_num=q_heads,
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
            head_num=k_heads,
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
            head_num=v_heads,
        )

        if use_ascendc_cp:
            # Convolution has handled global packed boundaries. Restore all
            # heads for this rank's contiguous sequence shard before KDA.
            q, k, v = (
                gather_heads_scatter_seq(
                    tensor, seq_dim=2, head_dim=1, sp_group=sp_group, sp_size=sp_size,
                )
                for tensor in (q, k, v)
            )

        # q/k/v already come back head-first; only the gate needs the split.
        g = g.unflatten(-1, (-1, self.head_dim))

        if sp_size > 1 and not use_ascendc_cp:
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

        # ``disable_recompute=True`` is required by the fused AscendC
        # backward (no recompute path on A2/A3); the native provider strips
        # it, so both backends run in the same memory regime.
        o, _ = ops.kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log_local,
            dt_bias=dt_bias_local,
            initial_state=None,
            output_final_state=not use_ascendc_cp,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=self.safe_gate,
            lower_bound=self.lower_bound,
            cu_seqlens=cu_seqlens,
            disable_recompute=True,
            head_major_qkv=True,
            **kda_kwargs,
        )

        if sp_size > 1 and not use_ascendc_cp:
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

    def kcp_forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, None]:
        from fla.ops.cp import build_cp_context

        batch_size, q_len, _ = hidden_states.shape
        assert past_key_value is None, "KDA training does not accept a KV cache."
        assert self.mode == "chunk", "Only chunk mode is supported in training."

        if batch_size != 1:
            raise ValueError(
                f"FLA KCP requires batch_size=1, got {batch_size}. "
                "Use a per-device batch size of 1 or disable use_fla_kcp."
            )

        # Use global boundaries and isolate any padded tail as a sequence.
        cu_seqlens = kwargs.get("cu_seqlens")
        packed_params = get_packed_params(cu_seqlens, q_len * sp_size)
        if packed_params is not None:
            cu_seqlens = packed_params.cu_seqlens
        else:
            cu_seqlens = torch.tensor(
                [0, q_len * sp_size],
                dtype=torch.int32,
                device=hidden_states.device,
            )
        cp_context = build_cp_context(
            cu_seqlens,
            group=sp_group,
            conv1d_kernel_size=self.conv_size,
        )

        q = self.q_proj(hidden_states)   # [1, T/sp, H*K]
        k = self.k_proj(hidden_states)   # [1, T/sp, H*K]
        v = self.v_proj(hidden_states)   # [1, T/sp, H*V]

        # Dispatch through the ops layer; cp_context selects FLA's
        # Triton CP path instead of AscendC. d,1,w -> d,w.
        q, _ = ops.causal_conv1d(
            x=q,
            weight=self.q_conv1d.weight.squeeze(1),
            bias=self.q_conv1d.bias,
            activation=self.q_conv1d.activation,
            initial_state=None,
            output_final_state=False,
            cp_context=cp_context,
        )
        k, _ = ops.causal_conv1d(
            x=k,
            weight=self.k_conv1d.weight.squeeze(1),
            bias=self.k_conv1d.bias,
            activation=self.k_conv1d.activation,
            initial_state=None,
            output_final_state=False,
            cp_context=cp_context,
        )
        v, _ = ops.causal_conv1d(
            x=v,
            weight=self.v_conv1d.weight.squeeze(1),
            bias=self.v_conv1d.bias,
            activation=self.v_conv1d.activation,
            initial_state=None,
            output_final_state=False,
            cp_context=cp_context,
        )

        if self.no_kda_lora:
            g = self.f_proj(hidden_states)
        else:
            g = self.f_b_proj(self.f_a_proj(hidden_states))

        beta = self.b_proj(hidden_states).float()

        q = q.unflatten(-1, (-1, self.head_k_dim))
        k = k.unflatten(-1, (-1, self.head_k_dim))
        v = v.unflatten(-1, (-1, self.head_dim))
        g = g.unflatten(-1, (-1, self.head_dim))

        # Full A_log/dt_bias under KCP (no slicing).
        o, _ = ops.kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=self.safe_gate,
            lower_bound=self.lower_bound,
            cp_context=cp_context,
            impl="fla_kda",
        )

        # o_norm uses g recomputed from hidden_states.
        if self.no_kda_lora:
            g = self.g_proj(hidden_states)
        else:
            g = self.g_b_proj(self.g_a_proj(hidden_states))
        g = g.unflatten(-1, (-1, self.head_dim))

        self.o_norm.forward = norm_forward.__get__(self.o_norm)
        o = self.o_norm(o, g)

        o = o.flatten(-2)
        o = self.o_proj(o)
        return o, None, None

    attention_cls.forward = kcp_forward if use_fla_kcp else sp_forward
