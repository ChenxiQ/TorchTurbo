import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class RMSNormGated(nn.RMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        activation: str = "silu",
    ):
        super().__init__(hidden_size, eps)

        self.hidden_size = hidden_size
        self.eps = eps
        self.activation = activation

        if self.activation not in ("swish", "silu", "sigmoid"):
            raise ValueError(f"Unsupported activation: {self.activation}")

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        s += f", eps={self.eps}"
        s += f", activation={self.activation}"
        s += ")"
        return s

    def forward(self, x: torch.Tensor, g: torch.Tensor):
        if self.activation in ("swish", "silu"):
            act_fn = F.silu
        elif self.activation == "sigmoid":
            act_fn = F.sigmoid
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        y = super().forward(x) * act_fn(g)
        return y

class NaiveShortConvolution(nn.Conv1d):
    """Pure-PyTorch replacement for ShortConvolution (Triton/CUDA causal_conv1d).

    Inherits from nn.Conv1d so state_dict keys (weight, bias) are compatible
    with the original ShortConvolution. Uses F.conv1d for prefill and a
    sliding-window update for decode.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = 'silu',
        **kwargs,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            padding=kernel_size - 1,
            bias=bias,
        )
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        if activation not in (None, 'silu', 'swish'):
            raise NotImplementedError(f"NaiveShortConvolution only supports None/silu/swish, got {activation}")
        self.activation = activation

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation in ('silu', 'swish'):
            return F.silu(x)
        return x

    def _causal_conv1d_prefill(
        self,
        x: torch.Tensor,        # [B, T, D]
        weight: torch.Tensor,    # [D, W]
        bias: torch.Tensor | None,
        initial_state: torch.Tensor | None,  # [B, D, W-1] or None
        output_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prefill path: full sequence causal conv1d using F.conv1d."""
        dtype_in = x.dtype
        B, T, D = x.shape
        W = weight.shape[1]
        x_t = x.transpose(1, 2).to(weight.dtype)  # [B, D, T]

        if initial_state is None:
            out = F.conv1d(x_t, weight.unsqueeze(1), bias, padding=W - 1, groups=D)
        else:
            x_t = torch.cat([initial_state, x_t], dim=-1)
            out = F.conv1d(x_t, weight.unsqueeze(1), bias, padding=0, groups=D)

        out = out[..., :T]  # causal trim
        out = self._apply_activation(out).to(dtype_in)
        out = out.transpose(1, 2)  # [B, T, D]

        final_state = None
        if output_final_state:
            # final_state: last W-1 positions of the input (left-padded with zeros if short)
            # Matches the reference: F.pad(x, (W-1-x.shape[-1], 0))
            final_state = F.pad(x_t, (W - 1 - x_t.shape[-1], 0)).to(dtype_in)  # [B, D, W-1]
            # Format: [B, D, W] with leading zero column for compatibility with ShortConvolution
            final_state = F.pad(final_state, (1, 0))  # [B, D, W]

        return out, final_state

    def _causal_conv1d_step(
        self,
        x: torch.Tensor,        # [N, 1, D] or [N, D]
        cache: torch.Tensor,     # [N, D, W] — updated in-place
        weight: torch.Tensor,    # [D, W]
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Decode path: single-token causal conv1d update."""
        dtype_in = x.dtype
        if x.dim() == 3:
            x = x.squeeze(1)  # [N, D]

        # Roll cache left and write new token at the end
        cache.copy_(cache.roll(shifts=-1, dims=-1))
        cache[:, :, -1] = x.to(cache.dtype)

        # Compute conv output
        out = torch.sum(cache * weight.unsqueeze(0), dim=-1)  # [N, D]
        if bias is not None:
            out = out + bias
        out = self._apply_activation(out.unsqueeze(1)).squeeze(1)  # [N, D]
        return out.to(dtype_in)

    def forward(
        self,
        x: torch.Tensor,                          # [B, T, D]
        residual: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: torch.Tensor | None = None,         # [N, D, W]
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        W = self.kernel_size
        D = self.hidden_size
        weight = self.weight.squeeze(1)  # [D, W] (inherited from nn.Conv1d)
        bias = self.bias

        # Decode path: single token per sequence
        if x.shape[1] == 1 and cache is not None:
            out = self._causal_conv1d_step(x, cache, weight, bias)
            out = out.unsqueeze(1)  # [N, 1, D]
            if residual is not None:
                out = out + residual
            final_state = cache if output_final_state else None
            return out, final_state

        # Prefill path
        if cu_seqlens is not None:
            # Variable-length: loop over sequences
            N = cu_seqlens.shape[0] - 1
            out_list = []
            state_list = []
            for i in range(N):
                s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
                x_i = x[:, s:e]  # [1, T_i, D]
                cache_i = cache[i:i+1, :, 1:] if cache is not None else None  # [1, D, W-1]
                out_i, state_i = self._causal_conv1d_prefill(
                    x_i, weight, bias,
                    initial_state=cache_i,
                    output_final_state=output_final_state,
                )
                out_list.append(out_i)
                if output_final_state:
                    state_list.append(state_i.squeeze(0))
            out = torch.cat(out_list, dim=1)
            final_state = torch.stack(state_list, dim=0) if output_final_state else None
        else:
            # Extract initial_state from cache (skip leading zero column)
            initial_state = cache[:, :, 1:] if cache is not None else None
            out, final_state = self._causal_conv1d_prefill(
                x, weight, bias,
                initial_state=initial_state,
                output_final_state=output_final_state,
            )

        if residual is not None:
            out = out + residual

        return out, final_state


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape ``[B, T, H, K]``.
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]``.
        v (torch.Tensor):
            Values of shape ``[B, T, HV, V]``. ``HV`` must be divisible by ``H``.
        g (torch.Tensor):
            Per-dimension decay gates (log-space) of shape ``[B, T, HV, K]``.
        beta (torch.Tensor):
            Beta scalars of shape ``[B, T, HV]``.
        scale (Optional[float]):
            Scale factor. Defaults to ``1 / sqrt(K)``.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape ``[B, HV, K, V]``.
        output_final_state (bool):
            Whether to return the final state.

    Returns:
        A tuple ``(o, S)`` where ``o`` has shape ``[B, T, HV, V]`` and
        ``S`` has shape ``[B, HV, K, V]`` if ``output_final_state`` else ``None``.
    """

    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q.repeat_interleave(G, dim=2) * scale   # [B, T, HV, K]
    k = k.repeat_interleave(G, dim=2)           # [B, T, HV, K]

    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
    if not output_final_state:
        S = None

    return o.to(dtype), S

def naive_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
):
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape ``[B, T, H, K]``.
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]``.
        v (torch.Tensor):
            Values of shape ``[B, T, HV, V]``. ``HV`` must be divisible by ``H``.
        g (torch.Tensor):
            Per-dimension decay gates (log-space) of shape ``[B, T, HV, K]``.
        beta (torch.Tensor):
            Beta scalars of shape ``[B, T, HV]``.
        scale (Optional[float]):
            Scale factor. Defaults to ``1 / sqrt(K)``.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape ``[B, HV, K, V]``.
        output_final_state (bool):
            Whether to return the final state.
        chunk_size (int):
            Chunk size for the chunked computation. Default: 64.

    Returns:
        A tuple ``(o, S)`` where ``o`` has shape ``[B, T, HV, V]`` and
        ``S`` has shape ``[B, HV, K, V]`` if ``output_final_state`` else ``None``.
    """
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    BT = chunk_size
    NT = T // BT
    if scale is None:
        scale = K ** -0.5
    assert T % BT == 0

    # Rearrange into chunks: [B, head, NT, BT, ...]
    q, k = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(torch.float) for x in [q, k]]
    v, g, beta = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(torch.float) for x in [v, g, beta]]
    # Expand q/k to value head dim for GVA: [B, H, ...] -> [B, HV, ...]
    q = q.repeat_interleave(G, dim=1) * scale  # [B, HV, NT, BT, K]
    k = k.repeat_interleave(G, dim=1)          # [B, HV, NT, BT, K]
    g = g.cumsum(-2)

    # note that diagonal is masked.
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)

    # Akk uses k (expanded to HV) and g (per value head)
    A = torch.zeros(*g.shape[:-1], BT, dtype=torch.float, device=q.device)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i:i+1, :]
        A[..., i] = torch.einsum('... c d, ... d -> ... c', k * (g - g_i).exp(), k_i)
    A = A * beta[..., None]

    A = -A.masked_fill(mask, 0)
    for i in range(1, BT):
        A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
    A = (A + torch.eye(BT, dtype=torch.float, device=q.device)) * beta[..., None, :]

    w = A @ (g.exp() * k)
    u = A @ v

    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, NT):
        # [B, HV, BT, ...]
        q_i = q[:, :, i]      # [B, HV, BT, K]
        k_i = k[:, :, i]      # [B, HV, BT, K]
        u_i = u[:, :, i]        # [B, HV, BT, V]
        g_i = g[:, :, i]        # [B, HV, BT, K]
        w_i = w[:, :, i]        # [B, HV, BT, K]
        # Aqk: per value head (q from qk head, g from value head, k from qk head)
        Aqk = torch.zeros(B, HV, BT, BT, dtype=torch.float, device=q.device)
        for j in range(BT):
            k_j = k[:, :, i, j]
            g_j = g[:, :, i, j:j+1, :]
            Aqk[..., j] = torch.einsum('... c d, ... d -> ... c', q_i * (g_i - g_j).exp(), k_j)
        Aqk = Aqk.masked_fill(mask, 0)
        v_i = u_i - w_i @ S
        o[:, :, i] = (q_i * g_i.exp()) @ S + Aqk @ v_i
        S = S * rearrange(g_i[:, :, -1].exp(), 'b h k -> b h k 1')
        S += rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, 'b h c k -> b h k c') @ v_i
    if not output_final_state:
        S = None
    return rearrange(o, 'b h n c d -> b (n c) h d').to(dtype), S


def naive_kda_lowerbound_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = -5.0,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    if lower_bound is None:
        g = -A_log.view(H, 1).exp() * F.softplus(g)
    else:
        g = lower_bound * F.sigmoid(A_log.view(H, 1).exp() * g)
    return g.to(output_dtype)


def naive_recurrent_kda_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log,
    dt_bias,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    lower_bound: float = -5.0,
    cu_seqlens: torch.Tensor | None = None,
    scale: float | None = None,
):

    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    if use_gate_in_kernel:
        g = naive_kda_lowerbound_gate(g, A_log, dt_bias, lower_bound=lower_bound)

    if cu_seqlens is not None:
        # Variable-length: loop over sequences (naive functions don't support cu_seqlens)
        N = cu_seqlens.shape[0] - 1
        o_list, ht_list = [], []
        for i in range(N):
            s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            h0_i = initial_state[i].unsqueeze(0) if initial_state is not None else None
            o_i, ht_i = naive_recurrent_kda(
                q=q[:, s:e], k=k[:, s:e], v=v[:, s:e],
                g=g[:, s:e], beta=beta[:, s:e],
                initial_state=h0_i, output_final_state=True,
            )
            o_list.append(o_i)
            ht_list.append(ht_i.squeeze(0))
        o = torch.cat(o_list, dim=1)
        # Pad output back to original sequence length (padding positions excluded by cu_seqlens)
        seq_len = q.shape[1]
        if o.shape[1] < seq_len:
            o = torch.cat([o, o.new_zeros(o.shape[0], seq_len - o.shape[1], *o.shape[2:])], dim=1)
        initial_state = torch.stack(ht_list, dim=0)
    else:
        o, initial_state = naive_recurrent_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=initial_state, output_final_state=True,
        )

    return o, initial_state


def naive_chunk_kda_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log,
    dt_bias,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    safe_gate: bool = True,
    lower_bound: float = -5.0,
    cu_seqlens: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = 64,
):

    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    if use_gate_in_kernel or safe_gate:
        g = naive_kda_lowerbound_gate(g, A_log, dt_bias, lower_bound=lower_bound)

    if cu_seqlens is not None:
        # Variable-length: loop over sequences (naive functions don't support cu_seqlens)
        N = cu_seqlens.shape[0] - 1
        o_list, ht_list = [], []
        for i in range(N):
            s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            h0_i = initial_state[i].unsqueeze(0) if initial_state is not None else None
            q_i, k_i, v_i, g_i, b_i = q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e]
            o_i, ht_i = naive_recurrent_kda(
                q=q_i, k=k_i, v=v_i,
                g=g_i, beta=b_i,
                initial_state=h0_i, output_final_state=True,
            )
            o_list.append(o_i)
            ht_list.append(ht_i.squeeze(0))
        o = torch.cat(o_list, dim=1)
        # Pad output back to original sequence length (padding positions excluded by cu_seqlens)
        seq_len = q.shape[1]
        if o.shape[1] < seq_len:
            o = torch.cat([o, o.new_zeros(o.shape[0], seq_len - o.shape[1], *o.shape[2:])], dim=1)
        initial_state = torch.stack(ht_list, dim=0)
    else:
        o, initial_state = naive_recurrent_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=initial_state, output_final_state=True,
        )

    return o, initial_state
