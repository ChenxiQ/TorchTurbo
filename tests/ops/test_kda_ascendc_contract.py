"""CPU checks of the KDA adapter's upstream layout and autograd contract."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from torchturbo.ops.providers.npu.kda import ascendc_chunk_kda


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("sigmoid", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_recompute_layout_and_gradients(monkeypatch, packed, normalize, sigmoid, with_bias):
    batch, tokens, heads, dim = (1 if packed else 2), 65, 2, 128
    shape = (batch, tokens, heads, dim)
    token_shape = shape[1:] if packed else shape
    head_shape = (heads, tokens, dim) if packed else (batch, heads, tokens, dim)
    calls = []

    def forward(q, k, v, g, beta, scale, chunk_size, **kw):
        assert kw["layout"] == ("TND" if packed else "BSND")
        assert kw["disable_recompute"] is False
        assert all(x.shape == token_shape and x.is_contiguous() for x in (q, k, v, g))
        assert beta.shape == token_shape[:-1]
        if normalize:
            for name, value in (("q_hat_out", 11), ("k_hat_out", 12)):
                assert kw[name].shape == head_shape
                kw[name].fill_(value)
            kw["q_rstd_out"].fill_(1)
            kw["k_rstd_out"].fill_(1)
        if sigmoid:
            assert kw["beta_eff_out"].shape == head_shape[:-1]
            kw["beta_eff_out"].fill_(0.5)
        matrix = torch.empty(*head_shape[:-1], 64, dtype=q.dtype)
        calls.append((g, beta))
        return v.clone(), None, None, matrix, matrix.clone(), None, None, None, None, None, None, None

    def backward(q, k, v, beta, gk, aqk, akk, w, qg, kg, v_new, h, do, scale, **kw):
        assert all(x is None for x in (gk, w, qg, kg, v_new, h))
        assert kw["disable_recompute"] is False
        assert kw["layout"] == ("TND" if packed else "BSND")
        assert do.shape == token_shape and do.is_contiguous()
        torch.testing.assert_close(do, upstream.squeeze(0) if packed else upstream)
        assert kw["raw_g"] is calls[0][0]
        assert beta is calls[0][1]  # Original logits, never beta_eff.
        assert kw["use_beta_sigmoid_in_kernel"] is sigmoid
        assert (kw["beta_eff"] is not None) is sigmoid
        assert (kw["q_rstd"] is not None) is normalize
        assert q.shape == (head_shape if normalize else token_shape)
        if normalize:
            assert torch.all(q == 11) and torch.all(k == 12)
        if packed:
            assert kw["cu_seqlens"] == [0, 33, 65]
            assert kw["chunk_indices"] == [0, 0, 1, 0]
        calls.append("backward")
        return (
            torch.full_like(v, 1), torch.full_like(v, 2), torch.full_like(v, 3),
            torch.full_like(beta, 5), torch.full_like(kw["raw_g"], 4), None,
            torch.full_like(kw["A_log"], 6),
            torch.full((heads, dim), 7, dtype=torch.float32) if with_bias else None,
        )

    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", SimpleNamespace(
        chunk_kda_fwd=forward, npu_chunk_kda_bwd=backward,
    ))
    q, k, v = [torch.randn(batch, tokens, heads * dim, dtype=torch.bfloat16,
                           requires_grad=True) for _ in range(3)]
    g = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
    beta = torch.randn(shape[:-1], requires_grad=True)
    a = torch.randn(heads, requires_grad=True)
    bias = torch.randn(heads * dim, requires_grad=True) if with_bias else None
    cu = torch.tensor([0, 0, 33, 65], dtype=torch.int32) if packed else None
    output, state = ascendc_chunk_kda(
        q, k, v, g, beta, A_log=a, dt_bias=bias, cu_seqlens=cu,
        safe_gate=True, use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=normalize, use_beta_sigmoid_in_kernel=sigmoid,
    )
    assert output.shape == shape and state is None
    upstream = torch.randn(batch, heads, tokens, dim, dtype=q.dtype).transpose(1, 2)
    output.backward(upstream)
    for value, tensor in enumerate((q, k, v, g, beta, a, bias), 1):
        if tensor is not None:
            torch.testing.assert_close(tensor.grad, torch.full_like(tensor, value))
    assert calls[-1] == "backward"


def test_inference_omits_backward_caches(monkeypatch):
    def forward(q, k, v, g, beta, scale, chunk_size, **kw):
        assert kw["disable_recompute"] is False
        for name in ("q_hat_out", "k_hat_out", "q_rstd_out", "k_rstd_out", "beta_eff_out"):
            assert kw[name] is None
        return v.clone(), None, None, None, None, None, None, None, None, None, None, None

    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", SimpleNamespace(chunk_kda_fwd=forward))
    q = torch.randn(1, 65, 256, dtype=torch.bfloat16)
    with torch.no_grad():
        output, _ = ascendc_chunk_kda(
            q, q, q, torch.randn(1, 65, 2, 128), torch.randn(1, 65, 2),
            A_log=torch.randn(2), safe_gate=True, use_gate_in_kernel=True,
            use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
        )
    assert output.shape == (1, 65, 2, 128)
    assert not output.requires_grad


def test_empty_sequences_preserve_final_state_slots(monkeypatch):
    def forward(q, k, v, g, beta, scale, chunk_size, **kw):
        assert kw["cu_seqlens"] == [0, 33, 65]
        assert kw["chunk_indices"] == [0, 0, 1, 0]
        states = torch.stack((torch.ones(2, 128, 128), torch.full((2, 128, 128), 2.)))
        return v.clone(), states, None, None, None, None, None, None, None, None, None, None

    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", SimpleNamespace(chunk_kda_fwd=forward))
    q = torch.randn(1, 65, 256, dtype=torch.bfloat16)
    with torch.no_grad():
        _, state = ascendc_chunk_kda(
            q, q, q, torch.randn(1, 65, 2, 128), torch.randn(1, 65, 2),
            A_log=torch.randn(2), safe_gate=True, use_gate_in_kernel=True,
            output_final_state=True, cu_seqlens=torch.tensor([0, 0, 33, 33, 65, 65]),
        )
    assert state.shape == (5, 2, 128, 128)
    for index, expected in enumerate((0, 1, 0, 2, 0)):
        torch.testing.assert_close(state[index], torch.full_like(state[index], expected))
