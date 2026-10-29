"""FLA KCP model-to-provider contracts without device kernels."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

import torchturbo.ops as turbo_ops
from torchturbo.models.bailing_hybrid import attention
from torchturbo.ops import get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime


@pytest.fixture
def fla_calls(monkeypatch):
    calls = {"context": [], "conv": [], "kda": []}
    for name in ("fla", "fla.ops", "fla.modules"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    def build_cp_context(cu_seqlens, *, group, conv1d_kernel_size):
        context = object()
        calls["context"].append((cu_seqlens, group, conv1d_kernel_size, context))
        return context

    def conv(**kwargs):
        calls["conv"].append(kwargs)
        return kwargs["x"], None

    def kda(**kwargs):
        calls["kda"].append(kwargs)
        return kwargs["q"], None

    monkeypatch.setitem(sys.modules, "fla.ops.cp", SimpleNamespace(build_cp_context=build_cp_context))
    monkeypatch.setitem(sys.modules, "fla.ops.kda", SimpleNamespace(chunk_kda=kda))
    monkeypatch.setitem(sys.modules, "fla.modules.convolution", SimpleNamespace(causal_conv1d=conv))
    monkeypatch.setitem(sys.modules, "fla.modules.fused_norm_gate", SimpleNamespace(
        rms_norm_gated=lambda x, *args, **kwargs: x,
    ))
    dispatcher = get_or_create_ops(device_runtime=CpuRuntime())
    monkeypatch.setattr(turbo_ops, "ops", dispatcher)
    return calls


@pytest.mark.parametrize("sp_size", [1, 2, 4])
@pytest.mark.parametrize("no_kda_lora", [True, False])
def test_fla_kcp_preserves_local_tokens_and_full_heads(fla_calls, sp_size, no_kda_lora):
    class Layer(nn.Module):
        pass

    layer = Layer()
    layer.mode, layer.no_kda_lora = "chunk", no_kda_lora
    layer.num_heads, layer.head_k_dim, layer.head_dim, layer.conv_size = 3, 2, 2, 4
    for name in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj", "o_proj",
                 "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj"):
        setattr(layer, name, nn.Identity())
    layer.b_proj = nn.Linear(6, 3, bias=False)
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        setattr(layer, name, SimpleNamespace(weight=torch.ones(6, 1, 4), bias=None, activation="silu"))
    layer.A_log, layer.dt_bias = torch.zeros(3), torch.zeros(6)
    layer.safe_gate, layer.lower_bound = True, -5.0
    layer.o_norm = nn.Module()
    layer.o_norm.weight, layer.o_norm.activation, layer.o_norm.eps = torch.ones(2), "silu", 1e-6
    group = object()
    attention.patch_kda_forward(Layer, group, sp_size, 0, use_fla_kcp=True)
    x = torch.randn(1, 4, 6, requires_grad=True)
    offsets = torch.tensor([0, 3, 4 * sp_size - 1])
    output, _, _ = layer(x, cu_seqlens=offsets)
    output.sum().backward()
    torch.testing.assert_close(output, x)
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    cu, actual_group, kernel_size, context = fla_calls["context"][0]
    assert cu.tolist() == [0, 3, 4 * sp_size - 1, 4 * sp_size]
    assert actual_group is group and kernel_size == 4
    assert len(fla_calls["conv"]) == 3
    for call in fla_calls["conv"]:
        assert call["x"].shape == (1, 4, 6)
        assert call["weight"].shape == (6, 4)
        assert call["cp_context"] is context
        assert call["cu_seqlens"] is None
    (call,) = fla_calls["kda"]
    assert call["q"].shape == (1, 4, 3, 2)
    assert call["cp_context"] is context
    assert call["A_log"] is layer.A_log and call["dt_bias"] is layer.dt_bias
    assert call["output_final_state"] is False
    assert "use_beta_sigmoid_in_kernel" not in call
    torch.testing.assert_close(call["beta"], layer.b_proj(x).float().sigmoid())


def test_cp_backend_switches_are_mutually_exclusive(fla_calls, monkeypatch):
    with pytest.raises(ValueError, match="enable only one"):
        attention.patch_kda_forward(nn.Module, object(), 2, 0,
                                    use_fla_kcp=True, use_fla_kcp_ascend=True)
