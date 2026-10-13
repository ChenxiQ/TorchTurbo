from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

from torchturbo.ops.providers.npu.cross_entropy_loss import NpuCrossEntropyLoss


def _install_fake_torch_npu(monkeypatch, result: torch.Tensor) -> None:
    torch_npu = ModuleType("torch_npu")
    torch_npu.npu_cross_entropy_loss = lambda *args, **kwargs: result
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)


@pytest.mark.parametrize("shape", [(1,), (2, 1)])
def test_npu_cross_entropy_preserves_unreduced_shape(monkeypatch, shape):
    expected = torch.ones(shape)
    _install_fake_torch_npu(monkeypatch, expected)

    actual = NpuCrossEntropyLoss()(
        torch.randn(2, 3),
        torch.tensor([0, 1]),
        reduction="none",
    )

    assert actual.shape == expected.shape


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_npu_cross_entropy_scalarizes_reduced_loss(monkeypatch, reduction):
    _install_fake_torch_npu(monkeypatch, torch.ones(1))

    actual = NpuCrossEntropyLoss()(
        torch.randn(2, 3),
        torch.tensor([0, 1]),
        reduction=reduction,
    )

    assert actual.shape == torch.Size([])
