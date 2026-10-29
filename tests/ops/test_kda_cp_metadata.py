"""CPU tests for packed CP ownership; no AscendC import is needed."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


@pytest.fixture
def cp_module():
    path = (Path(__file__).resolve().parents[2]
            / "src/torchturbo/ops/providers/npu/_fla_npu_kda/cp.py")
    spec = importlib.util.spec_from_file_location("kda_cp_metadata", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name)


@pytest.mark.parametrize("offsets,rank,expected,prefix,suffix", [
    ([0, 32], 0, (0, 8), 0, 3),
    ([0, 32], 2, (0, 8), 2, 1),
    ([0, 11, 25, 32], 1, (0, 3, 8), 1, 2),
    ([0, 11, 25, 32], 3, (0, 1, 8), 2, 0),
    ([0, 8, 16, 24, 32], 2, (0, 8), 0, 0),
    ([0, 0, 9, 9, 32], 1, (0, 1, 8), 1, 2),
    ([0, 1, 2, 32], 0, (0, 1, 2, 8), 0, 3),
])
def test_local_packed_ownership(cp_module, monkeypatch, offsets, rank, expected, prefix, suffix):
    group = object()
    monkeypatch.setattr(cp_module.dist, "get_rank", lambda g: rank if g is group else -1)
    monkeypatch.setattr(cp_module.dist, "get_world_size", lambda g: 4 if g is group else -1)
    context = cp_module.KDAContextParallel.from_global_offsets(offsets, 8, group)
    assert context.cu_seqlens == expected
    assert (context.prefix, context.suffix) == (prefix, suffix)
    assert context.group is group


def test_rejects_global_offsets_for_wrong_token_count(cp_module, monkeypatch):
    monkeypatch.setattr(cp_module.dist, "get_rank", lambda g: 0)
    monkeypatch.setattr(cp_module.dist, "get_world_size", lambda g: 2)
    with pytest.raises(ValueError, match="cover"):
        cp_module.KDAContextParallel.from_global_offsets([0, 31], 16, object())
    with pytest.raises(ValueError, match="explicit"):
        cp_module.KDAContextParallel.from_global_offsets([0, 32], 16, None)


@pytest.mark.parametrize("forward,index,count", [(True, 0, 1), (False, -1, 2)])
def test_exchange_uses_explicit_group_and_only_boundary_slot(
    cp_module, monkeypatch, forward, index, count,
):
    from types import SimpleNamespace

    group = object()
    context = cp_module.KDAContextParallel(group, 1, 4, (0, 3, 5, 8), 1, 2)
    calls = []

    def gather(outputs, tensor, *, group):
        assert group is context.group
        for i, output in enumerate(outputs):
            output.fill_(i)

    def merge(state, gathered, received, rank, *, forward, state_v_first):
        calls.append((received, rank, forward))
        assert gathered.shape == (4, 2, 128, 256)
        assert not state_v_first
        state.fill_(7)

    monkeypatch.setattr(cp_module.dist, "all_gather", gather)
    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc",
                        SimpleNamespace(merge_fwd_bwd_kernel=merge))
    result = context.exchange(torch.zeros(2, 128, 256), forward=forward, dtype=torch.bfloat16)
    assert calls == [(count, 1, forward)]
    assert result.dtype == torch.bfloat16
    assert torch.all(result[index] == 7)
    assert torch.count_nonzero(result) == result[index].numel()
