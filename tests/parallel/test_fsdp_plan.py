"""Unit tests for FSDP parallel plan spec semantics."""

import os
import sys
import types

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

if "grouped_gemm" not in sys.modules:  # pragma: no cover - env shim
    _gg = types.ModuleType("grouped_gemm")
    _gg.ops = types.SimpleNamespace()
    sys.modules["grouped_gemm"] = _gg

import pytest

from torchturbo.parallel.fsdp import (
    FSDPParallelPlan,
    FSDPPlacementSpec,
    FSDPPlanSpec,
)
from torchturbo.engine.utils import match_fqn


def test_standalone_world_shard_spec_is_valid():
    spec = FSDPPlanSpec(standalone_world_shard=True)

    assert spec.standalone_world_shard is True
    assert spec.no_shard is False
    assert spec.placement is None


def test_no_shard_conflicts_with_standalone_world_shard():
    with pytest.raises(ValueError, match="mutually exclusive"):
        FSDPPlanSpec(no_shard=True, standalone_world_shard=True)


def test_no_shard_conflicts_with_placement():
    with pytest.raises(ValueError, match="mutually exclusive"):
        FSDPPlanSpec(no_shard=True, placement=FSDPPlacementSpec(object(), "world"))


def test_standalone_world_shard_conflicts_with_placement():
    with pytest.raises(ValueError, match="redundant"):
        FSDPPlanSpec(
            standalone_world_shard=True,
            placement=FSDPPlacementSpec(object(), "edp"),
        )


def test_parallel_plan_still_normalizes_tuple_placement_specs():
    plan = FSDPParallelPlan(plan={"layers.*.mlp.experts": (object(), "edp")})

    spec = plan.plan["layers.*.mlp.experts"]
    assert isinstance(spec, FSDPPlanSpec)
    assert isinstance(spec.placement, FSDPPlacementSpec)
    assert spec.placement.mesh_dim == "edp"


def test_match_fqn_star_stays_within_one_segment():
    assert match_fqn(
        "model.layers.*.q_*.weight",
        "model.layers.0.q_proj.weight",
    )
    assert not match_fqn(
        "model.layers.*.weight",
        "model.layers.0.projection.weight",
    )


def test_match_fqn_double_star_crosses_segment_boundaries():
    assert match_fqn(
        "model.layers.**.weight",
        "model.layers.0.attention.q_proj.weight",
    )


def test_parallel_plan_matches_modeling_fqn_only():
    spec = FSDPPlanSpec(no_shard=True)
    plan = FSDPParallelPlan(
        plan={"model.layers.*.projection.weight": spec}
    )

    assert plan.match("model.layers.0.projection.weight") is spec
    assert plan.match(
        "base_model.model.model.layers.0.projection.base_layer.weight"
    ) is None
