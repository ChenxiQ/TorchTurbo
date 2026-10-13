"""Apply MLA QK clipping directly to FSDP2 parameter shards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from ....models.modules.attention.qk_clip import QKClipState
from ...optimizers.qk_clip import MLAQKClipSpec

if TYPE_CHECKING:
    from ...optimizers.qk_clip import QKClipPlan


@dataclass(frozen=True, slots=True)
class _BoundWeight:
    """Static per-row rescaling metadata for one projection weight.

    Resolved once against the local FSDP shard so each clip step is a
    single vectorized multiply.
    """

    parameter: nn.Parameter
    # Head owning each local row; per row because FSDP can split a head.
    head_indices: Tensor
    # Power of that head's scale applied to each row.
    exponents: Tensor

    @torch.no_grad()
    def scale(self, head_scales: Tensor) -> None:
        parameter = self.parameter
        local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
        row_scales = head_scales[self.head_indices].pow(self.exponents)
        local.mul_(row_scales.unsqueeze(1))
        if isinstance(parameter, DTensor):
            # A local view mutation does not update the DTensor's version counter.
            torch.autograd.graph.increment_version(parameter)


@dataclass(frozen=True, slots=True)
class _BoundLayer:
    state: QKClipState
    query: _BoundWeight
    key_value: _BoundWeight


class QKClipController:
    """Collect per-step head maxima and clip local MLA projection weights.

    Skipped AMP optimizer steps leave the weights unchanged, so accumulated
    finite maxima stay valid across them.
    """

    def __init__(
        self,
        plan: QKClipPlan,
        *,
        sp_size: int = 1,
        device: torch.device | None = None,
    ) -> None:
        if not plan.layers:
            raise ValueError("QK clipping requires at least one attention layer")
        if any(layer.spec.num_heads % sp_size for layer in plan.layers):
            raise ValueError("QK clipping requires num_heads divisible by sp_size")
        self._threshold = plan.config.threshold
        self._num_heads = tuple(layer.spec.num_heads for layer in plan.layers)
        weights = []
        meshes = []
        for layer in plan.layers:
            spec = layer.spec
            if not isinstance(spec, MLAQKClipSpec):
                raise ValueError(
                    f"FSDP2 QK clipping does not support {type(spec).__name__}"
                )
            query, query_mesh = self._bind_weight(
                layer.weights[spec.query_projection],
                num_heads=spec.num_heads,
                rows_per_head=spec.qk_nope_head_dim + spec.qk_rope_head_dim,
                nope_rows=spec.qk_nope_head_dim,
                nope_exponent=plan.config.alpha,
                remaining_exponent=1.0,
            )
            key_value, key_value_mesh = self._bind_weight(
                layer.weights[spec.key_value_projection],
                num_heads=spec.num_heads,
                rows_per_head=spec.qk_nope_head_dim + spec.v_head_dim,
                nope_rows=spec.qk_nope_head_dim,
                nope_exponent=1.0 - plan.config.alpha,
                remaining_exponent=0.0,
            )
            weights.append((query, key_value))
            meshes.extend((query_mesh, key_value_mesh))

        mesh = meshes[0]
        if any(weight_mesh != mesh for weight_mesh in meshes):
            raise ValueError("QK clipping requires all MLA weights on the same mesh")
        if sp_size > 1 and mesh is None:
            raise ValueError(
                "QK clipping with SP requires weights on the dense world mesh"
            )
        parameter_device = weights[0][0].parameter.device
        if any(
            weight.parameter.device != parameter_device
            for pair in weights
            for weight in pair
        ):
            raise ValueError("QK clipping requires all MLA weights on the same device")
        if device is None:
            device = parameter_device
        if mesh is not None:
            self._reduction_group = mesh.get_group()
            self._reduction_size = mesh.size()
        elif dist.is_initialized():
            # Replicated no_shard weights synchronize gradients over WORLD.
            self._reduction_group = dist.group.WORLD
            self._reduction_size = dist.get_world_size()
        else:
            self._reduction_group = None
            self._reduction_size = 1

        self._max_logits = torch.full(
            (sum(self._num_heads),), -torch.inf, dtype=torch.float32, device=device
        )
        states = [
            QKClipState(max_logits=layer_maxima)
            for layer_maxima in self._max_logits.split(self._num_heads)
        ]
        bound_layers = []
        for layer, state, (query, key_value) in zip(
            plan.layers, states, weights, strict=True
        ):
            layer.module.qk_clip_state = state
            bound_layers.append(_BoundLayer(state, query, key_value))
        self._layers = tuple(bound_layers)

    def set_collection(self, enabled: bool) -> None:
        """Pause or resume statistics collection, keeping observed maxima."""
        for layer in self._layers:
            layer.state.enabled = enabled

    @staticmethod
    def _bind_weight(
        parameter: nn.Parameter,
        *,
        num_heads: int,
        rows_per_head: int,
        nope_rows: int,
        nope_exponent: float,
        remaining_exponent: float,
    ) -> tuple[_BoundWeight, DeviceMesh | None]:
        if parameter.is_meta:
            raise ValueError("QK clipping requires materialized projection weights")
        if parameter.ndim != 2 or parameter.shape[0] != num_heads * rows_per_head:
            raise ValueError("QK clipping metadata does not match the MLA weight shape")

        mesh = None
        row_offset = 0
        local = parameter
        if isinstance(parameter, DTensor):
            mesh = parameter.device_mesh
            if (
                mesh.ndim != 1
                or len(parameter.placements) != 1
                or type(parameter.placements[0]) is not Shard
                or parameter.placements[0].dim != 0
            ):
                raise ValueError("QK clipping requires a one-dimensional Shard(0) mesh")
            if mesh.size() != dist.get_world_size():
                raise ValueError("QK clipping requires the dense world mesh")
            coordinate = mesh.get_coordinate()
            assert coordinate is not None
            local_rows, row_offset = Shard.local_shard_size_and_offset(
                parameter.shape[0], mesh.size(), coordinate[0]
            )
            local = parameter.to_local()
            if local.shape != (local_rows, parameter.shape[1]):
                raise ValueError("QK clipping local weight shape does not match Shard(0)")

        # FSDP boundaries may split a head, so map each local row independently.
        rows = torch.arange(row_offset, row_offset + local.shape[0], device=local.device)
        head_indices = rows // rows_per_head
        nope = rows.remainder(rows_per_head) < nope_rows
        exponents = torch.where(nope, nope_exponent, remaining_exponent)
        return _BoundWeight(parameter, head_indices, exponents), mesh

    @torch.no_grad()
    def reset(self) -> None:
        """Discard maxima after a completed update."""
        self._max_logits.fill_(-torch.inf)
        for layer in self._layers:
            layer.state.observed = False

    @torch.no_grad()
    def clip(self) -> None:
        """Reduce head maxima once, then scale Q/K rows without weight gathers."""
        if self._reduction_size > 1:
            dist.all_reduce(
                self._max_logits, op=dist.ReduceOp.MAX, group=self._reduction_group
            )
        scales = self._threshold / self._max_logits.clamp_min(self._threshold)
        for layer, head_scales in zip(
            self._layers, scales.split(self._num_heads), strict=True
        ):
            layer.query.scale(head_scales.to(layer.query.parameter.device))
            layer.key_value.scale(head_scales.to(layer.key_value.parameter.device))
        self.reset()

    def _step_post_hook(
        self,
        optimizer: torch.optim.Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self.clip()

    def _step_pre_hook(
        self,
        optimizer: torch.optim.Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        # Collection pauses while the optimizer is suspended; report that state
        # instead of the missing statistics it causes.
        if any(not layer.state.enabled for layer in self._layers):
            raise RuntimeError(
                "optimizer is suspended by engine.offload_optimizer(); "
                "call engine.restore_optimizer() before continuing"
            )
        if (len(args) > 1 and args[1] is not None) or kwargs.get("closure") is not None:
            raise ValueError("QK clipping does not support optimizer closures")
        if any(not layer.state.observed for layer in self._layers):
            raise RuntimeError(
                "QK clipping requires attention statistics from every configured "
                "layer before the optimizer step"
            )


def install_qk_clip(
    plan: QKClipPlan,
    optimizer: torch.optim.Optimizer,
    *,
    sp_size: int = 1,
    device: torch.device | None = None,
) -> QKClipController:
    """Bind attention statistics and clip after the complete optimizer update."""
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    for layer in plan.layers:
        if vars(layer.module).get("qk_clip_state") is not None:
            raise ValueError("QK clipping is already installed on an attention layer")
        if any(
            id(parameter) not in optimizer_parameters
            for parameter in layer.weights.values()
        ):
            raise ValueError(
                "QK clipping requires the optimizer to own every Q/K weight"
            )
    controller = QKClipController(
        plan,
        sp_size=sp_size,
        device=device,
    )
    optimizer.register_step_pre_hook(controller._step_pre_hook)
    optimizer.register_step_post_hook(controller._step_post_hook)
    return controller
