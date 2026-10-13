"""Explicit optimizer resource ownership across training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

from .....runtime import DeviceRuntime
from ....optimizers.container import OptimizersContainer
from .adamw import AdamWSwap
from .muon import FSDPDistributedMuon
from .state import wrap_cpu_state

if TYPE_CHECKING:
    from ..qk_clip import QKClipController


def _reject_suspended(*_args: Any) -> None:
    raise RuntimeError(
        "optimizer is suspended by engine.offload_optimizer(); "
        "call engine.restore_optimizer() before continuing"
    )


@dataclass(frozen=True)
class _TensorSpec:
    device: torch.device
    mesh: DeviceMesh | None
    placements: tuple[Any, ...]
    shape: torch.Size
    stride: tuple[int, ...]

    @classmethod
    def capture(cls, tensor: torch.Tensor) -> _TensorSpec:
        if isinstance(tensor, DTensor):
            return cls(
                tensor.device,
                tensor.device_mesh,
                tensor.placements,
                tensor.shape,
                tensor.stride(),
            )
        return cls(tensor.device, None, (), tensor.shape, tensor.stride())

    def restore(self, tensor: torch.Tensor) -> torch.Tensor:
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        local = local.to(self.device)
        if self.mesh is None:
            return local
        return DTensor.from_local(
            local,
            self.mesh,
            self.placements,
            run_check=False,
            shape=self.shape,
            stride=self.stride,
        )


def _copy_device_state_to_cpu(
    optimizers: tuple[Optimizer, ...],
    meshes: dict[DeviceMesh, DeviceMesh],
) -> tuple[
    list[tuple[dict[str, Any], str, torch.Tensor]],
    list[tuple[dict[str, Any], str, _TensorSpec]],
]:
    """Copy device state tensors to CPU and record how to restore them.

    Entries sharing one tensor object keep sharing one CPU copy. Source device
    tensors are referenced only by this frame, so they are released as soon
    as the caller replaces the state entries.
    """
    copies: dict[int, torch.Tensor] = {}
    pending = []
    parked = []
    for optimizer in optimizers:
        for state in optimizer.state.values():
            for key, value in state.items():
                if not isinstance(value, torch.Tensor) or value.device.type == "cpu":
                    continue
                if id(value) not in copies:
                    local = value.to_local() if isinstance(value, DTensor) else value
                    copies[id(value)] = wrap_cpu_state(local.to("cpu"), value, meshes)
                pending.append((state, key, copies[id(value)]))
                parked.append((state, key, _TensorSpec.capture(value)))
    return pending, parked


class OptimizerMemoryController:
    """Manage optimizer resources across training and temporary suspension.

    The configured CPU offload training policy allows steps and checkpoints.
    Only offload() and restore() change the temporary suspension state.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        runtime: DeviceRuntime,
        qk_clip: QKClipController | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.children = (
            tuple(optimizer)
            if isinstance(optimizer, OptimizersContainer)
            else (optimizer,)
        )
        self.runtime = runtime
        self.qk_clip = qk_clip
        self.is_suspended: bool = False
        # Original devices and layouts of the state tensors parked on CPU.
        self._parked: list[tuple[dict[str, Any], str, _TensorSpec]] = []
        self._meshes: dict[DeviceMesh, DeviceMesh] = {}
        # The final container must reject a step before any child can update.
        self._guarded = (
            (optimizer, *self.children)
            if isinstance(optimizer, OptimizersContainer)
            else self.children
        )
        # Hooks that reject updates and loads during temporary suspension.
        self._guards: list[RemovableHandle] = []

    def _validate_support(self) -> None:
        for child in self.children:
            if type(child) not in (torch.optim.AdamW, AdamWSwap, FSDPDistributedMuon):
                raise NotImplementedError(
                    f"optimizer memory management does not support {type(child).__name__}"
                )
            for group in child.param_groups:
                if group.get("capturable") or group.get("differentiable"):
                    raise NotImplementedError(
                        "optimizer offload requires eager, non-differentiable updates"
                    )
                for key, value in group.items():
                    if (
                        key != "params"
                        and isinstance(value, torch.Tensor)
                        and value.device.type != "cpu"
                    ):
                        raise NotImplementedError(
                            "optimizer offload does not support device tensor hyperparameters"
                        )
            for value in child.defaults.values():
                if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                    raise NotImplementedError(
                        "optimizer offload does not support device tensor defaults"
                    )

    def require_not_suspended(self) -> None:
        """Reject checkpoints between offload() and restore().

        DCP can initialize optimizer state on save and overwrite live tensors
        on load. Check before either operation starts. Configured CPU offload
        during training does not set is_suspended and allows checkpoint I/O.
        """
        if self.is_suspended:
            _reject_suspended()

    def _refresh_aliases(self) -> None:
        # AdamWSwap reads step counters through its own map, so rebind it to
        # the moved tensors. Its state map shares the state dicts themselves.
        for child in self.children:
            if isinstance(child, AdamWSwap):
                for param, state in child.state.items():
                    child.param_to_step_map[param] = state["step"]

    @torch.no_grad()
    def offload(self) -> None:
        """Suspend optimizer use and park device state until restore()."""
        if self.is_suspended:
            return
        self._validate_support()
        if any(
            param.grad is not None
            for group in self.optimizer.param_groups
            for param in group["params"]
        ):
            raise RuntimeError(
                "call optimizer.zero_grad(set_to_none=True) before offloading"
            )
        self.runtime.synchronize()
        # Allocate every CPU destination before changing any reference, so a
        # failed host allocation leaves the optimizer active and unchanged.
        pending, parked = _copy_device_state_to_cpu(self.children, self._meshes)
        for child in self.children:
            if isinstance(child, (FSDPDistributedMuon, AdamWSwap)):
                child.release_workspace()
        for state, key, value in pending:
            state[key] = value
        self._parked = parked
        self._refresh_aliases()
        # Inference forwards must not feed the next clip. QK statistics and row
        # indices are small, so they stay on the device.
        if self.qk_clip is not None:
            self.qk_clip.set_collection(False)
        for optimizer in self._guarded:
            self._guards.append(optimizer.register_step_pre_hook(_reject_suspended))
            # Reject a load before any other hook can rewrite parked state.
            self._guards.append(
                optimizer.register_load_state_dict_pre_hook(
                    _reject_suspended, prepend=True
                )
            )
        self.is_suspended = True
        self.runtime.empty_cache()

    @torch.no_grad()
    def restore(self) -> None:
        """End suspension using the configured CPU/device training policy."""
        if not self.is_suspended:
            return
        pending = []
        copies = {}
        try:
            for state, key, spec in self._parked:
                value = state[key]
                if id(value) not in copies:
                    copies[id(value)] = spec.restore(value)
                pending.append((state, key, copies[id(value)]))
            self.runtime.synchronize()
        except BaseException:
            # Release uncommitted device copies; the CPU state stays authoritative.
            pending.clear()
            copies.clear()
            self.runtime.empty_cache()
            raise
        for state, key, value in pending:
            state[key] = value
        self._refresh_aliases()
        if self.qk_clip is not None:
            self.qk_clip.set_collection(True)
        for guard in self._guards:
            guard.remove()
        self._guards = []
        self._parked = []
        self.is_suspended = False
