"""Local optimizer state and its distributed checkpoint metadata."""

from __future__ import annotations

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor


def wrap_cpu_state(
    tensor: torch.Tensor,
    reference: torch.Tensor,
    meshes: dict[DeviceMesh, DeviceMesh],
) -> torch.Tensor:
    """Wrap a CPU shard in a DTensor with ``reference``'s distributed layout.

    The CPU mesh only carries layout metadata, so no communication group is
    created. Scalars and state of non-DTensor references stay unwrapped.
    """
    if not isinstance(reference, DTensor) or tensor.ndim == 0:
        return tensor
    source = reference.device_mesh
    if source not in meshes:
        meshes[source] = DeviceMesh(
            "cpu",
            source.mesh,
            mesh_dim_names=source.mesh_dim_names,
            _init_backend=False,
        )
    return DTensor.from_local(
        tensor,
        meshes[source],
        reference.placements,
        run_check=False,
        shape=reference.shape,
        stride=reference.stride(),
    )
