"""On-disk layout of the TorchTurbo part of a training checkpoint.

A checkpoint root (for example ``<output_dir>/checkpoint-100``) holds one
directory or file per component. Each component has a commit marker that is
written last, so a component without its marker belongs to an interrupted save.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from .format import CheckpointFormat, ComponentStatus

HF_MODEL_DIR = "huggingface"
HF_INDEX_FILE = "model.safetensors.index.json"
SCHEDULER_FILE = "scheduler.pt"
METADATA_DIR = "torchturbo"


def hf_status(hf_dir: str) -> ComponentStatus:
    """Return the save state of an HF safetensors directory.

    The index is the last file written and names every shard file.
    """
    if not os.path.isdir(hf_dir):
        return ComponentStatus.MISSING
    index_path = os.path.join(hf_dir, HF_INDEX_FILE)
    if not os.path.isfile(index_path):
        return ComponentStatus.INCOMPLETE
    try:
        with open(index_path) as index_file:
            index = json.load(index_file)
    except (OSError, json.JSONDecodeError):
        return ComponentStatus.INCOMPLETE
    if not isinstance(index, dict):
        return ComponentStatus.INCOMPLETE
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return ComponentStatus.INCOMPLETE
    if not all(
        isinstance(name, str) and isinstance(shard, str)
        for name, shard in weight_map.items()
    ):
        return ComponentStatus.INCOMPLETE
    shard_names = set(weight_map.values())
    for shard_name in shard_names:
        if not os.path.isfile(os.path.join(hf_dir, shard_name)):
            return ComponentStatus.INCOMPLETE
    return ComponentStatus.COMPLETE


def resolve_model_format(
    checkpoint_dir: str,
    native: ComponentStatus,
    native_dir: str,
) -> CheckpointFormat:
    """Return how model weights in ``checkpoint_dir`` should be restored.

    ``native`` is the backend's exact model-state status. It wins over an HF
    export. An incomplete optional HF export does not invalidate complete
    native state.

    Raises:
        RuntimeError: A model weights directory belongs to an interrupted save.
        FileNotFoundError: ``checkpoint_dir`` holds no model weights.
    """
    if native is ComponentStatus.INCOMPLETE:
        raise RuntimeError(
            f"Model weights in {native_dir} are incomplete; the save was "
            "interrupted. Resume from an earlier checkpoint or remove "
            "the incomplete directory."
        )
    if native is ComponentStatus.COMPLETE:
        return CheckpointFormat.NATIVE

    hf_dir = os.path.join(checkpoint_dir, HF_MODEL_DIR)
    hf = hf_status(hf_dir)
    if hf is ComponentStatus.INCOMPLETE:
        raise RuntimeError(
            f"Model weights in {hf_dir} are incomplete; the save was "
            "interrupted. Resume from an earlier checkpoint or remove "
            "the incomplete directory."
        )
    if hf is ComponentStatus.COMPLETE:
        return CheckpointFormat.HF
    raise FileNotFoundError(f"No model weights found in {checkpoint_dir}.")


def save_scheduler(state: dict[str, Any], checkpoint_dir: str) -> None:
    """Write LR scheduler state atomically. Call on rank 0 only."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, SCHEDULER_FILE)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as scheduler_file:
        torch.save(state, scheduler_file)
        scheduler_file.flush()
        os.fsync(scheduler_file.fileno())
    os.replace(tmp_path, path)
    sync_directory(checkpoint_dir)


def load_scheduler(checkpoint_dir: str) -> dict[str, Any]:
    """Read LR scheduler state written by :func:`save_scheduler`.

    Raises:
        FileNotFoundError: The state file is missing.
        RuntimeError: The state file is incomplete or invalid.
    """
    path = os.path.join(checkpoint_dir, SCHEDULER_FILE)
    # Older saves wrote this file in place; an empty or truncated file is an
    # interrupted one.
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"LR scheduler state {path} is missing, but the optimizer "
            "state was saved. Resume from an earlier checkpoint."
        )
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"LR scheduler state {path} is incomplete; the save was "
            "interrupted. Resume from an earlier checkpoint."
        ) from exc
    if not isinstance(state, dict):
        raise RuntimeError(
            f"LR scheduler state {path} is invalid; expected a state dict."
        )
    return state


def validate_output_dir(output_dir: str, source_path: str) -> None:
    """Reject an output directory that is the source model or inside it.

    HF exports copy the source model directory tree, so an output inside the
    source would be copied into itself, and resetting a checkpoint there could
    delete source files. Call this before training or exporting starts.

    Args:
        output_dir: Directory that receives checkpoints or exports.
        source_path: Source model path given to the engine.

    Raises:
        ValueError: ``output_dir`` is ``source_path`` or inside it.
    """
    source = os.path.realpath(source_path)
    output = os.path.realpath(output_dir)
    if os.path.commonpath((source, output)) == source:
        raise ValueError(
            f"Output directory {output_dir} must not be the source model "
            f"directory {source_path} or inside it. HF exports copy the source "
            "model directory tree, so the output would be copied into itself, "
            "and resetting a checkpoint there could delete source files. "
            "Choose an output directory outside the source model."
        )


def save_json(data: dict[str, Any], path: str) -> None:
    """Atomically write and sync a JSON file and its parent directory."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w") as file:
        json.dump(data, file, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    sync_directory(directory)


def sync_directory(path: str | os.PathLike[str]) -> None:
    """Make directory entry updates durable after creating or replacing files."""
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
