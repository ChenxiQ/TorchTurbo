"""FSDP2 on-disk checkpoint components.

The engine layout owns shared files such as the HF export and scheduler.
Exact model state is stored as DCP, which is specific to this backend.
"""

from __future__ import annotations

import os

from ....checkpoint.format import ComponentStatus

MODEL_DCP_DIR = "model_distcp"
OPTIMIZER_DCP_DIR = "optimizer_distcp"

# DCP writes ``.metadata`` (``__<rank>.metadata`` without rank coordination)
# after all data files.
_DCP_METADATA_SUFFIX = ".metadata"


def dcp_status(dcp_dir: str) -> ComponentStatus:
    """Return the save state of a DCP directory."""
    if not os.path.isdir(dcp_dir):
        return ComponentStatus.MISSING
    for name in os.listdir(dcp_dir):
        is_global_metadata = name == _DCP_METADATA_SUFFIX
        is_rank_metadata = name.startswith("__") and name.endswith(
            _DCP_METADATA_SUFFIX
        )
        if is_global_metadata or is_rank_metadata:
            return ComponentStatus.COMPLETE
    return ComponentStatus.INCOMPLETE
