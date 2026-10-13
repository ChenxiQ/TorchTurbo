"""Generic parallel plan base."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ParallelPlan:
    """Base generic parallel plan for any backend.

    Attributes
    ----------
    plan:
        The backend-specific sharding specification.
    """

    plan: Any = None
