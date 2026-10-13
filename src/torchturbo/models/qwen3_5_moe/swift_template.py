"""Re-exports the shared Qwen3.5 swift template wrap.

The MTP ``input_ids`` preservation is identical for dense and MoE, so the real
implementation lives in
:mod:`torchturbo.models.qwen3_5.swift_template`; this module just keeps the
per-model ``register`` entrypoint the msswift adapter looks up
(``models/<name>/swift_template.py``) for ``qwen3_5_moe``.
"""

from __future__ import annotations

from ..qwen3_5.swift_template import register

__all__ = ["register"]
