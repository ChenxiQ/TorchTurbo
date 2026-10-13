"""Per-stage wiring for the ms-swift adapter: a stage (SFT/PT; RLHF later)
is a :class:`SwiftStageSpec` in this registry; :mod:`plugin`,
:mod:`model_registry` and ``run_swift.py`` are all driven from it.
``stage_wiring`` imports lazily so this module stays swift-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SwiftStageSpec:
    """Wiring for one ms-swift training stage (SFT, PT, DPO, ...)."""

    name: str
    #: swift module holding the pipeline class global to swap, e.g.
    #: ``"swift.pipelines.train.sft"``.
    pipeline_module: str
    #: attribute name on that module, e.g. ``"SwiftSft"``.
    pipeline_attr: str


TURBO_SWIFT_STAGES: List[SwiftStageSpec] = []


def register_stage(spec: SwiftStageSpec) -> None:
    """Append a stage spec (idempotent on ``spec.name``)."""
    if any(s.name == spec.name for s in TURBO_SWIFT_STAGES):
        return
    TURBO_SWIFT_STAGES.append(spec)


def iter_stages() -> Tuple[SwiftStageSpec, ...]:
    """Snapshot of registered stages (tuple so callers can iterate safely)."""
    return tuple(TURBO_SWIFT_STAGES)


def stage_wiring(stage: str) -> tuple:
    """Return (turbo pipeline class, native swift main) for the stage."""
    from . import pipeline

    if stage == "sft":
        from swift.pipelines.train.sft import sft_main

        return pipeline.TurboSwiftSft, sft_main
    if stage == "pt":
        from swift.pipelines.train.pretrain import pretrain_main

        return pipeline.TurboSwiftPretrain, pretrain_main
    if stage == "dpo":
        from swift.pipelines.train.rlhf import rlhf_main

        return pipeline.TurboSwiftRLHF, rlhf_main
    raise KeyError(stage)


# Turbo trainers keyed by swift's task/rlhf type (orthogonal to stages:
# SFT and PT both resolve through "causal_lm"; a future RLHF stage adds
# "dpo"/"kto"/... here).
TURBO_TRAINERS: Dict[str, str] = {
    "causal_lm": "torchturbo.adapters.msswift.trainer_factory.TurboSeq2SeqTrainer",
    "dpo": "torchturbo.adapters.msswift.trainer_factory.TurboDPOTrainer",
}

SFT_STAGE = SwiftStageSpec(
    name="sft",
    pipeline_module="swift.pipelines.train.sft",
    pipeline_attr="SwiftSft",
)

# PT -- ``SwiftPretrain`` is a thin ``SwiftSft`` subclass; reuses SFT's
# ``causal_lm`` trainer, no new entries.
PT_STAGE = SwiftStageSpec(
    name="pt",
    pipeline_module="swift.pipelines.train.pretrain",
    pipeline_attr="SwiftPretrain",
)

# DPO uses Swift's RLHF pipeline and dpo trainer mapping.
DPO_STAGE = SwiftStageSpec(
    name="dpo",
    pipeline_module="swift.pipelines.train.rlhf",
    pipeline_attr="SwiftRLHF",
)

register_stage(SFT_STAGE)
register_stage(PT_STAGE)
register_stage(DPO_STAGE)
