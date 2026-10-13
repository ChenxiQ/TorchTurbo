"""turbo-owned swift argument classes.

The turbo ``--tt_*`` CLI surface lives on the shared
:class:`torchturbo.engine.args.TorchTurboArgs`; the launcher
(``scripts/runs/run_swift.py``) strips ``--tt_*`` from ``sys.argv`` and stashes
a parsed ``TorchTurboArgs`` via :func:`set_torchturbo_args` before swift's
parser runs. The swift argument classes below therefore declare NO ``tt_*``
fields: swift's dataclass parser sees only swift-native args, and the turbo
config builder reads ``--tt_*`` back through :func:`get_torchturbo_args`.
Adding a new ``--tt_*`` flag touches only ``engine/args.py`` + the shared
``engine/config.py`` (``build_from_args``) — ms-swift and llamafactory pick
it up together.

:class:`TurboSwiftArgsMixin` carries the ``__post_init__`` that turns OFF the
swift/HF settings the turbo engine takes ownership of (activation
checkpointing, sequence parallelism, FSDP / DeepSpeed sharding), so the two
never compete. It is reused by every stage's arguments class (SFT, PT and
DPO), so each stage only declares the swift-native fields specific to it.

These classes are **module-level** on purpose (not built inside a factory
closure): swift's multiprocess dataset pipelines (``dataset.map`` with
``num_proc>1``) pickle objects with ``dill``; a closure-local class pickles
*by value* (``save_type``) and trips on the ABCMeta internals
(``_abc._abc_data``) of ``SwiftPipeline(ABC)``. Module-level pickles *by
reference*.

This module imports ``swift.arguments`` (heavy) at top level. It is imported
lazily — only from the msswift ``pipeline`` on the turbo-swift code path
(activated via ``torchturbo.activate("msswift")``) — so it never fires on a
plain ``import torchturbo``.
"""

from __future__ import annotations

from dataclasses import dataclass

from swift.arguments import PretrainArguments, RLHFArguments, SftArguments


class TurboSwiftArgsMixin:
    """Force OFF the swift/HF settings the turbo engine owns.

    Mixed in AHEAD of the swift arguments base (e.g.
    ``class TurboSftArguments(TurboSwiftArgsMixin, SftArguments)``) so its
    ``__post_init__`` runs first, sets the engine-owned knobs, then chains to
    the swift base's own ``__post_init__`` via ``super()``.
    """

    def __post_init__(self):
        # turbo owns activation checkpointing — never let swift/HF re-wrap
        # model forwards after FSDP2 sharding.
        self.gradient_checkpointing = False
        if hasattr(self, "vit_gradient_checkpointing"):
            self.vit_gradient_checkpointing = False
        if getattr(self, "gradient_checkpointing_kwargs", None):
            self.gradient_checkpointing_kwargs = None
        # turbo owns sequence parallelism — keep swift's own SP path off.
        self.sequence_parallel_size = 1
        # turbo's FSDP2 owns sharding — leave accelerate out of it.
        self.fsdp = ""
        self.deepspeed = None
        super().__post_init__()


@dataclass
class TurboSftArguments(TurboSwiftArgsMixin, SftArguments):  # type: ignore[misc]
    """Swift ``SftArguments`` for the turbo SFT stage.

    Holds NO ``tt_*`` fields — those live on the shared ``TorchTurboArgs`` and
    are stripped from ``sys.argv`` by the launcher before swift's parser runs.
    The engine-owned swift settings are forced off by :class:`TurboSwiftArgsMixin`.

    ``@dataclass`` gives the subclass its own ``__dataclass_fields__`` (a copy
    of the inherited ones); the generated ``__init__`` still calls
    ``self.__post_init__`` because ``hasattr(cls, '__post_init__')`` sees the
    mixin's. So the mixin's force-off runs before ``super().__post_init__()``.
    """


@dataclass
class TurboPretrainArguments(TurboSwiftArgsMixin, PretrainArguments):  # type: ignore[misc]
    """Swift ``PretrainArguments`` for the turbo PT stage. The mixin must run
    ahead of its ``__post_init__`` so accelerate/deepspeed stay off (the engine
    owns them); with the native class they would compete with the engine.
    """


@dataclass
class TurboRLHFArguments(TurboSwiftArgsMixin, RLHFArguments):  # type: ignore[misc]
    """Swift arguments for the TorchTurbo DPO stage."""
