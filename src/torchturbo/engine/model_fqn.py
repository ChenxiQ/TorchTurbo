"""Map model-authored FQNs to the prepared module tree."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.util import find_spec
from types import MappingProxyType

import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)


@dataclass(frozen=True, slots=True)
class ModelFQNMap:
    """Map model-authored FQNs to exact FQNs in the prepared model.

    Keys are used by recipes and parallel plans; values identify the prepared
    objects. Adapter-only parameters and inactive model copies are omitted.
    """

    modules: Mapping[str, str]
    parameters: Mapping[str, str]


def build_model_fqn_map(model: nn.Module) -> ModelFQNMap:
    """Build the active modeling-FQN to prepared-FQN mapping for ``model``.

    Wrapper types define the structural edges explicitly. This keeps pattern
    matching independent of wrapper implementation names and records only the
    branch that participates in the current forward pass.

    Args:
        model: Prepared model whose active execution tree is mapped.

    Returns:
        An immutable snapshot of module and parameter FQN mappings.
    """
    if find_spec("peft") is not None:
        from peft import PeftModel
        from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
        from peft.utils.other import (
            ModulesToSaveWrapper,
            TrainableTokensWrapper,
        )
    else:
        PeftModel = None
        BaseTuner = None
        BaseTunerLayer = None
        ModulesToSaveWrapper = None
        TrainableTokensWrapper = None

    modules: dict[str, str] = {}
    parameters: dict[str, str] = {}
    visited_module_ids: set[int] = set()
    visited_parameter_ids: set[int] = set()

    def join(prefix: str, name: str) -> str:
        return f"{prefix}.{name}" if prefix else name

    def add(mapping: dict[str, str], model_fqn: str, prepared_fqn: str) -> None:
        if model_fqn in mapping and mapping[model_fqn] != prepared_fqn:
            raise ValueError(
                f"Model FQN {model_fqn!r} resolves to both "
                f"{mapping[model_fqn]!r} and {prepared_fqn!r}"
            )
        mapping[model_fqn] = prepared_fqn

    def walk(
        module: nn.Module,
        model_fqn: str,
        prepared_fqn: str,
        *,
        map_current_module: bool = True,
    ) -> None:
        if id(module) in visited_module_ids:
            return
        visited_module_ids.add(id(module))

        if PeftModel is not None and isinstance(module, PeftModel):
            # PeftModel and BaseTuner add container edges, not modeling edges.
            walk(
                module.base_model,
                model_fqn,
                join(prepared_fqn, "base_model"),
                map_current_module=map_current_module,
            )
            return

        if BaseTuner is not None and isinstance(module, BaseTuner):
            walk(
                module.model,
                model_fqn,
                join(prepared_fqn, "model"),
                map_current_module=map_current_module,
            )
            return

        if ModulesToSaveWrapper is not None and isinstance(
            module, ModulesToSaveWrapper
        ):
            # Only one branch participates in forward. Mapping the wrapper's
            # other registered copies would give one modeling FQN many targets.
            if module.disable_adapters or not module.active_adapters:
                target = module.original_module
                target_fqn = join(prepared_fqn, "original_module")
            else:
                adapter_name = module.active_adapters[0]
                target = module.modules_to_save[adapter_name]
                target_fqn = join(
                    join(prepared_fqn, "modules_to_save"), adapter_name
                )
            walk(
                target,
                model_fqn,
                target_fqn,
                map_current_module=map_current_module,
            )
            return

        if TrainableTokensWrapper is not None and isinstance(
            module, TrainableTokensWrapper
        ):
            # The wrapper is the modeling module boundary. Its token adapter
            # executes both the base embedding and adapter delta, but only the
            # base embedding has modeling parameter FQNs.
            if map_current_module:
                add(modules, model_fqn, prepared_fqn)
            walk(
                module.token_adapter,
                model_fqn,
                join(prepared_fqn, "token_adapter"),
                map_current_module=False,
            )
            return

        if BaseTunerLayer is not None and isinstance(module, BaseTunerLayer):
            # The tuner layer is the executable module boundary. Its base layer
            # supplies the original modeling parameters; adapter parameters do
            # not have modeling FQNs and are deliberately left out of the map.
            if map_current_module:
                add(modules, model_fqn, prepared_fqn)
            walk(
                module.base_layer,
                model_fqn,
                join(prepared_fqn, "base_layer"),
                map_current_module=False,
            )
            return

        if isinstance(module, CheckpointWrapper):
            # CheckpointWrapper is the executable module boundary, while its
            # wrapped-module edge is transparent to child and parameter FQNs.
            if map_current_module:
                add(modules, model_fqn, prepared_fqn)
            walk(
                module._checkpoint_wrapped_module,
                model_fqn,
                join(prepared_fqn, "_checkpoint_wrapped_module"),
                map_current_module=False,
            )
            return

        if map_current_module:
            add(modules, model_fqn, prepared_fqn)

        for name, parameter in module.named_parameters(recurse=False):
            if id(parameter) in visited_parameter_ids:
                continue
            visited_parameter_ids.add(id(parameter))
            add(
                parameters,
                join(model_fqn, name),
                join(prepared_fqn, name),
            )

        for name, child in module.named_children():
            walk(
                child,
                join(model_fqn, name),
                join(prepared_fqn, name),
            )

    walk(model, "", "")
    return ModelFQNMap(
        modules=MappingProxyType(modules),
        parameters=MappingProxyType(parameters),
    )


__all__ = ["ModelFQNMap", "build_model_fqn_map"]
