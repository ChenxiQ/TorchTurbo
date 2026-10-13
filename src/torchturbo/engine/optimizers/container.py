# The facade over several optimizers follows the approach taken by TorchTitan's
# OptimizersContainer.

"""A scheduler-compatible container for prepared optimizer instances."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from torch.optim import Optimizer

from .plan import OptimizerPlan, ParamGroup


class OptimizersContainer(Optimizer):
    """Expose multiple optimizers through one PyTorch optimizer interface.

    Child parameter-group dictionaries are reused directly in ``param_groups``
    so an LR scheduler operating on this container updates the real optimizers.
    """

    _STATE_DICT_FORMAT_VERSION = 1

    def __init__(
        self,
        plan: OptimizerPlan,
        optimizers_by_name: Mapping[str, Optimizer],
    ) -> None:
        if not isinstance(plan, OptimizerPlan):
            raise TypeError("plan must be an OptimizerPlan")
        if not isinstance(optimizers_by_name, Mapping):
            raise TypeError("optimizers_by_name must be a mapping")
        expected_param_groups = plan.get_param_groups()
        expected_groups = plan.get_param_groups_by_optimizer_name()
        if not isinstance(expected_param_groups, tuple) or not all(
            isinstance(group, ParamGroup) for group in expected_param_groups
        ):
            raise TypeError("plan.get_param_groups() must return a tuple of ParamGroup")
        if not isinstance(expected_groups, dict) or not all(
            isinstance(groups, tuple)
            and all(isinstance(group, ParamGroup) for group in groups)
            for groups in expected_groups.values()
        ):
            raise TypeError(
                "plan.get_param_groups_by_optimizer_name() must return "
                "dict[str, tuple[ParamGroup, ...]]"
            )

        expected_names = tuple(
            dict.fromkeys(group.optimizer_name for group in expected_param_groups)
        )
        if tuple(expected_groups) != expected_names:
            raise ValueError("optimizer plan parameter-group views are inconsistent")
        for optimizer_name, groups in expected_groups.items():
            flat_groups = tuple(
                group
                for group in expected_param_groups
                if group.optimizer_name == optimizer_name
            )
            if len(groups) != len(flat_groups) or any(
                group is not flat_group
                for group, flat_group in zip(groups, flat_groups, strict=True)
            ):
                raise ValueError(
                    "optimizer plan parameter-group views are inconsistent"
                )

        prepared_names = tuple(optimizers_by_name)
        if set(prepared_names) != set(expected_names):
            raise ValueError(
                "backend prepared optimizer names do not match optimizer plan: "
                f"expected {expected_names!r}, got {prepared_names!r}"
            )

        self.optimizer_names = list(expected_names)
        self.optimizers_by_name = {
            name: optimizers_by_name[name] for name in expected_names
        }
        self.optimizers = list(self.optimizers_by_name.values())

        expected_fqn_by_param_id = {}
        expected_optimizer_by_param_id = {
            id(param): optimizer_name
            for optimizer_name, groups in expected_groups.items()
            for group in groups
            for param in group.params_by_fqn.values()
        }
        expected_kwargs_by_param_id = {
            id(param): group.optimizer_kwargs
            for groups in expected_groups.values()
            for group in groups
            for param in group.params_by_fqn.values()
        }
        all_params = []
        for groups in expected_groups.values():
            for group in groups:
                for param_name, param in group.params_by_fqn.items():
                    expected_fqn_by_param_id[id(param)] = param_name
                    all_params.append(param)

        prepared_param_ids = []
        for name, optimizer in self.optimizers_by_name.items():
            if not isinstance(optimizer, Optimizer):
                raise TypeError(
                    f"backend optimizer {name!r} must be a torch.optim.Optimizer"
                )
            for group in optimizer.param_groups:
                params = group["params"]
                for param in params:
                    param_name = expected_fqn_by_param_id.get(id(param))
                    if param_name is None:
                        raise ValueError(
                            f"backend optimizer {name!r} returned a parameter "
                            "outside the optimizer plan"
                        )
                    if expected_optimizer_by_param_id.get(id(param)) != name:
                        raise ValueError(
                            f"backend optimizer {name!r} returned parameter {param_name!r} "
                            "assigned to a different optimizer"
                        )
                    for option_name, expected_value in expected_kwargs_by_param_id[
                        id(param)
                    ].items():
                        if (
                            option_name not in group
                            or group[option_name] != expected_value
                        ):
                            raise ValueError(
                                f"backend optimizer {name!r} changed parameter-group "
                                f"option {option_name!r} for {param_name!r}"
                            )
                prepared_param_ids.extend(id(param) for param in params)

        expected_param_ids = {id(param) for param in all_params}
        actual_param_ids = set(prepared_param_ids)
        if len(prepared_param_ids) != len(actual_param_ids):
            raise ValueError("backend optimizers contain duplicate parameters")
        if actual_param_ids != expected_param_ids:
            raise ValueError(
                "backend optimizers do not contain exactly the assigned parameters"
            )

        self._param_groups_frozen = False
        try:
            Optimizer.__init__(
                self,
                all_params,
                self._shared_child_defaults(self.optimizers),
            )
        finally:
            self._param_groups_frozen = True
        # Keep the exact child dictionaries so schedulers mutate child learning
        # rates instead of a detached facade copy.
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]

    @staticmethod
    def _shared_child_defaults(optimizers: Sequence[Optimizer]) -> dict[str, Any]:
        """Return the default options every child optimizer agrees on.

        Schedulers index ``defaults["lr"]`` directly. An option the children
        disagree on, such as ``eps``, has no meaningful facade value.
        """
        first, *rest = optimizers
        return {
            key: value
            for key, value in first.defaults.items()
            if all(
                key in other.defaults and other.defaults[key] == value
                for other in rest
            )
        }

    def __iter__(self) -> Iterator[Optimizer]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if self._param_groups_frozen:
            raise RuntimeError(
                "OptimizersContainer does not support add_param_group() "
                "after construction"
            )
        super().add_param_group(param_group)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if closure is not None:
            raise ValueError("OptimizersContainer does not support closures")
        for optimizer in self.optimizers:
            optimizer.step()
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def refresh_param_groups(self) -> None:
        """Rebind the facade to the child optimizers' live parameter groups.

        ``Optimizer.load_state_dict`` swaps a child's group dictionaries for
        fresh copies, which leaves the facade pointing at orphaned dictionaries.
        Callers that load state into the children must call this afterwards or an
        LR scheduler would keep writing learning rates nothing reads.
        """
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]

    def state_dict(self) -> dict[str, Any]:
        """Return versioned state owned by each named child optimizer."""
        return {
            "format_version": self._STATE_DICT_FORMAT_VERSION,
            "optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in self.optimizers_by_name.items()
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load versioned state into each named child optimizer."""
        if not isinstance(state_dict, dict):
            raise TypeError("state_dict must be a dict")
        state_dict = state_dict.copy()
        for hook in self._optimizer_load_state_dict_pre_hooks.values():
            replacement = hook(self, state_dict)
            if replacement is not None:
                state_dict = replacement

        expected_keys = {"format_version", "optimizers"}
        if set(state_dict) != expected_keys:
            raise ValueError(
                "OptimizersContainer state_dict must contain exactly "
                "'format_version' and 'optimizers'"
            )
        if state_dict["format_version"] != self._STATE_DICT_FORMAT_VERSION:
            raise ValueError(
                "unsupported OptimizersContainer state_dict format version: "
                f"{state_dict['format_version']!r}"
            )

        optimizer_states = state_dict["optimizers"]
        if not isinstance(optimizer_states, Mapping):
            raise TypeError("state_dict['optimizers'] must be a mapping")

        expected_names = tuple(self.optimizers_by_name)
        actual_names = tuple(optimizer_states)
        if actual_names != expected_names:
            raise ValueError(
                "optimizer state names do not match the container: "
                f"expected {expected_names!r}, got {actual_names!r}"
            )

        try:
            for name, optimizer in self.optimizers_by_name.items():
                optimizer.load_state_dict(optimizer_states[name])
        finally:
            # Optimizer.load_state_dict replaces parameter-group dictionaries.
            self.refresh_param_groups()
        for hook in self._optimizer_load_state_dict_post_hooks.values():
            hook(self)


__all__ = [
    "OptimizersContainer",
]
