"""Tests for engine-owned freeze policy application."""

import os

import pytest
import torch.nn as nn

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")


class _GroupedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Linear(2, 2)
        self.language = nn.Linear(2, 2)

    def get_parameter_groups(self):
        from torchturbo.engine.config import ParameterGroup

        return {
            ParameterGroup.VISION_TOWER: self.vision.named_parameters(
                prefix="vision"
            ),
            ParameterGroup.LANGUAGE_MODEL: self.language.named_parameters(
                prefix="language"
            ),
        }


def test_engine_applies_freeze_policy_to_model_parameter_group():
    from torchturbo.engine.core import TorchTurboEngine
    from torchturbo.engine.config import FreezePolicy, ParameterGroup

    engine = object.__new__(TorchTurboEngine)
    model = _GroupedModel()

    engine._apply_freeze_policy(
        model, FreezePolicy((ParameterGroup.VISION_TOWER,))
    )

    assert all(not p.requires_grad for p in model.vision.parameters())
    assert all(p.requires_grad for p in model.language.parameters())


def test_engine_rejects_unsupported_freeze_group():
    from torchturbo.engine.core import TorchTurboEngine
    from torchturbo.engine.config import FreezePolicy, ParameterGroup

    engine = object.__new__(TorchTurboEngine)
    model = _GroupedModel()

    with pytest.raises(ValueError, match="unsupported parameter group"):
        engine._apply_freeze_policy(
            model, FreezePolicy((ParameterGroup.MULTI_MODAL_PROJECTOR,))
        )
