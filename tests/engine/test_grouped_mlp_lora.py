"""GroupedMLPLoRA behavior on ranks whose local experts receive no tokens."""

import pytest
import torch

from torchturbo.engine.peft.lora import GroupedMLPLoRA
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.moe_config import MoEConfig


@pytest.mark.parametrize("num_local_experts", [1, 2])
def test_zero_token_forward_keeps_adapters_in_graph(num_local_experts):
    torch.manual_seed(0)
    base = GroupedMLP(
        num_local_experts=num_local_experts,
        use_te=False,
        config=MoEConfig(hidden_size=4, moe_ffn_hidden_size=3, gated_linear_unit=True),
    )
    lora = GroupedMLPLoRA(base, "default", r=2, lora_alpha=4)
    hidden_states = torch.empty(0, 4, requires_grad=True)

    output, _ = lora(
        hidden_states,
        torch.zeros(num_local_experts, dtype=torch.long),
        torch.empty(0),
    )
    output.sum().backward()

    assert output.shape == (0, 4)
    adapter_parameters = {
        name: parameter
        for name, parameter in lora.named_parameters()
        if name.startswith("lora_")
    }
    assert len(adapter_parameters) == 4
    for name, parameter in adapter_parameters.items():
        # Every rank must reduce-scatter the same FSDP gradients, so ranks
        # without tokens still produce zero adapter gradients.
        assert parameter.grad is not None, name
        assert not parameter.grad.any(), name
