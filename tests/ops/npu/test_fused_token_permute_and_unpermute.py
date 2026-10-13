
import pytest

pytest.importorskip("torch_npu")
pytest.importorskip("mindspeed")

import torch
import torch_npu

from mindspeed import megatron_adaptor
from megatron.core.transformer.moe.moe_utils import permute
from megatron.core.transformer.moe.moe_utils import unpermute
from mindspeed.ops.npu_moe_token_permute import npu_moe_token_permute
from mindspeed.ops.npu_moe_token_unpermute import npu_moe_token_unpermute

from torchturbo.utils.torch_utils import compare_tensors

import time

TOL_MAPPING = {
    torch.float: dict(atol=1e-4, rtol=1e-4),
    torch.float16: dict(atol=1e-3, rtol=1e-3),
    torch.bfloat16: dict(atol=5e-3, rtol=5e-3),
}

class TestNpuFusedPermuteAndUnpermute():

    @pytest.mark.parametrize('num_tokens', [1024, 2048, 8192])
    @pytest.mark.parametrize('hidden_size', [6144, 8192, 12288])
    @pytest.mark.parametrize('topk', [1, 4])
    @pytest.mark.parametrize('num_experts', [4, 128])
    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    def test_permute(self, num_tokens, hidden_size, topk, num_experts, dtype):
        tols = TOL_MAPPING.get(dtype)
        # 1. simulate logits
        logits = torch.randn(num_tokens, num_experts)

        # 2. topk experts
        _, indices = torch.topk(logits, k=topk, dim=-1)
        token_ori = torch.randn(num_tokens, hidden_size).npu().to(dtype).requires_grad_(True)
        indices_ori = indices.npu()
        topk_map = torch.zeros(num_tokens, num_experts).npu().int().scatter(1, indices_ori, 1).bool()

        token_ori = token_ori.requires_grad_(True)
        token_fused = token_ori.clone().detach().requires_grad_(True)
        indices_fused = indices_ori.clone().detach()

        start_time = time.time()
        permuted_tokens_ori, _, sorted_indices_ori = permute(token_ori, topk_map, logits.npu())
        permuted_tokens_ori.backward(torch.ones(permuted_tokens_ori.shape).to(torch.bfloat16).npu())
        end_time = time.time()
        orig_time = end_time - start_time

        start_time = time.time()
        # indices_fused = torch.nonzero(topk_map, as_tuple=False)[:, 1].view(token_fused.shape[0], -1).to(token_fused.device)
        permuted_tokens_fused, sorted_indices_fused = npu_moe_token_permute(token_fused, indices_fused)
        # topk_map = topk_map.bool().T.contiguous()
        # permuted_probs = logits.npu().T.contiguous().masked_select(topk_map)
        permuted_tokens_fused.backward(torch.ones(permuted_tokens_fused.shape).to(torch.bfloat16).npu())
        end_time = time.time()
        fused_time = end_time - start_time
        print(f"permute - orig time / fused time {orig_time / fused_time}")

        assert torch.allclose(permuted_tokens_ori, permuted_tokens_fused, **tols)

    @pytest.mark.parametrize('num_tokens', [1024, 2048, 8192])
    @pytest.mark.parametrize('hidden_size', [6144, 8192, 12288])
    @pytest.mark.parametrize('topk', [1, 4])
    @pytest.mark.parametrize('num_experts', [4, 128])
    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    def test_unpermute(self, num_tokens, hidden_size, topk, num_experts, dtype):
        tols = TOL_MAPPING.get(dtype)
        # 1. simulate logits
        logits = torch.randn(num_tokens, num_experts)

        # 2. topk experts
        _, indices = torch.topk(logits, k=topk, dim=-1)
        permuted_tokens_ori = torch.randn(num_tokens * topk, hidden_size).npu().to(dtype)
        indices = indices.npu()
        routing_map_ori = torch.zeros(num_tokens, num_experts).npu().int().scatter(1, indices, 1).bool()
        routing_map = routing_map_ori.clone().bool().T.contiguous()
        token_indices = (
            torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
        )
        sorted_indices_ori = token_indices.masked_select(routing_map)
        probs_ori = None
        probs_fused = None
        if topk > 1:
            probs_ori = (torch.ones(num_tokens, topk)).npu().to(dtype).requires_grad_(True)
            probs_fused = probs_ori.clone().detach().requires_grad_(True)
        permuted_tokens_fused = permuted_tokens_ori.clone().detach().requires_grad_(True)
        sorted_indices_fused = torch.argsort(indices.view(-1), stable=True).npu().to(dtype=torch.int32)
        sorted_indices_fused = torch.argsort(sorted_indices_fused.view(-1), stable=True).npu().to(dtype=torch.int32)
        # The fusion operator will perform two torch.argsort operations internally
        restore_shape = (num_tokens, hidden_size)
        start_time = time.time()
        unpermuted_tokens_ori = unpermute(
            permuted_tokens_ori, sorted_indices_ori, restore_shape=restore_shape, probs=None)
        # unpermuted_tokens_ori.backward(torch.ones(unpermuted_tokens_ori.shape).to(torch.bfloat16).npu())
        end_time = time.time()
        orig_time = end_time - start_time
        start_time = time.time()
        # topk = int(routing_map_ori[0].sum(dim=0).item())
        # probs_fused = (torch.ones(routing_map_ori.shape[0], topk)).to(permuted_tokens_fused.device).to(permuted_tokens_fused.dtype).detach()
        probs_fused = (torch.ones(routing_map_ori.shape[0], topk, device=permuted_tokens_fused.device, dtype=permuted_tokens_fused.dtype))
        unpermuted_tokens_fused = npu_moe_token_unpermute(
            permuted_tokens_fused, sorted_indices_fused, probs=probs_fused)
        # unpermuted_tokens_fused.backward(torch.ones(unpermuted_tokens_fused.shape).to(torch.bfloat16).npu())
        end_time = time.time()
        fused_time = end_time - start_time
        print(f"unpermute - orig time / fused time {orig_time / fused_time}")
        assert torch.allclose(unpermuted_tokens_ori, unpermuted_tokens_fused, **tols)

    @pytest.mark.parametrize('num_tokens', [1024, 2048, 8192])
    @pytest.mark.parametrize('hidden_size', [6144, 8192, 12288])
    @pytest.mark.parametrize('topk', [1, 4])
    @pytest.mark.parametrize('num_experts', [4, 128])
    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    def test_npu_npu_moe_token_permute_unpermute(self, num_tokens, hidden_size, topk, num_experts, dtype):
        tols = TOL_MAPPING.get(dtype)
        # 1. simulate logits
        logits = torch.randn(num_tokens, num_experts)

        # 2. topk experts
        _, indices = torch.topk(logits, k=topk, dim=-1)
        token_ori = torch.randn(num_tokens, hidden_size).npu().to(dtype).requires_grad_(True)
        indices_ori = indices.npu()
        topk_map = torch.zeros(num_tokens, num_experts).npu().int().scatter(1, indices_ori, 1).bool()

        token_ori = token_ori.requires_grad_(True)
        token_fused = token_ori.clone().detach().requires_grad_(True)
        indices_fused = indices_ori.clone().detach()
        restore_shape = (num_tokens, hidden_size)
        start_time = time.time()
        permuted_tokens_ori, _, sorted_indices_ori = permute(token_ori, topk_map, logits.npu())
        unpermuted_tokens_ori = unpermute(
            permuted_tokens_ori, sorted_indices_ori, restore_shape=restore_shape, probs=None)
        end_time = time.time()
        orig_time = end_time - start_time

        start_time = time.time()
        permuted_tokens_fused, sorted_indices_fused = npu_moe_token_permute(token_fused, indices_fused)
        probs_fused = (torch.ones(topk_map.shape[0], topk, device=permuted_tokens_fused.device, dtype=permuted_tokens_fused.dtype))
        unpermuted_tokens_fused = npu_moe_token_unpermute(
            permuted_tokens_fused, sorted_indices_fused, probs=probs_fused)
        end_time = time.time()
        fused_time = end_time - start_time
        print(f"permute + unpermute - orig time / fused time {orig_time / fused_time}")


        tokens = torch.randn(num_tokens, hidden_size).npu().to(dtype)
        indices = torch.randint(0, num_experts, (num_tokens, topk)).npu()
        probs = None
        if topk > 1:
            probs = (torch.ones_like(indices) / topk).npu().to(dtype)


        permuted_tokens, sorted_indices = npu_moe_token_permute(tokens, indices)
        unpermuted_tokens = npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs=probs)

        if topk == 1:
            assert torch.equal(unpermuted_tokens, tokens)
        else:
            assert torch.allclose(unpermuted_tokens, tokens, **tols)
