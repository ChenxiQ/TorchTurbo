from copy import deepcopy
import sys
import pytest

pytest.importorskip("torch_npu")
pytest.importorskip("mindspeed")

import torch
import torch_npu
from tests_extend.unit_tests.common import DistributedTest
from mindspeed import megatron_adaptor
from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding
from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb

class TestNpuFusedRotaryEmbedding(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
    def test_npu_fused_rotary_pos_embedding(self, dtype):
        x_ori = torch.rand(4, 8192, 4, 128).npu().to(dtype)

        x_fused = deepcopy(x_ori)

        cos = torch.rand(1, 8192, 1, 128).npu().to(dtype)
        sin = torch.rand(1, 8192, 1, 128).npu().to(dtype)

        cos_fused = deepcopy(cos)
        sin_fused = deepcopy(sin)

        output_ori, _ = apply_rotary_pos_emb(x_ori, x_ori, cos, sin)
        output_fused = npu_rotary_position_embedding(x_fused, cos_fused, sin_fused, 0)

        tol = 0.004 if dtype == torch.bfloat16 else 0.001
        assert torch.allclose(output_ori, output_fused, rtol=tol, atol=tol)
