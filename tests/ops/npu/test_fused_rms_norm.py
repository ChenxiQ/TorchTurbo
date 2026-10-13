import pytest
pytest.importorskip("torch_npu")
pytest.importorskip("mindspeed")
import torch
from torch import nn
import os
import torch_npu
from mindspeed import megatron_adaptor
from mindspeed.op_builder import RmsNormOpBuilder
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm
from torch_npu import npu_rms_norm
import time

class TestNpuRmsNorm:

    @pytest.mark.parametrize('batch', [6])
    @pytest.mark.parametrize('seq', [60])
    @pytest.mark.parametrize('hidden_size', [1024])
    @pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
    def test_npu_rms_norm(self, batch, seq, hidden_size, dtype):
        x = torch.randn((batch, seq, hidden_size), device="npu", dtype=dtype)

        #* transformers 中的 RMS
        RMSNorm = Qwen3MoeRMSNorm(hidden_size).npu().to(dtype)

        start_time = time.time()
        output_ori = RMSNorm(x)
        end_time = time.time()
        orig_time = end_time - start_time

        #* mindspeed 中的 RMS
        start_time = time.time()
        gamma = nn.Parameter(torch.ones(hidden_size, device="npu", dtype=RMSNorm.weight.dtype))


        # mindspeed_ops = RmsNormOpBuilder().load()
        #* 替换为 torch_npu 中的融合算子
        output_fused = npu_rms_norm(x, gamma, epsilon=1e-6)[0]
        # output_fused = RMSNorm_npu(x)
        end_time = time.time()
        fused_time = end_time - start_time
        print(f"rms_norm - orig time / fused time {orig_time / fused_time}")

        diff =  output_ori - output_fused
        print("diff", diff)
        print("diff_abs_sum ", torch.abs(diff).sum())
        # output_fused = mindspeed_ops.rms_norm(x, gamma)
        tol = 0.004 if dtype == torch.bfloat16 else 0.001


        assert torch.allclose(output_ori, output_fused, rtol=tol, atol=tol)
