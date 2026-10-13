import pytest
pytest.importorskip("torch_npu")
pytest.importorskip("mindspeed")
import torch
import torch_npu
from mindspeed import megatron_adaptor
import torch.nn.functional as F
from megatron.core.jit import jit_fuser
from torchturbo.utils.torch_utils import compare_tensors

@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_npu_swiglu(dtype):
    sequence_length, batch_size, hidden_size = 32, 2, 12
    hidden_states = torch.randn((sequence_length, batch_size, hidden_size)).npu().to(dtype)
    @jit_fuser
    def glu(x):
        x = torch.chunk(x, 2, dim=-1)
        return F.silu(x[0]) * x[1]
    output_ori = glu(hidden_states)
    output_fused = torch_npu.npu_swiglu(hidden_states, dim=-1)
    tol = 0.004 if dtype == torch.bfloat16 else 0.001
    compare_tensors("swiglu", output_fused, output_ori, rank=0, atol=tol)
    assert torch.allclose(output_ori, output_fused, rtol=tol, atol=tol)