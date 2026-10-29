"""NPU CP adapter parity, without importing the model's optional dependencies.

Run with torchrun --standalone --nproc_per_node=2 tests/ops/run_kda_allgather_cp.py.
Use --interleaved-groups with four processes to test two noncontiguous CP groups.
This checks real AscendC kernels and HCCL, but does not validate FSDP training.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
import fla_npu  # noqa: F401


def load_adapter():
    root = Path(__file__).resolve().parents[2]
    path = root / "src/torchturbo/ops/providers/npu/_fla_npu_kda/__init__.py"
    spec = importlib.util.spec_from_file_location("kda_cp_adapter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compare(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict:
    a, b = actual.float(), expected.float()
    error = (a - b).norm() / b.norm().clamp_min(1e-8)
    max_abs = (a - b).abs().max()
    # Chunk boundaries change BF16 rounding. Check both norm and elementwise error.
    passed = bool(torch.allclose(a, b, atol=0.02, rtol=0.02) and error < 0.03)
    return dict(name=name, passed=passed, rel_l2=error.item(), max_abs=max_abs.item())


def run_case(adapter, group, offsets: list[int] | None, seed: int) -> list[dict]:
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    tokens, heads, dim = 512, 4, 128
    local_tokens = tokens // size
    lo, hi = rank * local_tokens, (rank + 1) * local_tokens
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    gen = torch.Generator().manual_seed(seed)
    qkv = [torch.randn(1, heads, tokens, dim, generator=gen).bfloat16().to(device)
           for _ in range(3)]
    gate = (torch.randn(1, tokens, heads, dim, generator=gen) * 0.1 - 5).to(device)
    beta = torch.randn(1, tokens, heads, generator=gen).to(device)
    params = [torch.zeros(heads, device=device), torch.zeros(heads * dim, device=device)]
    source = qkv + [gate, beta] + params
    do = torch.randn(1, tokens, heads, dim, generator=gen).bfloat16().to(device)
    reference_inputs = [t.detach().requires_grad_() for t in source]
    kwargs = dict(disable_recompute=True, head_major_qkv=True,
                  use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                  use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0)
    # The existing fused baseline rejects repeated offsets; empty sequences
    # have no tokens or state contribution and can be omitted in that reference.
    reference_offsets = None if offsets is None else (
        [offsets[0]] + [b for a, b in zip(offsets, offsets[1:]) if b > a]
    )
    global_cu = None if offsets is None else torch.tensor(reference_offsets, dtype=torch.int64)
    ref, _ = adapter.ascendc_chunk_kda(
        *reference_inputs[:5], A_log=reference_inputs[5], dt_bias=reference_inputs[6],
        cu_seqlens=global_cu, **kwargs,
    )
    ref_grad = torch.autograd.grad(ref, reference_inputs, do)
    context = adapter.KDAContextParallel.from_global_offsets(
        [0, tokens] if offsets is None else offsets, local_tokens, group,
    )
    local_inputs = []
    for i, tensor in enumerate(source):
        t = tensor[:, :, lo:hi] if i < 3 else tensor[:, lo:hi] if i < 5 else tensor
        local_inputs.append(t.detach().contiguous().requires_grad_())
    out, final_state = adapter.ascendc_chunk_kda(
        *local_inputs[:5], A_log=local_inputs[5], dt_bias=local_inputs[6],
        cu_seqlens=torch.tensor(context.cu_seqlens), cp_context=context, **kwargs,
    )
    assert final_state is None
    if offsets is None and rank == 0:
        # This shard has zero h0 and the same aligned chunk boundaries as the
        # reference. A BF16 h0 silently switches FwdH's recurrent precision.
        torch.testing.assert_close(out, ref[:, lo:hi], atol=0, rtol=0)
    grads = torch.autograd.grad(out, local_inputs, do[:, lo:hi].contiguous())
    rows = [compare("output", out, ref[:, lo:hi])]
    for i, (name, actual, expected) in enumerate(zip(
        ("dq", "dk", "dv", "dg", "dbeta", "dA_log", "ddt_bias"), grads, ref_grad,
    )):
        if i < 3:
            expected = expected[:, :, lo:hi]
        elif i < 5:
            expected = expected[:, lo:hi]
        else:
            # Test-only reduction: production leaves this to FSDP/DDP.
            actual = actual.clone()
            dist.all_reduce(actual, group=group)
        rows.append(compare(name, actual, expected))
    print(json.dumps(dict(rank=dist.get_rank(), offsets=offsets,
                          local_offsets=context.cu_seqlens, results=rows)), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interleaved-groups", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.npu.matmul.allow_hf32 = False
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        world = dist.get_world_size()
        if args.interleaved_groups:
            assert world == 4
            groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
            group = groups[dist.get_rank() % 2]
        else:
            group = dist.new_group(list(range(world)))
        adapter = load_adapter()
        rows = []
        # Dense, crossing packed boundary, exact boundary, and an empty sequence.
        for offsets in (None, [0, 291, 482, 512], [0, 256, 512], [0, 0, 17, 512]):
            rows.extend(run_case(adapter, group, offsets, 29))
        ok = torch.tensor(int(all(row["passed"] for row in rows)), device="npu")
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if not ok.item():
            raise AssertionError("KDA AllGather CP parity failed; inspect per-gradient results.")
        if dist.get_rank() == 0:
            print("KDA AllGather CP parity PASS", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
