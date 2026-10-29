"""Exercise the actual patched KDA forward with real conv/KDA kernels and A2A.

This standalone layer fixture loads source files without importing optional
model/engine packages. It uses the repository's Ulysses functions unchanged,
and a PyTorch RMSNorm gate in place of the optional FLA Triton norm. It is not
a full model or FSDP test. Run under torchrun with two or four NPU processes.
"""

from __future__ import annotations

import ast
import argparse
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Optional, Tuple

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch import Tensor
import torch_npu  # noqa: F401
import fla_npu  # noqa: F401

from run_kda_allgather_cp import load_adapter


ROOT = Path(__file__).resolve().parents[2] / "src/torchturbo"


def load_file(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def norm_gate(x, g, weight, bias, activation, *, eps, **kwargs):
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return (normalized * weight.float() * torch.nn.functional.silu(g.float())).to(x.dtype)


def install_test_imports(adapter):
    runtime = SimpleNamespace(
        new_stream=torch.npu.Stream, new_event=torch.npu.Event,
        current_stream=torch.npu.current_stream, stream=torch.npu.stream,
    )
    def get_device_runtime():
        return runtime
    sys.modules["torchturbo.runtime"] = SimpleNamespace(get_device_runtime=get_device_runtime)
    # Import only the production communication definitions; engine registration
    # and unrelated attention implementations require optional model packages.
    path = ROOT / "models/modules/attention/ulysses/ulysses.py"
    tree = ast.parse(path.read_text())
    names = {"_all_to_all", "_all_to_all_single", "all_to_all_tensor", "SeqAllToAll",
             "gather_seq_scatter_heads", "gather_heads_scatter_seq"}
    subset = ast.Module(body=[node for node in tree.body
                             if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                             and node.name in names], type_ignores=[])
    namespace = dict(torch=torch, dist=dist, Tensor=Tensor, ProcessGroup=ProcessGroup,
                     Optional=Optional, Any=Any, Tuple=Tuple,
                     get_device_runtime=get_device_runtime)
    exec(compile(subset, str(path), "exec"), namespace)
    module_name = "torchturbo.models.modules.attention.ulysses.ulysses"
    comm = ModuleType(module_name)
    comm.__dict__.update(namespace)
    sys.modules[module_name] = comm

    # Packed metadata's cache decorator is the only dependency outside torch.
    sys.modules["torchturbo.utils.functools"] = SimpleNamespace(cache=lambda fn: fn)
    packed = load_file("torchturbo.ops.packed_params", "ops/packed_params.py")
    conv = load_file("bridge_conv", "ops/providers/npu/_fla_gdn/ascendc_causal_conv1d_function.py")

    def causal_conv1d(*, x, weight, head_num, bias, activation, initial_state,
                      output_final_state, packed_params):
        return conv.AscendCCausalConv1dFunction.apply(
            x, weight, head_num, bias, None, initial_state, activation,
            packed_params.cu_seqlens, output_final_state, packed_params.cu_seqlens_list,
        )

    # Execute the real provider method; only dispatcher selection is stubbed.
    provider_path = ROOT / "ops/providers/npu/kda.py"
    provider_tree = ast.parse(provider_path.read_text())
    provider_class = next(node for node in provider_tree.body
                          if isinstance(node, ast.ClassDef) and node.name == "AscendCKDA")
    provider_class.decorator_list = []
    provider_namespace = dict(OpImpl=object, torch=torch)
    exec(compile(ast.Module(body=[provider_class], type_ignores=[]),
                 str(provider_path), "exec"), provider_namespace)
    provider = provider_namespace["AscendCKDA"]()

    def dispatch(op_name, *args, **kwargs):
        impl = kwargs.pop("impl", None)
        if op_name == "causal_conv1d":
            assert impl is None
            return causal_conv1d(x=args[0], weight=args[1], bias=args[2], **kwargs)
        assert op_name == "kda"
        if kwargs.get("cp_context") is not None:
            assert impl == "ascendc_kda"
        return provider(*args, **kwargs)

    # Keep the real facade signatures and metadata normalization in this test.
    dispatcher_path = ROOT / "ops/dispatcher.py"
    dispatcher_tree = ast.parse(dispatcher_path.read_text())
    dispatcher_class = next(node for node in dispatcher_tree.body
                            if isinstance(node, ast.ClassDef) and node.name == "OpDispatcher")
    dispatcher_class.body = [node for node in dispatcher_class.body
                             if isinstance(node, ast.FunctionDef)
                             and node.name in ("kda", "causal_conv1d")]
    dispatcher_namespace = dict(torch=torch, Any=Any, PackedParams=packed.PackedParams,
                                get_packed_params=packed.get_packed_params)
    exec(compile(ast.Module(body=[dispatcher_class], type_ignores=[]),
                 str(dispatcher_path), "exec"), dispatcher_namespace)
    dispatcher = dispatcher_namespace["OpDispatcher"]()
    dispatcher._dispatch = dispatch

    sys.modules["torchturbo.ops"] = SimpleNamespace(ops=dispatcher)
    sys.modules["torchturbo.ops.providers.npu._fla_npu_kda"] = adapter
    sys.modules["fla.modules.fused_norm_gate"] = SimpleNamespace(rms_norm_gated=norm_gate)
    return load_file("bridge_attention", "models/bailing_hybrid/attention.py"), packed


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode, self.no_kda_lora = "chunk", True
        self.num_heads, self.head_dim, self.head_k_dim = 4, 128, 128
        self.safe_gate, self.lower_bound = True, -5.0
        for name in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj"):
            setattr(self, name, nn.Linear(64, 512, bias=False))
        self.b_proj = nn.Linear(64, 4, bias=False)
        self.o_proj = nn.Linear(512, 64, bias=False)
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            conv = nn.Conv1d(512, 512, 4, groups=512, bias=True)
            conv.activation = "silu"
            setattr(self, name, conv)
        self.A_log = nn.Parameter(torch.zeros(4))
        self.dt_bias = nn.Parameter(torch.full((512,), -5.0))
        self.o_norm = nn.Module()
        self.o_norm.weight = nn.Parameter(torch.ones(128))
        self.o_norm.activation, self.o_norm.eps = "silu", 1e-6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel-mode", choices=("none", "allgather"), default="allgather")
    parser.add_argument("--reference-mode", choices=("single", "ulysses"), default="ulysses")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.npu.matmul.allow_hf32 = False
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        group = dist.new_group(list(range(size)))
        adapter = load_adapter()
        attention, _ = install_test_imports(adapter)

        class ReferenceLayer(Layer):
            pass

        class ParallelLayer(Layer):
            pass

        torch.manual_seed(19)
        reference = ReferenceLayer().to(device="npu", dtype=torch.bfloat16)
        parallel = ParallelLayer().to(device="npu", dtype=torch.bfloat16)
        parallel.load_state_dict(reference.state_dict())
        distributed_reference = args.reference_mode == "ulysses"
        attention.patch_kda_forward(
            ReferenceLayer, group if distributed_reference else None,
            size if distributed_reference else 1, rank if distributed_reference else 0,
        )
        attention.patch_kda_forward(
            ParallelLayer, group, size, rank,
            use_fla_kcp_ascend=args.parallel_mode == "allgather",
        )
        gen = torch.Generator().manual_seed(41)
        source = torch.randn(1, 512, 64, generator=gen).bfloat16().to("npu")
        do = torch.randn(1, 512, 64, generator=gen).bfloat16().to("npu")
        lo, hi = rank * 512 // size, (rank + 1) * 512 // size
        # get_packed_params appends 512 as an isolated padding sequence.
        offsets = torch.tensor([0, 291, 482], device="npu", dtype=torch.int64)
        x_ref = (source[:, lo:hi] if distributed_reference else source).detach().contiguous().requires_grad_()
        x_cp = source[:, lo:hi].detach().contiguous().requires_grad_()
        expected, _, _ = reference(x_ref, cu_seqlens=offsets)
        actual, _, _ = parallel(x_cp, cu_seqlens=offsets)
        expected.backward(do[:, lo:hi].contiguous() if distributed_reference else do)
        actual.backward(do[:, lo:hi].contiguous())
        rows = []

        def check(name, a, b):
            a, b = a.float(), b.float()
            rel = ((a - b).norm() / b.norm().clamp_min(1e-8)).item()
            max_abs = (a - b).abs().max().item()
            # End-to-end BF16 rounding includes conv, norm, and linear layers.
            passed = rel < 0.03 and torch.allclose(a, b, atol=0.05, rtol=0.05)
            outside = ((a - b).abs() > 0.05 + 0.05 * b.abs()).sum().item()
            rows.append(dict(name=name, passed=bool(passed), rel_l2=rel,
                             max_abs=max_abs, outside_tolerance=outside, numel=a.numel()))

        check("output", actual, expected if distributed_reference else expected[:, lo:hi])
        check("hidden_grad", x_cp.grad, x_ref.grad if distributed_reference else x_ref.grad[:, lo:hi])
        for (name, p), (ref_name, p_ref) in zip(parallel.named_parameters(), reference.named_parameters()):
            assert name == ref_name and p.grad is not None
            # Test-only SUM reconstructs the global gradient; no FSDP is running.
            gradient = p.grad.float()
            dist.all_reduce(gradient, group=group)
            expected_gradient = p_ref.grad.float()
            if distributed_reference:
                dist.all_reduce(expected_gradient, group=group)
            check(name, gradient, expected_gradient)
        print(json.dumps(dict(rank=rank, reference=args.reference_mode, results=rows)), flush=True)
        ok = torch.tensor(int(all(row["passed"] for row in rows)), device="npu")
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if not ok.item():
            raise AssertionError("KDA model bridge parity failed.")
        if rank == 0:
            print("KDA model bridge parity PASS (PyTorch norm; no FSDP)", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
