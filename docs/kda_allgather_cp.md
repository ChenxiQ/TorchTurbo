# KDA context parallelism on the fused-training baseline

Base: `feat/ascendc-kda-fused-training` at
`4f42bd2`. This branch is `feat/ascendc-kda-fused-training-cp`.
The FLAN interface target is PR #998's merge commit
`d77c37382aa8cf53e25005615c8c3f761fc137d2`.

Customer commit `d154f184f98a4e9611ca28962e80e53baffe1b36` is cherry-picked
for FLA Triton KCP. Its KDA call uses this baseline's dispatcher name,
`ops.chunk_kda(..., impl="fla_chunk_kda")`. Its local convolution halo
exchange and external beta sigmoid are retained.

## Enable a backend

| Mode | `use_fla_kcp` | `use_fla_kcp_ascend` |
| --- | --- | --- |
| Baseline Ulysses | false | false |
| Customer FLA Triton KCP | true | false |
| AscendC AllGather KCP | false | true |

The switches belong to `engine_config.parallel_plan`, default to false,
and are mutually exclusive. Both reuse the existing SP group and size.
The backend keeps normal single-rank dispatch at `sp_size=1`.
Both Bailing text and Bailing V3 VL propagate the switches.

AscendC retains the base branch's opt-in environment variable. Set it
before the engine initializes, including when selecting CP explicitly:

```bash
export TORCHTURBO_KDA_ASCENDC=1
```

```yaml
engine_config:
  parallel_plan:
    sp_size: 2
    use_fla_kcp: false
    use_fla_kcp_ascend: true
```

CLI equivalents are `--tt_use_fla_kcp` and `--tt_use_fla_kcp_ascend`,
each taking `true` or `false`. The corresponding config environment
inputs are `TORCH_TURBO_USE_FLA_KCP` and `TORCH_TURBO_USE_FLA_KCP_ASCEND`.
The existing `TORCHTURBO_KDA_ASCENDC` enables the AscendC provider; the
parallel-plan boolean selects its AllGather CP path. Triton KCP explicitly
selects FLA even when the AscendC provider is enabled.

## AscendC data flow and requirements

The base branch's QKV projection, Ulysses exchange and full-sequence
convolution remain shared. An inverse QKV exchange restores all heads on
each rank's local tokens before KDA. Gate and beta remain local, with full
`A_log` and `dt_bias` rather than head slices.

Forward uses fused Prepare, summary AllGather, predecessor merge, FwdH,
and BSND Finalize. Backward uses shared Prepare, summary AllGather and
successor merge, then Dhu and Finalize. Only AllGather is implemented for
AscendC; there is no PTP selector.

CP keeps L2 normalization, beta sigmoid and gate fusion, with
`disable_recompute=True`. Forward boundary states remain FP32; backward
boundary gradients use the Dhu ABI's BF16 format. The non-CP fused
forward/backward remain in the same autograd adapter.

Inputs must be B=1, BF16 Q/K/V, K=V=128, with equal contiguous token shards.
Convolution still requires `num_heads % sp_size == 0`. Packed boundaries
are global at model entry and intersected locally for KDA, including the
isolated padding tail. Internal sequence boundaries reset recurrence.
The dynamic `chunk_kda` facade forwards local CPU metadata directly, without
building generic packed chunk indices. CP exposes no global final state
and accepts no external initial state. FSDP/DDP owns parameter reduction.

## Validation entry points

```bash
PYTHONPATH=src python -m pytest tests/ops/test_kda_cp_metadata.py tests/ops/test_kcp_backends.py tests/ops/test_kda_dispatch.py -q
torchrun --standalone --nproc_per_node=2 tests/ops/run_kda_allgather_cp.py
torchrun --standalone --nproc_per_node=2 tests/ops/run_kda_cp_model_bridge.py
```

CPU tests use stubbed FLA kernels. The standalone NPU layer fixture executes
actual model/communication/provider source with PyTorch gated RMSNorm and
test-only parameter reductions. These are not full-model FSDP or Triton
numerical acceptance tests.
