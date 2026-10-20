# KDA CP backends: AscendC AllGather and FLA Triton

This branch adds an opt-in KDA context-parallel path on top of
`feat/ascendc-chunk-kda`, baseline
`b773bf4fa9581782246a970db5d461d12deabe11`.
The branch name is `feat/ascendc-chunk-kda-cp`.

Customer commit `d154f184f98a4e9611ca28962e80e53baffe1b36` from
`Cruise20/TorchTurbo` adds the FLA Triton KCP path. Its configuration and
convolution CP support are integrated alongside the AscendC path.

## Choose a backend

| Mode | `parallel_plan.use_fla_kcp` | `TORCHTURBO_KDA_CP` |
| --- | --- | --- |
| Baseline Ulysses | `false` | unset or `none` |
| FLA Triton KCP | `true` | unset or `none` |
| AscendC AllGather CP | `false` | `allgather` |

Both CP modes reuse the existing SP group and size. Enabling both switches
raises before either communication path is installed. With `sp_size=1`, the
FSDP backend disables KCP and preserves the normal single-rank dispatch.

The customer option supports the same public inputs as the source commit:

```yaml
engine_config:
  parallel_plan:
    sp_size: 2
    use_fla_kcp: true
```

Equivalent explicit CLI selection is `--tt_use_fla_kcp true`; the environment
input is `TORCH_TURBO_USE_FLA_KCP=true`. Existing precedence is CLI > file >
environment > argument defaults. Set the AscendC switch to `none` or unset
it when selecting Triton.

Triton KCP builds FLA's context from global packed offsets, runs local
full-head convolution with CP halo exchange, and dispatches KDA explicitly
to `fla_kda`. AscendC keeps the Ulysses convolution bridge and dispatches
KDA explicitly to `ascendc_kda`. Neither mode silently falls back to the
other. Both Bailing hybrid and Bailing V3 VL forward the choice.

The Triton model path passes raw beta logits to the native provider, which
applies sigmoid exactly once. Unlike the AscendC convolution bridge, FLA's
KDA/conv path does not shard heads. Other model attention layers may still
impose their own SP divisibility constraints.

## Enable AscendC AllGather

```bash
export TORCHTURBO_KDA_CP=allgather
# Launch the existing training command with sp_size=2, 4, or 8.
```

Set the switch before the model parallelization hook runs. It reuses the
previous CP bridge's switch and the existing `sp_group`/`sp_size`; there is
no new engine configuration or independent CP mesh. The old
`TORCHTURBO_KDA_ASCENDC` switch is not needed by this baseline, which uses
`ops.kda`. CP explicitly passes `impl="ascendc_kda"` to the dispatcher.
A missing AscendC provider fails instead of falling back to Triton with an
incompatible CP context.

Unset `TORCHTURBO_KDA_CP` or set it to `none` to use the baseline Ulysses
path. With `sp_size=1`, the baseline single-rank dispatch is unchanged.
The hook is shared by Bailing hybrid text and Bailing V3 VL. MLA is unchanged.

Use a `fla_npu` wheel containing the split KDA forward/backward and CP
pre-process/merge operators. The baseline and this bridge use FLAN source
snapshot `fa5493f10e0bebe7a96ebcc81a4acf81d79f1e12`.

## Scope and data flow

The AscendC integration uses the shared attention hook, KDA wrapper,
autograd function, `cp.py` state-exchange helper, and dispatcher facade.
The facade forwards `cp_context` and `cu_seqlens_cpu`, and uses Q's sequence
axis (axis 2 for head-major inputs) when normalizing packed metadata.

```text
local tokens, all heads
  -> existing asynchronous Q/K/V AllToAll and full-sequence convolution
  -> inverse AllToAll: local tokens, all heads
  -> KDA Prepare (Q/K L2 norm, beta sigmoid, gate fused)
  -> forward summary AllGather -> same-sequence predecessor merge
  -> FwdH(initial_state=FP32 h0) -> Finalize(output_layout=BSND)
  -> local output gate, norm and projection
```

The baseline's projection/communication overlap and per-convolution event
waits remain in place. CP skips the gate/beta sequence gathers and keeps
the full-head `A_log` and `dt_bias`. Backward adds summary AllGather and a
same-sequence successor merge before `Dhu(dht=...)`.

Forward `h0` must remain FP32: FwdH selects recurrence precision from its
initial-state dtype, even for zero states. Backward `dht` remains BF16,
as required by the Dhu ABI. The baseline's FP32 gate conversion outside
`Function.apply` is preserved so autograd restores the original gate dtype.

Convolution consumes global packed metadata, including an isolated padded
tail. KDA consumes the intersection of each sequence with this rank's equal,
contiguous token shard. Internal sequence boundaries reset state. Only the
first local sequence receives forward state and the last receives backward
state. Collectives use explicit groups and group-relative ranks; all ranks
participate even if their own merge has zero inputs.

The current bridge requires B=1, BF16, key/value dimensions 128, and
`num_heads % sp_size == 0` because convolution still uses head sharding.
It keeps `disable_recompute=True` and all three fusions enabled. CP does not
expose global final states or accept user-supplied initial states.
Parameter gradient reduction remains the responsibility of FSDP/DDP.

This is a connectivity implementation. It retains full-sequence convolution
tensors and adds inverse QKV AllToAll; it does not implement convolution halo
exchange or an independent CP size that is unrestricted by the head count.

## Validation

```bash
python -m pytest tests/ops/test_kda_cp_metadata.py -q
torchrun --standalone --nproc_per_node=2 tests/ops/run_kda_allgather_cp.py
torchrun --standalone --nproc_per_node=4 tests/ops/run_kda_allgather_cp.py
torchrun --standalone --nproc_per_node=4 tests/ops/run_kda_allgather_cp.py --interleaved-groups
torchrun --standalone --nproc_per_node=2 tests/ops/run_kda_cp_model_bridge.py
torchrun --standalone --nproc_per_node=2 tests/ops/run_kda_cp_model_bridge.py --reference-mode single
torchrun --standalone --nproc_per_node=2 tests/ops/run_kda_cp_model_bridge.py --parallel-mode none
```

Operator checks compare output and all seven gradients with single-rank
AscendC for dense, crossing packed, aligned packed, and empty-segment cases.
They require elementwise `atol=rtol=0.02` and relative L2 below 0.03.
The dense first-shard output additionally has an exact-equality assertion
to guard FwdH's FP32 initial-state contract.

The layer fixture executes this branch's actual attention forward, Ulysses
communication functions, KDA/conv dispatcher facades, NPU KDA provider method,
and AscendC convolution/KDA. It substitutes provider selection and runtime plumbing for isolated
imports, plus a PyTorch RMSNorm gate for the optional FLA Triton norm.
It compares output, input gradients and projection/convolution/gate parameter
gradients against distributed Ulysses or a single-rank layer.
Test-only parameter SUM reductions are not part of production code.

The initial AscendC migration was validated on 2026-10-29:

| Check | Result |
| --- | --- |
| CPU packed metadata | 10 passed |
| Two-rank operator parity | 64/64 tensor checks passed |
| Four-rank operator parity | 128/128 tensor checks passed |
| Groups `[0, 2]` and `[1, 3]` | 128/128 tensor checks passed |
| Two-rank layer vs Ulysses | 36/36 tensor checks passed |
| Two-rank layer vs single-rank | 36/36 tensor checks passed |
| CP-disabled Ulysses control | 36/36 tensor checks exactly equal |

In the layer comparison against Ulysses, output and `o_proj.weight` gradients
are exactly equal on both ranks. The maximum relative L2 across all checked
layer tensors is 0.0044033. The single-rank layer reference also passes, but
its parameter-gradient accumulation order differs and is not bitwise equal.
Operator checks reach a maximum relative L2 of 0.0060337. Tolerances have
not been relaxed from the previous bridge. Python compilation, Ruff error
checks and `git diff --check` also pass.

During the Triton cherry-pick, real dispatcher tests exposed that the previous
layer fixture bypassed the KDA facade: it lacked both CP keyword arguments.
The facade is now fixed and covered by CPU tests and the updated NPU fixture.
The head-major metadata test also covers more heads than tokens, which detects
accidental padding against the head axis.

The updated two-rank NPU fixture passes all 36 checks against Ulysses and
all 36 against the single-rank layer, now including the actual facades.
Output and `o_proj.weight` gradient remain exactly equal to Ulysses on both
ranks. The CP-disabled control is also exactly equal (36 checks). The
two-rank standalone operator regression passes all 64 checks.

The combined CPU regression passes 268 tests, with 6 skips and 3 expected
failures. It covers dispatcher/provider contracts, dense/packed/padded KCP
metadata, single sigmoid, full-head local shapes, LoRA and non-LoRA gate
projections, mutually exclusive backend switches, config precedence, and
config propagation through the engine/backend/model plan. FLA kernels are
stubbed in CPU contract tests; these are not Triton numerical validation.
The CPU test environment uses Python 3.12, torch 2.10.0+cpu and transformers
5.21.0 in an isolated local environment.

The operator environment on the NPU server lacks FLA/Triton. Triton CP
forward/backward numerical and performance validation remains outstanding;
its results cannot be inferred from the AscendC regression.

Hardware validation on `179.166.1.3` uses Ascend910B3. These tests do not
certify full A3 model training, FSDP, activation checkpointing, the actual
FLA norm, optimizer steps, or long-sequence gradient accuracy/performance.
The server's operator environment is Python 3.10 / torch 2.7.1; source-isolated
fixtures do not meet the full model's Python >=3.11 / torch >=2.10 requirements.
