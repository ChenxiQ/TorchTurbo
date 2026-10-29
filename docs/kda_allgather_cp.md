# KDA CP backends: AscendC AllGather and FLA Triton

This branch adds an opt-in KDA context-parallel path on top of
`feat/ascendc-chunk-kda`, baseline
`b773bf4fa9581782246a970db5d461d12deabe11`.
The branch name is `feat/ascendc-chunk-kda-cp`.

Customer commit `d154f184f98a4e9611ca28962e80e53baffe1b36` from
`Cruise20/TorchTurbo` adds the FLA Triton KCP path. Its configuration and
convolution CP support are integrated alongside the AscendC path.

## Choose a backend

| Mode | `parallel_plan.use_fla_kcp` | `parallel_plan.use_fla_kcp_ascend` |
| --- | --- | --- |
| Baseline Ulysses | `false` | `false` |
| FLA Triton KCP | `true` | `false` |
| AscendC AllGather CP | `false` | `true` |

Both booleans default to false and reuse the existing SP group and size.
Enabling both raises a configuration error. With sp_size=1, the FSDP backend
preserves normal single-rank dispatch. Both settings flow through the engine
plan, backend SP information and model hook.

CLI flags are `--tt_use_fla_kcp true` and `--tt_use_fla_kcp_ascend true`.
Registered environment inputs are `TORCH_TURBO_USE_FLA_KCP=true` and
`TORCH_TURBO_USE_FLA_KCP_ASCEND=true`. Precedence is CLI > file > environment
> argument defaults. The model no longer reads the old
`TORCHTURBO_KDA_CP=allgather` variable; migrate to the boolean setting.

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

```yaml
engine_config:
  parallel_plan:
    sp_size: 2
    use_fla_kcp: false
    use_fla_kcp_ascend: true
```

Set both booleans to false for the baseline Ulysses path. AscendC CP uses
AllGather internally and explicitly selects `impl="ascendc_kda"`.
The hook is shared by Bailing hybrid text and Bailing V3 VL. MLA is unchanged.

Use a `fla_npu` wheel containing the split KDA forward/backward and CP
pre-process/merge operators. The baseline and this bridge use FLAN source
snapshot `fa5493f10e0bebe7a96ebcc81a4acf81d79f1e12`.

## Scope and data flow

The AscendC integration uses the shared attention hook, KDA wrapper,
autograd function, `cp.py` state-exchange helper, and dispatcher facade.
The facade forwards `cp_context`, `cu_seqlens_cpu`, and CP-local
`cu_seqlens` directly to the provider. The CP adapter validates that the
local offsets match the context and cover Q's local tokens. Without CP,
the facade uses Q's sequence axis (axis 2 for head-major inputs) when
normalizing packed metadata.

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
FLA norm, or optimizer steps. Long-sequence operator measurements and the
remaining gradient accuracy failure are recorded below.
The server's operator environment is Python 3.10 / torch 2.7.1; source-isolated
fixtures do not meet the full model's Python >=3.11 / torch >=2.10 requirements.

## Open precision issue: long packed sequences

Status: unresolved, investigation deferred on 2026-10-29. Do not treat this
branch as having passed full precision acceptance or loosen its tolerances.

Reproducer: case31 with B=1, T=131072, H=32, K=V=128,
global offsets `[0, 70290, 129130, 131072]`, seed 20261027, BF16 Q/K/V,
FP32 raw gate and beta, zero A_log/dt_bias, all three fusions enabled,
and `disable_recompute=True`. Reference: the single-rank AscendC operator,
not an FP64 ground truth. CP partitions tokens into equal contiguous shards.

| Mode | Forward + backward time | Precision against single rank |
| --- | --- | --- |
| No CP | 168.592 ms | Reference |
| AllGather CP=2 | 117.949 ms | 16/16 rank/tensor checks pass |
| AllGather CP=4 | 66.580 ms | 28/32 rank/tensor checks pass |

Times include operator CP communication, but exclude model projections,
convolution, Ulysses AllToAll, FSDP and optimizer steps. They are previous
case31 measurements, not timings of the metadata-cache fix below.

CP=4 fails `dA_log` elementwise tolerance (`atol=rtol=0.02`) on four unique
heads: 13, 20, 25 and 26. Relative L2 is 0.27147%; maximum absolute error is
0.559276. Output and the other gradients pass the configured checks.

Evidence collected so far:

- Rank 3 begins at token 98304, which is 46 tokens past the chunk grid
  anchored at sequence start 70290. Moving only this diagnostic boundary
  to 98322 makes all 32 checks pass. Unequal shards in this experiment are
  not a production fix, and this does not establish accuracy for all lengths.
- Fused normalization outputs and effective beta agree exactly at matching
  tokens. The first observed same-semantic discrepancy is FwdPrepare's Aqk.
  Forward FwdPrepare/FwdH rounding changes propagate into backward dg.
- Reconstructing dA_log from dg agrees with the operator to about 3.5e-5;
  the large discrepancy already exists upstream of the final reduction.
- A continuous-state CPU replay that bypasses all three CP preprocessing/
  merge operators and AllGather reproduces head 20's discrepancy: -0.56104
  versus the measured -0.55928. An FP64 replay reduces the difference to
  about 4.3e-6. This points to chunk-dependent finite-precision arithmetic
  as the main source, not an established CP wiring error. Small contributions
  from CP operators remain unquantified; they have not been fully exonerated.

Detailed local evidence is archived as `kcp-stage-debug-20261029.zip`.
No numerical workaround or kernel change is included in this review.

## Integration review: packed metadata cache

The KDA facade previously converted each newly allocated CP-local offsets
tensor into generic PackedParams. This built unused device chunk indices
and evicted convolution's global metadata from the shared single-entry
cache on every layer. Subsequent layers rebuilt the global metadata and
could repeat device-to-host synchronization for global offsets.

CP-local offsets now go directly to the provider, preserving the global
cache. Non-CP normalization and rejection of conflicting metadata remain
unchanged. A regression test reproduces eviction before the fix and verifies
cache reuse afterward. This is removal of redundant metadata work, not a
measured end-to-end speedup or a fix for the open precision issue.

Validation of this fix: 324 CPU tests passed, 7 skipped and 3 expected
failures. On Ascend910B3, the two-rank actual-source layer fixture passed
36/36 checks against Ulysses and 36/36 against a single-rank layer. The
CP-disabled control passed 36/36 with exact equality. Output and
`o_proj.weight` gradients remain exactly equal to Ulysses. These small
fixture results do not supersede the long-sequence failure above.

The remaining convolution head-divisibility restriction, extra inverse QKV
AllToAll, and equal contiguous token partitioning are explicit limitations
of this connectivity design. The customer's Triton forward is unchanged.
