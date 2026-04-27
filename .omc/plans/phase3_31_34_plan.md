# Phase 3 Plan — 3.1/3.2 Remediation + 3.3/3.4 Implementation

_Date: 2026-04-25. Reference: `docs/training_modeller_zh.md` phase 3._

---

## Scope

This plan separates work into:

1. **Remediation** for partially landed items:
   - 3.1 `ContextParallelPass`
   - 3.2 `DataParallelPass`
2. **New implementation** for not-yet-landed items:
   - 3.3 `StreamAssignPass` CoC/MC2/Ring-CP overlap rules
   - 3.4 Graph-native training memory modeling

Phase 4 integration should consume these outputs after this phase stabilizes.

---

## Current Status Snapshot

### 3.1 Context Parallel
- `ContextParallelPass` sets `cp_split` annotations for `ulysses`/`ring`/`hybrid`.
- `CommInserterPass` inserts CP comm nodes (`all_to_all`, `send_recv` rounds).
- Gap: no explicit message-size modeling (`b*s/cp*h`) on CP comm nodes.
- Gap: Ring overlap uses coarse `overlap_target=node.id`, not FA tile pairing.

### 3.2 Data Parallel
- `DataParallelPass` inserts `comm.all_reduce` (ZeRO-0) or `comm.reduce_scatter` (ZeRO-2/3).
- Pass adds `dp_comm` and optional `overlap_in_bubble`.
- `TrainingPipelinePass` subtracts bubble-hidden DP time.
- Gap: DP pass inserts one tail communication node, not per-parameter/per-gradient placement after backward nodes.

### 3.3 and 3.4
- Not implemented to phase spec level.
- `StreamAssignPass` currently only round-robins streams.
- Memory remains formula-based (`Korthikanti` upper-bound path), not stitched-graph lifetime-based.

---

## Remediation Plan — 3.1

### R1. CP comm metadata completeness
**Files**:
- `python/zrt/transform/parallel/context_parallel.py`
- `python/zrt/transform/parallel/comm_inserter.py`

**Changes**:
- Add `attrs["message_size_bytes"]` to CP comm nodes.
- Ulysses pre/post A2A should include role-specific size estimates.
- Ring P2P rounds should include per-round KV chunk bytes.

### R2. Ring overlap granularity
**Files**:
- `python/zrt/transform/parallel/context_parallel.py`
- `python/zrt/transform/parallel/comm_inserter.py`

**Changes**:
- Replace coarse `overlap_target=node.id` with tile or chunk-level identity:
  - `overlap_target="fa_tile:<tile_id>"` or equivalent stable key.
- Preserve round index and stage/layer info in attrs for downstream scheduler logic.

### R3. CP regression tests
**Files**:
- `tests/training/test_transform_integration.py` (or new `test_context_parallel.py`)

**Assertions**:
- `cp>1, cp_kind=ulysses` inserts pre/post `comm.all_to_all` around attention ops.
- `cp>1, cp_kind=ring` inserts exactly `cp` send/recv rounds per targeted attention op.
- Inserted CP nodes carry `message_size_bytes` and overlap metadata.

---

## Remediation Plan — 3.2

### R4. DP comm placement from gradient producers
**Files**:
- `python/zrt/transform/parallel/data_parallel.py`

**Changes**:
- Stop using one global tail node (`comm_grad_reduce`).
- Build gradient groups from backward nodes that emit grad tensors.
- Insert comm after gradient-producing boundaries (per bucket/group).
- Keep ZeRO mapping:
  - ZeRO-0 -> `all_reduce`
  - ZeRO-2/3 -> `reduce_scatter`

### R5. DP exposed-time accounting uses comm-node latency
**Files**:
- `python/zrt/transform/analysis/training.py`
- optional: `python/zrt/transform/analysis/comm_latency.py`

**Changes**:
- Sum actual DP comm-node latency annotations when available.
- Fallback to alpha-beta formula only if latency annotations are missing.
- Keep `t_exposed_dp = max(0, t_dp - bubble_duration)` contract.

### R6. DP regression tests
**Files**:
- new `tests/training/test_data_parallel.py`
- update `tests/training/test_captured_graph_modelling.py` as needed

**Assertions**:
- DP pass creates expected number/type of comm nodes by ZeRO stage.
- `dp_comm` / `overlap_in_bubble` annotations are present.
- Pipeline step time increases only by exposed DP portion when DP overlap is enabled.

---

## Implementation Plan — 3.3 (CoC/MC2/Ring overlap)

### I1. Extend stream assignment into overlap assignment
**Files**:
- `python/zrt/transform/analysis/passes.py` (`StreamAssignPass`)
- `python/zrt/executor/overlap.py` (if needed)

**Changes**:
- Add overlap annotations and exposed-time coefficients:
  - CoC: `coc_tile_k`
  - MC2: fused AG+matmul marker
  - Ring-CP: paired P2P <-> FA tile marker
- Maintain backward compatibility with current stream IDs.

### I2. Add exposed comm-time helper
**Files**:
- `python/zrt/transform/analysis/training.py` or a new helper module

**Changes**:
- Compute exposed communication time with phase-3 formulas:
  - CoC: `max(0, t_comm - t_matmul*(k-1)/k)`
  - MC2: exposed `0`
  - Ring-CP: `max(0, t_p2p - t_fa_tile)`
- Feed exposed time into stage timing and final step-time calculation.

### I3. Tests for overlap semantics
**Files**:
- new `tests/training/test_stream_overlap.py`

**Assertions**:
- CoC reduces exposed comm relative to no-overlap baseline.
- MC2 path exposes zero AG/RS communication.
- Ring-CP uses paired target and computes reduced exposed time.

---

## Implementation Plan — 3.4 (Graph-native memory)

### I4. Graph-native memory pass on stitched graph
**Files**:
- `python/zrt/transform/analysis/training.py` (`TrainingMemoryPass`) or new `memory_graph.py`
- `python/zrt/ir/adapter.py` (if stitch metadata hooks needed)

**Changes**:
- Parameter memory: sum tensors from nodes with `annotations["is_param"]`.
- Activation memory: compute saved forward tensors that are consumed by backward path.
- Apply recompute reductions from `RecomputePass` annotations.
- Grad/optimizer memory: consume optimizer annotations (`state_bytes`) and ZeRO sharding metadata.

### I5. Inflight microbatch depth by stage
**Files**:
- `python/zrt/transform/analysis/training.py`

**Changes**:
- Replace global `inflight=pp` approximation with stage-aware depth:
  - stage `s` depth approx `pp - s` in 1F1B steady state.
- Keep fallback for missing stage annotations.

### I6. Memory-model regression tests
**Files**:
- new `tests/training/test_memory_graph_native.py`

**Assertions**:
- Activation memory responds to stitched fwd->bwd liveness, not only formula constants.
- Recompute-tagged layers reduce saved activations.
- ZeRO shard factors impact weights/grads/opt-state exactly as expected.
- Stage-aware inflight depth changes peak activation memory by stage.

---

## Execution Order

1. R1-R3 (CP remediation)
2. R4-R6 (DP remediation)
3. I1-I3 (overlap semantics)
4. I4-I6 (graph-native memory)
5. Re-run phase-2 and phase-4 reference tests for regression safety

---

## Verification Gate

Minimum gate after 3.1/3.2 remediation:

```bash
PYTHONPATH=python pytest tests/training/test_transform_integration.py -q
PYTHONPATH=python pytest tests/training/test_captured_graph_modelling.py tests/training/test_pipeline_parallel.py -q
```

Full gate after 3.3/3.4 implementation:

```bash
PYTHONPATH=python pytest tests/training -q
```

Plus targeted suites if split out:

```bash
PYTHONPATH=python pytest tests/training/test_context_parallel.py tests/training/test_data_parallel.py tests/training/test_stream_overlap.py tests/training/test_memory_graph_native.py -q
```

---

## Risks and Adaptation Notes

1. EP world-size semantics are still under decision; keep policy pluggable and avoid hard-coding assumptions in phase-3 tests.
2. Overlap formulas depend on scheduler data quality; keep fallbacks for missing latency annotations.
3. Graph-native memory requires stable tensor identity across stitch/fusion boundaries; enforce invariants in tests.
4. Phase 4 should only tighten anchor gates after phase-3 communication/memory behavior is validated.
