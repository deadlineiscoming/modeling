# Phase 0 Improvement Plan — `stitch_fwd_bwd` Hardening

_2026-04-23. Follow-up to the Phase 0 landing in `python/zrt/ir/adapter.py` (SESSION_PROGRESS.md)._

Phase 0 met the structural intent (unified graph, phase labels, param flags, wired into `estimate_training_from_graphs`) and 65/65 training tests pass. This document captures four issues that should be resolved **before** building Phase 2 (PP stage assignment) on top of the stitched graph — otherwise those downstream passes will inherit silent accuracy bugs.

---

## Issue 1 — Cross-graph edges use heuristic (shape, dtype) matching, not tensor IDs

### Problem

`stitch_fwd_bwd` at `adapter.py:683-712` matches bwd input tensors to fwd output tensors by `(shape, dtype)` with a same-layer tiebreaker. The code comment (lines 619-621) acknowledges this is because `train_forward` and `train_backward` use independent `TensorTracker` instances — real tensor IDs are not comparable across phases.

In practice, within a single transformer layer there are many fwd tensors of shape `[b, s, h]` with the same dtype (`x_ln1_out`, `x_attn_out`, `x_ln2_out`, `x_ffn_out`, ...). The `_best_cross_match` function picks the first same-layer candidate, which may not be the correct producer.

### Impact on downstream phases

- **Phase 3.4 memory (activation liveness)**: aggregate activation-memory estimate is approximately right because any `[b, s, h]` bf16 tensor in the layer *is* an activation that must be saved. So total bytes are close.
- **Phase 2 PP scheduling**: wrong edges can cross stage boundaries, creating spurious `stage_i → stage_j` dependencies in the unified graph. When the stage-view is extracted and run through `DAGScheduler`, these false dependencies silently serialize ops that should parallelize. Per-stage timelines will be too long, step time will be overestimated.
- **Phase 3.1–3.2 CP/DP collective insertion**: passes that scan for "bwd consumers of fwd outputs" to insert gradient collectives may target the wrong nodes.

### Recommended fix (primary): share the TensorTracker across phases

Modify the capture layer to use **one `TensorTracker` instance** for both `train_forward` and `train_backward`. This makes tensor IDs globally unique within a training step and enables exact matching in `stitch_fwd_bwd`.

**Implementation steps:**
1. Read `python/zrt/graph/main.py::_trace_phase` and `python/zrt/graph/dispatch.py` (tracker creation point).
2. If tracker is created per-phase, hoist creation to `run_trace_phases` and thread the same instance into both phase calls.
3. Verify that `RecordingDispatch` doesn't reset tracker state at phase boundaries (it shouldn't — but check).
4. In `stitch_fwd_bwd` Phase 5 (adapter.py:683-712), replace the `(shape, dtype) → candidates` lookup with direct `tensor_id → (producer, slot)` matching. Keep the shape+dtype fallback for robustness when `tensor_id` is `None`.

**Acceptance criteria:**
- New test `test_stitch_cross_edges_use_tensor_ids`: captures a 2-layer model, asserts that every cross-graph edge has a `tensor_id` and that the producer-slot matches the actual fwd op that produced the consumed tensor.
- Run on Llama-7B (2 layers, `trust_remote_code=False`) and assert no false cross-stage edges when `stage_id` annotations are added (synthetic stage_id assignment for the test).

### Fallback fix: tighten the heuristic

If shared-tracker capture is not feasible, keep shape+dtype matching but add:
- **Producer scope proximity**: prefer candidates whose `node.scope` is a prefix of the bwd consumer's scope (when bwd preserves scope info).
- **Topological position in layer**: for multiple candidates in the same layer, prefer the last-produced one (LIFO — most recently computed tensor is most likely to be the immediate input to the gradient op).
- **Producer op_type constraint**: matmul outputs, LN outputs, and activation outputs have different semantic roles; the bwd consumer's op_type constrains which fwd op_type can be its producer (e.g., `aten.mm.default` backward typically consumes another matmul output or an activation output — not a slice/reshape result).

**Files to modify:**
- `python/zrt/ir/adapter.py:721-737` (replace `_best_cross_match`)

---

## Issue 2 — Param detection misses non-Llama naming & embedding ops

### Problem

`_PARAM_SCOPE_SUFFIXES` (adapter.py:580-586) is a hardcoded list of Llama/DeepSeek-style names (`q_proj`, `kv_a_proj`, etc.). `_is_param_node` also only matches nodes with `op_type in _MATMUL_OPS`. Two concrete miss cases:

- **Embedding tables**: `aten.embedding.default` is not in `_MATMUL_OPS`, so `embed_tokens` nodes are never flagged `is_param=True` — despite `embed_tokens` appearing in `_PARAM_SCOPE_SUFFIXES`. These are often the single largest parameter tensor in the model (vocab × hidden).
- **Non-Llama architectures**: GPT-2 uses `c_attn`, `c_proj`, `c_fc`; Qwen2 MoE uses different router names; custom models use anything. These will be silently uncounted by Phase 3.4's weight accounting.

### Recommended fix

Replace scope-suffix matching with a more general rule:

1. **Detect params by the graph structure, not by name.** A parameter is any tensor whose producer is a `get_attr`-style node (weight load), or equivalently, a tensor consumed by a matmul/embedding/conv that has no producer within the captured graph (i.e., it's an input to the graph, not an intermediate).
2. **Use `module_class` instead of scope suffix** when a name check is still desired. The fusion layer already maps `module_class` → semantic labels; extend this to `is_param_bearing`.
3. **Include embedding ops** in the classifier: `aten.embedding.default` is a param-reading op. Also `aten._convolution.default` for models that use conv (rare for LLMs but cheap to include).

**Files to modify:**
- `python/zrt/ir/adapter.py:588-600` (generalize `_is_param_node`)
- Optionally: `python/zrt/graph/fusion_rules.py` (extend `SEMANTIC_LABELS` or add a new `PARAM_BEARING_MODULES` set)

**Acceptance criteria:**
- New test `test_stitch_param_detection_covers_embedding`: assert embedding node is flagged `is_param=True`.
- New test `test_stitch_param_detection_gpt_style`: synthesize a graph with `c_attn` scope and assert it is flagged (if going down the module_class route, test `GPT2Attention`).

---

## Issue 3 — bwd-phase metadata dropped

### Problem

`adapter.py:639-643`:
```python
metadata={
    **fwd_graph.metadata,
    "fwd_graph_name": fwd_graph.name,
    "bwd_graph_name": bwd_graph.name,
},
```

`bwd_graph.metadata` is not merged in. If any downstream capture-side annotation lives only on the bwd graph (e.g., a phase-specific `num_layers_traced` override, or a bwd-only fusion summary), it's silently dropped.

### Recommended fix

Merge bwd metadata with fwd taking precedence on key conflicts, and preserve each side's metadata under a namespaced key for debugging:

```python
metadata={
    **bwd_graph.metadata,   # bwd first so fwd wins on conflicts
    **fwd_graph.metadata,
    "fwd_graph_name": fwd_graph.name,
    "bwd_graph_name": bwd_graph.name,
    "fwd_metadata": dict(fwd_graph.metadata),
    "bwd_metadata": dict(bwd_graph.metadata),
},
```

**Files to modify:**
- `python/zrt/ir/adapter.py:636-644`

**Acceptance criteria:**
- New test `test_stitch_preserves_both_metadata`: set distinct keys in fwd.metadata and bwd.metadata, assert both are accessible in the stitched graph.

---

## Issue 4 — Test coverage gap: cross-graph edge correctness on multi-layer graphs

### Problem

The current `test_stitch_cross_graph_edges` at `test_captured_graph_modelling.py:512` only asserts that cross edges exist (`assert len(cross) > 0`). It does not verify that the edges are semantically correct — i.e., that each bwd node's matched fwd producer is actually the tensor the bwd op should consume.

### Recommended fix

Add two tests:

1. **`test_stitch_cross_edges_within_same_layer`** — build a synthetic 2-layer graph where layer 0 has scope `model.layers.0.*` and layer 1 has `model.layers.1.*`. Assert that a bwd node with `layer="1"` never has a cross-graph edge from a fwd node with `layer="0"` when a same-layer fwd candidate exists.

2. **`test_stitch_cross_edges_match_tensor_id_after_fix`** — once Issue 1 is fixed, assert that `edge.tensor_id is not None` for every cross-graph edge, and that `edge.src` matches the actual producer (verify via a controlled capture with known tensor-producer mapping).

**Files to modify:**
- `tests/training/test_captured_graph_modelling.py` (append new tests)

---

## Priority & Ordering

| Issue | Severity | Blocks | Effort |
|---|---|---|---|
| 1. Cross-graph matching heuristic | **High** — silent correctness bug | Phase 2 scheduling, Phase 3 CP/DP passes | Medium (capture-layer change) or Low (heuristic tightening) |
| 2. Param detection completeness | Medium — under-counts weights for non-Llama models and all embeddings | Phase 3.4 memory accuracy | Low |
| 3. Metadata merge | Low — defensive hygiene | None currently | Trivial |
| 4. Test coverage | Medium — gates confidence in fixes 1 and 2 | PR approval of fixes 1 and 2 | Low |

**Recommended order:**
1. Fix Issue 1 via shared `TensorTracker` (unblocks everything else).
2. Add Issue 4 tests against the new tensor-ID edges.
3. Fix Issue 2 generalization + embedding inclusion.
4. Fix Issue 3 metadata merge (can go in same PR as 2 or 3).
5. Re-run full training test suite; expect 65+ passing, zero regressions.

Only after these land, proceed to Phase 1 correctness bug fixes (per `docs/training_modeller_zh.md`: step-time formula, activation memory, total FLOPs).

---

## Critical Files

| File | Role |
|---|---|
| `python/zrt/ir/adapter.py` | `stitch_fwd_bwd`, `_is_param_node`, `_best_cross_match` — all fixes land here |
| `python/zrt/graph/main.py` | `_trace_phase`, `run_trace_phases` — shared-tracker hoist (Issue 1 primary fix) |
| `python/zrt/graph/dispatch.py` | `TensorTracker` creation — verify no per-phase reset |
| `tests/training/test_captured_graph_modelling.py` | Add Issue 4 tests |

---

## Follow-up Review (post-fix verification)

_Added after reviewing the landed fixes against the issues above._

### Status Summary

| Issue | Status | Evidence |
|---|---|---|
| 1. Cross-graph tensor-ID matching | ✅ Code fixed | Shared `TensorTracker` wired in `graph/main.py:868-894`; `stitch_fwd_bwd` uses two-strategy lookup in `adapter.py:695-742` (tensor ID primary, shape+dtype fallback) |
| 2. Param detection (embeddings/conv) | ✅ Fixed | `_PARAM_READ_OPS` includes `aten.embedding.default` + `aten._convolution.default` at `adapter.py:589-594`; `_is_param_node` bypasses scope check for these op types |
| 2b. Non-Llama matmul naming (GPT-2, etc.) | ⚠ Still not handled | `_PARAM_SCOPE_SUFFIXES` unchanged — hardcoded Llama/DeepSeek names only |
| 3. Metadata merge | ✅ Fixed | `adapter.py:648-655` matches the recommended pattern exactly |
| 4. Test coverage (new tests landed) | ✅ Added | `test_stitch_cross_edges_within_same_layer`, `test_stitch_preserves_both_metadata`, `test_stitch_param_detection_covers_embedding` in `test_captured_graph_modelling.py:556-666` |
| 4b. Tensor-ID path test | ⚠ Missing | No test exercises the new primary strategy — all existing tests build synthetic graphs where fwd and bwd tensor IDs never overlap, so they exercise only the fallback path |

### Remaining Follow-ups

#### Follow-up A — Add a tensor-ID match test (closes Issue 4 gap)

**Why**: The shared-tracker change is the most impactful correctness fix in Phase 0. Its unit-test coverage is currently zero — `test_stitch_cross_graph_edges` and `test_stitch_cross_edges_within_same_layer` both construct fwd and bwd graphs with independently-allocated `TensorMeta.id` values (e.g., fwd output `t5` but bwd input `t17`), so they never hit the `if tmeta.id in fwd_id_index` branch at `adapter.py:715`.

**Fix**: Add a new test that shares a tensor ID across fwd and bwd:

```python
def test_stitch_cross_edges_use_tensor_ids():
    """Cross-graph edges must use exact tensor-ID matching when IDs align."""
    from zrt.ir.adapter import stitch_fwd_bwd

    # Two fwd nodes produce tensors with the SAME shape/dtype (ambiguous by
    # fallback heuristic), but bwd consumes the second one via its exact
    # tensor ID. The stitcher must pick the second, not the first.
    t_shared_id = "t_grad_target"
    t_other = TensorMeta(id="t_other", shape=(2048, 4096),
                         dtype=DType.BF16, mem_bytes=2048*4096*2)
    t_target = TensorMeta(id=t_shared_id, shape=(2048, 4096),
                          dtype=DType.BF16, mem_bytes=2048*4096*2)

    fwd_nodes = {
        "fwd_A": OpNode(id="fwd_A", op_type="aten.mm.default",
                        inputs=[], outputs=[t_other],
                        scope="model.layers.0.self_attn.q_proj", layer="0"),
        "fwd_B": OpNode(id="fwd_B", op_type="aten.mm.default",
                        inputs=[], outputs=[t_target],
                        scope="model.layers.0.self_attn.k_proj", layer="0"),
    }
    fwd = OpGraph(name="fwd", phase="train_forward",
                  nodes=fwd_nodes, edges=[],
                  metadata={"seq_len": 2048, "hidden": 4096})

    # Backward consumes the target tensor by the SHARED id
    t_bwd_in = TensorMeta(id=t_shared_id, shape=(2048, 4096),
                          dtype=DType.BF16, mem_bytes=2048*4096*2)
    bwd_nodes = {
        "bwd_X": OpNode(id="bwd_X", op_type="aten.mm.default",
                        inputs=[t_bwd_in], outputs=[],
                        scope="model.layers.0.self_attn.k_proj", layer="0"),
    }
    bwd = OpGraph(name="bwd", phase="train_backward",
                  nodes=bwd_nodes, edges=[],
                  metadata={"seq_len": 2048, "hidden": 4096})

    stitched = stitch_fwd_bwd(fwd, bwd)
    cross = [e for e in stitched.edges
             if e.src in fwd_nodes and e.dst == "bwd_bwd_X"]
    assert len(cross) == 1, f"Expected 1 cross-edge, got {len(cross)}"
    assert cross[0].src == "fwd_B", (
        f"Tensor-ID match must pick fwd_B (producer of {t_shared_id}), "
        f"not fwd_A. Got {cross[0].src}."
    )
```

**Acceptance:** This test must fail if the tensor-ID branch in `adapter.py:715` is removed, and pass with the current code.

**File:** `tests/training/test_captured_graph_modelling.py` (append)

---

#### Follow-up B — Broaden matmul param detection beyond Llama/DeepSeek

**Why**: `_PARAM_SCOPE_SUFFIXES` is still a hardcoded Llama/DeepSeek suffix list. Models with different conventions will be silently miscounted by Phase 3.4 weight accounting:

- GPT-2 / GPT-J: `c_attn`, `c_proj`, `c_fc`
- Qwen MoE: expert gate has different naming
- Mistral / Mixtral: uses `w1`, `w2`, `w3` for FFN
- Custom HF models: arbitrary

**Decision gate**: Defer until a non-Llama model is actually traced through the training pipeline and produces a visibly-wrong memory estimate. The current scope (Llama, DeepSeek, Qwen2 dense) is covered. When the time comes, prefer **Option 2** from the original plan (use `module_class` via `fusion_rules.SEMANTIC_LABELS`) over expanding the suffix list — semantic labels already encode this knowledge and are maintained per-platform.

**File:** `python/zrt/ir/adapter.py:580-610` (when triggered)

**Acceptance**: Test `test_stitch_param_detection_gpt_style` asserts a `c_attn` / `c_proj` node is flagged `is_param=True`.

---

### Updated Priority Before Phase 1

1. **Follow-up A** — add tensor-ID match test (blocks claim of Issue 1 correctness)
2. Then proceed to Phase 1 correctness bugs per `docs/training_modeller_zh.md`:
   - Step-time formula (`TrainingPipelinePass`)
   - Activation memory (`TrainingMemoryPass`)
   - Total FLOPs (sum per-node annotations, drop 6P override)
3. **Follow-up B** — defer until needed; triggered by first non-Llama/DeepSeek model trace

Phase 0 is otherwise solid enough to build Phase 2 on top of.
