# Training Modelling: Gap Analysis & Forward-Compatibility Plan

**Date:** 2026-04-25  
**Scope:** Review implementation vs `docs/training_modeller_zh.md`; structural issues for DeepSeek V4; missing features; paper-backed gaps.

---

## Context

`docs/training_modeller_zh.md` describes a 4-phase roadmap for a **graph-native** training modeller where captured `OpGraph` objects drive all performance estimates. As of today (Phases 0–4 committed), the implementation is partially complete. This plan documents what was actually built, what remains broken or absent, and what new work is needed for DeepSeek V4 forward-compatibility.

---

## 1. What the Design Prescribes vs What Exists

### A. `modeller.py` — Integration Bridge (MISSING)

**Design:** `python/zrt/transform/analysis/modeller.py` (lines 307–343) should be the orchestrator: call `stitch_fwd_bwd` → run transform pipeline → extract per-stage timelines → dispatch to pipeline composer.

**Reality:** File does not exist. The project memory lists it as a hot path (48×) — this is stale metadata from planned work that was never committed. The transform passes (`TrainingPipelinePass`, etc.) exist in `training.py`, but there is no single function that wires graph capture → stitching → transform → report.

**Impact:** No end-to-end path from `run_trace_phases("train_forward", "train_backward")` → `TrainingReport` using captured graphs. Stack A (`python/zrt/training/search/estimator.py`) is the only working estimator.

---

### B. `stitch_fwd_bwd` — Implemented but Not Integrated

**Design:** `stitch_fwd_bwd(fwd, bwd) → unified OpGraph` with `metadata["fwd_bwd_stitched"] = True`.

**Reality:**
- Implementation exists: `python/zrt/ir/adapter.py:613–748` — cross-graph edges, `phase="fwd"/"bwd"` annotations, `is_param` detection.
- **Bug:** `stitch_fwd_bwd` never sets `combined.metadata["fwd_bwd_stitched"] = True`.
- **Integration gap:** `stitch_fwd_bwd` is never called in the production training path — only exercised by tests. `graph/main.py:870` only references it in a comment.

**Downstream effect:** `TrainingMemoryPass._graph_native_activations` (`training.py:170`) is gated on `fwd_bwd_stitched` — this path is permanently unreachable. Every training estimate falls back to the Korthikanti formula.

---

### C. `TrainingMemoryPass` — Korthikanti Fallback Issues

**Design:** Graph-native activation liveness once stitched; Korthikanti only as fallback.

**Reality (`training.py:196–223`):**
- Fallback formula: `34 * h * s * L * bs / (tp * cp)` with recompute multiplier.
- Stage-aware inflight depth: implemented (`training.py:207–218`) but only effective when `stage_id` annotations exist.
- **Known issue:** CP divisor `max(cp, 1)` double-divides for some paths — minor arithmetic bug.
- **Structural gap:** Recompute multipliers are hardcoded (`{none: 1.0, selective: 0.5, full: 0.1}`) — not derived from `RecomputePass` annotations.

---

### D. `TrainingPipelinePass` — Substantially Complete

**Status:** `training.py:286–513` — per-stage DAGScheduler, heterogeneous 1F1B formula, VPP/DualPipe schedule adjustments, DP-in-bubble, overlap-aware comm deduction.

**Gaps:**
- VPP/DualPipe branch in the `pp=1` fallback path has a `# TODO Phase 3` note (`training.py:400`).
- `ZERO_BUBBLE` schedule: in `PPSched` enum (`training/spec/strategy.py:30`) but no case branch in `TrainingPipelinePass` and no ZeroBubble composer in `training/compose/`.

---

### E. `ContextParallelPass` + `CommInserterPass` — Split Responsibility

**Design:** CP comm insertion was supposed to be in `CommInserterPass`.

**Reality:**
- `context_parallel.py`: annotation-only (adds `cp_split` dict to attention nodes). No comm nodes inserted here.
- `comm_inserter.py:182–`: `_insert_cp_comm` method **is** implemented — reads `cp_split` annotations and inserts `comm.all_to_all` (Ulysses) and `comm.send_recv` (Ring) nodes. The split-responsibility design is correct.
- **Risk:** Integration depends on `ContextParallelPass` running before `CommInserterPass`. Need to verify pass order in `build_default_pipeline()`.

---

### F. Stack A (`python/zrt/training/`) — Working but Not the Target Path

Stack A is spec-driven and fully functional for `estimate()`. The design doc (`training_modeller_zh.md:132`) explicitly calls it "not the primary path" — but it's the only working path today. Stack A should be retained as validation reference; Stack B (graph-native) is the target.

---

## 2. Verified Gap Table

| Gap | File(s) | Severity | Status |
|-----|---------|----------|--------|
| `modeller.py` missing (no graph→report bridge) | N/A (to create) | **CRITICAL** | Not started |
| `stitch_fwd_bwd` not setting `fwd_bwd_stitched=True` | `ir/adapter.py:648–656` | **CRITICAL** | 1-line fix |
| `stitch_fwd_bwd` never called in production path | `graph/main.py` or new `modeller.py` | **CRITICAL** | Blocked on modeller.py |
| Graph-native activation memory unreachable | `training.py:170` | **HIGH** | Blocked on above |
| `ZERO_BUBBLE` — enum exists, no composer | `training/compose/` | **HIGH** | Not started |
| Compressed attention FLOPs (DeepSeek V4 CSA/HCA) | `flops_train.py:85–97` | **HIGH** | Not started |
| Recompute multiplier not from `RecomputePass` annotations | `training.py:202–203` | **MEDIUM** | Quick fix |
| Anchor `estimate()` integration (`strict_mfu_check`) | `tests/training/anchors/` | **MEDIUM** | Phase 3 TODO |
| EP rank product policy (`ep` in/out of rank product) | `training/spec/strategy.py` | **MEDIUM** | Phase 3 TODO |
| `test_context_parallel.py` is empty | `tests/training/` | **LOW** | Coverage gap |

---

## 3. DeepSeek V4 Structural Issues

DeepSeek V4 (1.6T params, 1M token context) introduces two architectural changes that break current modelling assumptions:

### 3.1 Compressed Attention FLOPs

Current (`flops_train.py:85–97`):
```python
fwd_flops = 2 * batch * (seq_len ** 2) * heads * head_dim
```

DeepSeek V4 uses **Compressed Sparse Attention (CSA)** + **Heavily Compressed Attention (HCA)**: effectively ~27% of standard attention FLOPs at 1M tokens. Without modeling this:
- At `seq_len=1M`, dense attention FLOPs = `2 * b * (1e6)² * h * d` — absurdly overestimates compute
- MFU will appear near zero (denominator blows up)

**Fix required:** Add `attn_compression_ratio` to `ModelSpec` (or derive from model graph metadata). Scale attention FLOPs: `fwd_flops *= compression_ratio` where `compression_ratio ∈ (0, 1]` (default 1.0 = dense, 0.27 for DeepSeek V4).

### 3.2 Long-Context CP Requirements

At 1M sequence length, CP is mandatory (single-device attention is infeasible). The current CP search pruning rule (`search/space.py`: CP only enabled for `seq ≥ 32k`) needs updating — at 1M tokens, CP ≥ 8 is likely required, and the search space should reflect this.

### 3.3 ZeroBubble Schedule

DeepSeek V4 training reportedly uses ZeroBubble-style scheduling to reduce pipeline idle time. `PPSched.ZERO_BUBBLE` is in the enum but has no implementation. For competitive simulation accuracy, this must be added.

**ZeroBubble model** (Qi et al., 2024, arXiv:2401.10241):
- Splits backward into `B` (input gradient) and `W` (weight gradient) phases
- Bubble fraction ≈ 0 in ideal case; practical ~(pp-1)/(M+pp-1) * reduction_factor
- Requires distinguishing `flops_dx` from `flops_dw` per node — already present in `flops_train.py`

---

## 4. Paper-Backed Missing Features

Based on research into LLM training simulation literature:

| Feature | Paper Reference | Gap |
|---------|----------------|-----|
| ZeroBubble schedule | Qi et al. 2024 (arXiv:2401.10241) | No composer; enum only |
| MFU vs HFU distinction | PaLM (Chowdhery 2022); ml-engineering guide | Only MFU tracked; HFU (accounting for recompute) not computed |
| Hierarchical comm topology | Demystifying Comm (arXiv:2408.10197) | Single alpha-beta tier; multi-rack not modeled |
| MoE expert load imbalance | DeepSeek-V3 tech report §4 | `expert_imbalance` in design §4.3 but not in `flops_train.py` |
| Compressed attention | DeepSeek V4 tech report | Hardcoded quadratic formula |
| Gradient bucketing | PyTorch DDP docs | DataParallelPass: one bucket per layer; no size-based bucketing |

---

## 5. Proposed Work Items (Prioritized)

### P0 — Unlock Graph-Native Path (2 files, low risk)

**5.1** Add `combined.metadata["fwd_bwd_stitched"] = True` to `stitch_fwd_bwd`:
- File: `python/zrt/ir/adapter.py:648–656` (inside the metadata dict literal)
- One line. No logic change.

**5.2** Create `python/zrt/transform/analysis/modeller.py`:
```python
def estimate_training_from_graphs(fwd_graph, bwd_graph, ctx) -> dict:
    """Bridge: captured graphs → stitch → transform → TrainingReport."""
    unified = stitch_fwd_bwd(fwd_graph, bwd_graph)
    pipeline = build_default_pipeline(ctx)
    g = pipeline.run(unified, ctx)
    return {
        "pipeline_metrics": g.metadata.get("pipeline_metrics"),
        "memory_breakdown": g.metadata.get("memory_breakdown"),
        "training_flops": g.metadata.get("training_flops"),
    }
```
Wire this into `graph/main.py` when `--train` flag is set and both phases are captured.

### P1 — ZeroBubble Composer

**5.3** Add `ZeroBubbleComposer` to `python/zrt/training/compose/`:
- Inherits `PipelineComposer`
- Formula: `step_time = M * t_b + (pp-1) * (t_b - t_w) / 2` (simplified ZB-H model)
- Requires: per-stage `t_b` (input grad) and `t_w` (weight grad) timing
- Source per-node `flops_dw` from `flops_train.py` to split backward time
- Register in `training.py:350` dispatch: `pp_schedule == "zero_bubble"`

### P2 — Compressed Attention FLOPs

**5.4** Add `attn_compression_ratio: float = 1.0` to `ModelSpec` (`training/spec/model.py`)

**5.5** In `flops_train.py:97`:
```python
compression = getattr(node, 'attn_compression_ratio', 1.0)
fwd_flops = 2 * batch * (seq_len ** 2) * heads * head_dim * compression
```
Also accept from `g.metadata.get("attn_compression_ratio", 1.0)` for graph-based path.

### P3 — Anchor Integration

**5.6** Enable `strict_mfu_check=True` in `gpt3_175b_megatron.yaml` and hook `test_anchor_estimate_integration_placeholder` to run actual `estimate()` via Stack A.

**5.7** Add 3 test cases to `test_context_parallel.py` covering Ulysses A2A insertion, Ring P2P insertion, and hybrid mode.

### P4 — HFU Metric

**5.8** Add `hfu` alongside `mfu` in `PipelineStepMetrics`:
```python
# HFU accounts for recomputed activations: multiply FLOPs by recompute overhead
recompute_factor = 1.0 + recomputed_flops / total_flops
hfu = mfu * recompute_factor
```
HFU is the metric published in PaLM and LLaMA papers — needed for apples-to-apples comparison with reported hardware efficiencies.

---

## 6. Critical Files

| File | Action | Priority |
|------|--------|----------|
| `python/zrt/ir/adapter.py:648–656` | Add `fwd_bwd_stitched=True` to metadata | P0 |
| `python/zrt/transform/analysis/modeller.py` | Create (new file) | P0 |
| `python/zrt/graph/main.py` | Wire `estimate_training_from_graphs` | P0 |
| `python/zrt/training/compose/zero_bubble.py` | Create ZeroBubble composer | P1 |
| `python/zrt/transform/analysis/training.py:350` | Add zero_bubble dispatch branch | P1 |
| `python/zrt/transform/analysis/flops_train.py:85–97` | Add compression ratio scaling | P2 |
| `python/zrt/training/spec/model.py` | Add `attn_compression_ratio` field | P2 |
| `tests/training/anchors/test_anchors.py` | Enable strict + actual estimate() runs | P3 |
| `tests/training/test_context_parallel.py` | Add CP comm insertion tests | P3 |
| `python/zrt/transform/analysis/training.py:498–500` | Add HFU alongside MFU | P4 |

---

## 7. Verification

```bash
# After P0 — verify stitch integration
pytest tests/training/test_captured_graph_modelling.py -v -k "stitch or pp_routing"

# After P0 — verify graph-native memory path activated
pytest tests/training/test_captured_graph_modelling.py -v -k "memory"

# After P1 — verify ZeroBubble schedule
pytest tests/training/test_graph_schedule.py -v -k "zero_bubble"

# After P2 — verify compressed attention FLOPs
pytest tests/training/test_flops.py -v -k "attention"

# After P3 — anchor regression (GPT-3, LLaMA-3, DeepSeek-V3 within 15% MFU)
pytest tests/training/anchors/test_anchors.py -v

# Full training suite
pytest tests/training/ -v 2>&1 | tail -n 50
```

---

## 8. Summary

**The design doc is ~65% realized.** The graph-native transform passes (`TrainingPipelinePass`, `TrainingMemoryPass`, `TrainingFlopsPass`) are solid and tested. The critical missing piece is the integration bridge (`modeller.py`) and a one-line metadata fix that together unlock the graph-native activation memory path. Stack A remains a working fallback and validation reference.

For DeepSeek V4 forward-compatibility, the two urgent additions are: compressed attention FLOPs scaling and ZeroBubble schedule support. Both are bounded, well-specified additions that don't require restructuring existing code.
