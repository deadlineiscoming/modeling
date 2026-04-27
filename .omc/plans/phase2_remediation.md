# Phase 2 Remediation Plan — Heterogeneous 1F1B + Phase-Aware Timeline

_2026-04-24. Source review: `/home/shane/.claude/plans/inherited-snacking-petal.md`; design spec: `docs/training_modeller_zh.md`._

---

## Context

Phase 0 (fwd+bwd stitching) and Phase 1 (correctness bug fixes) are complete enough to unlock Phase 2, but the current Phase 2 implementation is only structurally landed. The main remaining gaps are not all at the formula layer:

1. `Timeline` cannot decompose fwd vs bwd latency per stage, so the heterogeneous 1F1B formula from `docs/training_modeller_zh.md:199–201` still cannot be computed. The current homogeneous formula is numerically acceptable only for symmetric stages.
2. `estimate_training_from_graphs` duplicates step-time math on top of Pass output, so even a correct Pass-level composer can be silently bypassed.
3. `PipelineParallelPass` inserts `comm.send_recv` nodes, but the inserted node is not yet wired into receiver-stage consumers. That means stage-local scheduling can ignore boundary communication on the true critical path, and stitched train graphs also lack the reverse-direction backward transfer semantics required for realistic PP timing. **Verified bug** — `pipeline_parallel.py:210–218` only creates a `last_src → p2p` edge; no stage-`s+1` consumer is rewired to depend on the p2p node.
4. `FusionPass` already propagates `stage_id` and `phase` annotations onto fused nodes (`fusion/pass_.py:247–270`). The remaining gap is purely defensive — a fused group with mixed values silently drops the annotation instead of asserting. Not a bug under current pass order, but the invariant should be hard-enforced to fail loudly if future pass changes break it.

No existing end-to-end test exercises stitched `pp > 1`. Current baseline is green with `PYTHONPATH=python`, but coverage is still dominated by stitch correctness and pp=1 modelling, plus forward-only PP tests.

This plan closes the graph-semantic and timing gaps in Phase 2 and adds the missing stitched `pp>1` regression. Activation-memory refinements remain deferred to Phase 3.

---

## Spec Recap (training_modeller_zh.md:199–201)

```
t_stage  = max(t_fwd[s] + t_bwd[s])                       # bottleneck
step_us  = (pp−1)·t_fwd[0] + M·t_stage + (pp−1)·t_bwd[pp−1]
bubble   = 2·(pp−1)·t_stage_avg / step_us
```

Per-stage extraction (training_modeller_zh.md:77–81):
```
for s in range(pp):
    stage_nodes = [n for n in g.nodes if n.annotations["stage_id"] == s]
    timeline[s] = DAGScheduler(hw).schedule(stage_nodes)
    t_fwd[s] = timeline[s].phase_latency("fwd")
    t_bwd[s] = timeline[s].phase_latency("bwd")
```

---

## Item 1 — Put PP boundary communication on the real dependency path

**Files**: `python/zrt/transform/parallel/pipeline_parallel.py`, `tests/training/test_pipeline_parallel.py`

**Problem**: the current PP pass inserts one `comm.send_recv` after the sender-stage tail node, but it does not rewire receiver-stage consumers to depend on that communication node. As a result, subgraph scheduling for stage `s+1` can start as if the activation were already present. On stitched train graphs, the same issue applies to backward-direction boundary traffic.

**Change**:

1. Replace the current "append-after-last-src" behavior with explicit boundary-edge rewriting:
   - detect edges that cross `stage_id = s -> s+1`
   - insert `comm.send_recv`
   - rewire the receiver-side edge(s) to consume the comm node's output tensor, not the sender node's output directly
2. Ensure backward-path stage boundaries are represented too. For a stitched graph this means the pass must not assume all inter-stage traffic is forward-only.
3. Preserve message-size metadata already written today (`src_stage`, `dst_stage`, `message_size_bytes`), but make the inserted node observable in the receiver stage's dependency chain.

**Verification**:

- unit test: any node in stage 1 that originally consumed a stage-0 tensor must have the inserted `comm.send_recv` as a predecessor after the pass runs
- stitched train test: at least one backward-phase boundary transfer must exist or be proven unnecessary by stage layout

---

## Item 2 — Harden fusion invariant: reject stage/phase-mixing groups

**Files**: `python/zrt/transform/fusion/pass_.py`, `tests/training/test_captured_graph_modelling.py`

**Status of concern**: `_fused_node()` at `fusion/pass_.py:247–270` **already propagates** `stage_id` and `phase` from the grouped nodes. Single-node relabels at `pass_.py:326–332` mutate the existing node in place, so annotations are inherently preserved. **Under current pass order this is not a real defect.**

**Remaining gap (defensive)**: when a group contains mixed `stage_id` or `phase` values, the propagation silently drops the annotation (lines 252–254 require `len(vals) == 1`). If a future pass change allows fusion groups to span stage/phase boundaries, the fused node would lose stage routing without any signal.

**Change**:

1. In `_fused_node()`, raise an assertion (or log an error + refuse to fuse) when `len(vals) > 1` for `stage_id` or `phase`. This is the fusion invariant — Phase 2 depends on it.
2. Regression test: build a synthetic group with mixed `stage_id` (contrived, but cheap to construct) and verify either the assertion fires or the group is left un-fused.
3. Non-mixed test (optional but cheap): run the stitched pp=2 graph through PP + Fusion and assert every fused node still carries `stage_id ∈ {0, 1}` and `phase ∈ {"fwd", "bwd"}`.

**Why this is defensive, not corrective**: under today's pass ordering (`PipelineParallelPass` → `FusionPass`, fusion groups by `(scope, layer)`, PP groups by layer), fusion groups never span stage boundaries. But the assertion is the signal you want when that invariant breaks later.

---

## Item 3 — Add `Timeline.phase_latency(phase)` API

**File**: `python/zrt/executor/scheduler.py`

**Problem**: `ScheduledOp` (scheduler.py:28–44) records `node_id`, `op_type`, `category`, `stream_type` but not the source node's `annotations["phase"]`. `Timeline` (scheduler.py:47–107) exposes `total_latency_us`, `compute_time_us`, `comm_time_us`, `overlap_us` — no phase decomposition.

**Change**:

1. Add `phase: str = ""` field to `ScheduledOp` dataclass (scheduler.py:28–44).
2. In `DAGScheduler.schedule()` (scheduler.py:126–170), populate `phase` from `node.annotations.get("phase", "")` when creating the `ScheduledOp` at line 155.
3. Add `phase_latency(phase: str) -> float` method on `Timeline` that computes wall-clock latency of ops matching the phase:
   ```python
   def phase_latency(self, phase: str) -> float:
       ops = [op for op in self.scheduled_ops if op.phase == phase]
       if not ops:
           return 0.0
       start = min(op.start_us for op in ops)
       end   = max(op.end_us   for op in ops)
       return end - start
   ```
   Use wall-clock span (end − start of the subset), not `sum(lat)` — this preserves overlap semantics consistent with `total_latency_us`. For a subgraph containing only fwd or only bwd ops this equals the serialized latency; for the combined subgraph Phase 0 cross-graph edges force fwd-then-bwd order so spans don't overlap.

**Verification**: unit test on a synthetic 3-node graph with two `phase="fwd"` ops and one `phase="bwd"` op → assert `phase_latency("fwd") + phase_latency("bwd") ≈ total_latency_us` when fwd and bwd don't overlap.

---

## Item 4 — Extract `t_fwd[s]` and `t_bwd[s]` per stage

**File**: `python/zrt/transform/analysis/training.py`

**Change** (`TrainingPipelinePass.run`, lines 253–273):

Replace the current single-dict collection with two:

```python
stage_fwd: dict[int, float] = {}
stage_bwd: dict[int, float] = {}
for s_id in range(pp):
    node_ids = stage_node_sets.get(s_id, set())
    if not node_ids:
        stage_fwd[s_id] = 0.0
        stage_bwd[s_id] = 0.0
        continue
    sub = g.subgraph(node_ids)
    tl = sched.schedule(sub)
    fwd = tl.phase_latency("fwd")
    bwd = tl.phase_latency("bwd")
    if layer_scale != 1.0:
        fwd *= layer_scale
        bwd *= layer_scale
    stage_fwd[s_id] = fwd
    stage_bwd[s_id] = bwd

g.metadata["stage_timelines_fwd"] = dict(stage_fwd)
g.metadata["stage_timelines_bwd"] = dict(stage_bwd)
```

**Backward-compat**: `stage_timelines_fwd` key is reused — existing consumers (none in tree currently) see the fwd-only split instead of combined. Add `stage_timelines_bwd` as a new key. If a subgraph contains only `phase="bwd"` nodes, `phase_latency("fwd")` returns `0.0` — report will flag this case.

**Handle inference path**: when `phase` annotations are absent (inference graph, no stitch), `phase_latency("fwd")` returns 0. Fall back to `total_latency_us` as `stage_fwd[s]` with `stage_bwd[s] = 0` in that branch.

---

## Item 5 — Heterogeneous 1F1B composer

**File**: `python/zrt/transform/analysis/training.py:285–298`

**Change**: replace the homogeneous formula with the heterogeneous one when both `stage_fwd` and `stage_bwd` are non-empty:

```python
if pp > 1 and stage_fwd and stage_bwd and any(stage_bwd.values()):
    # Heterogeneous 1F1B: stages may have asymmetric fwd/bwd times
    t_fwd_0    = stage_fwd.get(0, 0.0)
    t_bwd_last = stage_bwd.get(pp - 1, 0.0)
    t_stage    = max((stage_fwd[s] + stage_bwd[s]) for s in range(pp))
    step_time_us = ((pp - 1) * t_fwd_0
                    + num_microbatches * t_stage
                    + (pp - 1) * t_bwd_last)
    t_stage_avg = sum(stage_fwd[s] + stage_bwd[s] for s in range(pp)) / pp
    bubble_us = 2 * (pp - 1) * t_stage_avg
    bubble_fraction = bubble_us / step_time_us if step_time_us > 0 else 0.0
    per_stage_us = t_stage
else:
    # Homogeneous fallback (pp=1 or no phase annotations)
    per_stage_us = max(stage_fwd.values(), default=0.0) + max(stage_bwd.values(), default=0.0) \
                   if stage_fwd else 0.0
    if pp == 1 and per_stage_us == 0.0:
        tl = sched.schedule(g)
        per_stage_us = tl.total_latency_us
    if layer_scale != 1.0 and pp == 1:
        per_stage_us *= layer_scale
    effective_steps = num_microbatches + pp - 1
    step_time_us = per_stage_us * effective_steps
    bubble_fraction = (pp - 1) / effective_steps if effective_steps > 0 else 0.0
```

**Rationale**: heterogeneous branch uses the spec formula. Homogeneous fallback preserves current behavior for inference / `pp=1`. Keep `per_stage_us` populated so `PipelineStepMetrics.per_stage_ms` remains meaningful.

**Edge case**: if only `phase="fwd"` annotations exist (no stitch done), `any(stage_bwd.values())` is false → falls back to homogeneous, matching current behavior.

---

## Item 6 — Remove duplicate step-time math in `estimate_training_from_graphs`

**File**: `python/zrt/transform/analysis/modeller.py:333–339`

**Change**: delete the duplicate formula and read from `pipeline_metrics` directly:

```python
# Delete:
#   step_time_ms = (num_microbatches + pp_val - 1) * per_stage_ms
#   bubble_time_ms = (pp_val - 1) * per_stage_ms
#   bubble_fraction = bubble_time_ms / step_time_ms if step_time_ms > 0 else 0.0

# Replace with:
step_time_ms    = pipeline_metrics.step_time_ms    if pipeline_metrics else 0.0
bubble_fraction = pipeline_metrics.bubble_fraction if pipeline_metrics else 0.0
per_stage_ms    = pipeline_metrics.per_stage_ms    if pipeline_metrics else 0.0
```

Also remove the now-unused `pp_val` / `num_microbatches` locals if they become dead code.

**Check**: verify the "no backward_graph" branch (lines 316–331) still sums fwd+bwd `per_stage_ms` from separate pipelines correctly — that path does not use stitch, so its heterogeneous 1F1B formula must still fall through the Pass fallback.

---

## Item 7 — Add `test_pp_routing_end_to_end`

**File**: `tests/training/test_captured_graph_modelling.py` (append).

**Gap**: `_captured_style_graph()` does **not** set `node.layer` on its OpNodes (lines 31–113 — no `layer=` kwarg). `PipelineParallelPass` parses `int(node.layer)` to group layers, so the test needs a graph variant that annotates `layer`. `_backward_graph_for_fwd()` already sets `layer=str(layer_id)` (line 444).

**Plan**:

1. Add a helper `_captured_style_graph_with_layers(num_traced=2)` that mirrors `_captured_style_graph` but passes `layer=str(layer_id)` to every OpNode. Alternatively, extend `_captured_style_graph` to set `layer` unconditionally (safer — all existing tests pass even with the annotation set, since none relied on absent `layer`).
2. Append the test:

```python
def test_pp_routing_end_to_end():
    """End-to-end: stitch fwd+bwd → pipeline with pp=2 → per-stage timelines + P2P + 1F1B."""
    from zrt.ir.adapter import stitch_fwd_bwd
    from zrt.transform.pipeline import build_training_pipeline

    fwd = _captured_style_graph()        # must have layer='0'/'1' annotations
    bwd = _backward_graph_for_fwd(fwd)
    stitched = stitch_fwd_bwd(fwd, bwd)

    hw = _hw()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=2, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8, num_microbatches=4),
    )
    pipe = build_training_pipeline()
    result = pipe.run(stitched, ctx)

    # 1) Every node has stage_id ∈ {0, 1}
    for nid, node in result.nodes.items():
        sid = node.annotations.get("stage_id")
        assert sid in (0, 1), f"Node {nid} has bad stage_id={sid}"

    # 2) At least one P2P send_recv with src=0, dst=1
    p2p = [n for n in result.nodes.values()
           if n.op_type == "comm.send_recv"
           and n.attrs.get("src_stage") == 0
           and n.attrs.get("dst_stage") == 1]
    assert p2p, "Expected at least one comm.send_recv crossing stage 0→1"

    # 3) Receiver-side dependency must go through the P2P node
    p2p_ids = {n.id for n in p2p}
    stage1_nodes = [n for n in result.nodes.values() if n.annotations.get("stage_id") == 1]
    assert any(set(result.predecessors(n.id)) & p2p_ids for n in stage1_nodes), (
        "Stage-1 consumers should depend on boundary comm nodes"
    )

    # 4) Per-stage timelines populated
    stage_fwd = result.metadata.get("stage_timelines_fwd", {})
    stage_bwd = result.metadata.get("stage_timelines_bwd", {})
    assert set(stage_fwd.keys()) == {0, 1}, f"stage_timelines_fwd keys={set(stage_fwd.keys())}"
    assert set(stage_bwd.keys()) == {0, 1}, f"stage_timelines_bwd keys={set(stage_bwd.keys())}"
    assert all(v > 0 for v in stage_fwd.values()), f"stage_timelines_fwd has zero: {stage_fwd}"
    assert any(v > 0 for v in stage_bwd.values()), f"stage_timelines_bwd unexpectedly empty: {stage_bwd}"

    # 5) Pipeline metrics sensible
    pm = result.metadata.get("pipeline_metrics")
    assert pm is not None and pm.step_time_ms > 0
    # For symmetric stages + pp=2, M=4: bubble ≈ (pp-1)/(M+pp-1) = 1/5 = 0.2
    M = 4
    pp = 2
    expected_bubble = (pp - 1) / (M + pp - 1)
    assert abs(pm.bubble_fraction - expected_bubble) / expected_bubble < 0.2, \
        f"bubble_fraction={pm.bubble_fraction}, expected ≈{expected_bubble}"

    # 6) Cross-graph edge survival — strict: every stage must contain bwd nodes
    #    and at least one bwd in every stage must have a predecessor in that stage.
    for s in (0, 1):
        stage_node_ids = {nid for nid, n in result.nodes.items()
                          if n.annotations.get("stage_id") == s}
        sub = result.subgraph(stage_node_ids)
        bwd_nodes = [nid for nid, n in sub.nodes.items()
                     if n.annotations.get("phase") == "bwd"]
        assert bwd_nodes, (
            f"Stage {s} has no bwd nodes after stitch+PP split — "
            "stage_id assignment did not cover the backward graph"
        )
        bwd_with_preds = [nid for nid in bwd_nodes if sub.predecessors(nid)]
        assert bwd_with_preds, (
            f"Stage {s} bwd nodes {bwd_nodes} all lack same-stage predecessors — "
            "cross-graph fwd→bwd edges did not survive the subgraph split"
        )


def test_pp_heterogeneous_1f1b_formula():
    """Asymmetric per-stage timing must drive the heterogeneous 1F1B formula
    (distinct from the homogeneous fallback).

    Without this test, Item 5's heterogeneous branch is never exercised — the
    symmetric synthetic graph makes both formulas produce the same number.

    Strategy: construct a stitched pp=2 graph where stage-0 nodes carry
    latency_us = 10, stage-1 nodes carry latency_us = 100 (10× imbalance).
    Homogeneous formula:    step = 5 * max(t_stage) = 5 * 100 + any-fwd/bwd-mix
    Heterogeneous formula:  step = (pp-1)*t_fwd[0] + M*max(fwd+bwd) + (pp-1)*t_bwd[-1]
    The two formulas produce numerically different step_time for this case,
    so an identical result indicates the heterogeneous branch did not run.
    """
    from zrt.ir.adapter import stitch_fwd_bwd
    from zrt.transform.pipeline import build_training_pipeline

    fwd = _captured_style_graph()
    bwd = _backward_graph_for_fwd(fwd)
    stitched = stitch_fwd_bwd(fwd, bwd)

    # Inject asymmetric latency_us: layer 0 nodes = 10µs, layer 1 nodes = 100µs
    for node in stitched.nodes.values():
        try:
            lid = int(node.layer)
        except (ValueError, TypeError):
            continue
        node.annotations["latency_us"] = 10.0 if lid == 0 else 100.0

    hw = _hw()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=2, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8, num_microbatches=4),
    )
    result = build_training_pipeline().run(stitched, ctx)

    stage_fwd = result.metadata.get("stage_timelines_fwd", {})
    stage_bwd = result.metadata.get("stage_timelines_bwd", {})
    pm = result.metadata.get("pipeline_metrics")

    # Asymmetry must be visible in per-stage output
    assert stage_fwd.get(0, 0) < stage_fwd.get(1, 0), (
        f"stage 0 fwd should be smaller than stage 1 fwd: {stage_fwd}"
    )
    assert stage_bwd.get(0, 0) < stage_bwd.get(1, 0), (
        f"stage 0 bwd should be smaller than stage 1 bwd: {stage_bwd}"
    )

    # Heterogeneous step_time must match the spec formula, NOT the
    # homogeneous formula (M + pp − 1) * max_stage.
    M, pp = 4, 2
    t_fwd_0    = stage_fwd[0]
    t_bwd_last = stage_bwd[1]
    t_stage    = max(stage_fwd[s] + stage_bwd[s] for s in (0, 1))
    expected_step_us = ((pp - 1) * t_fwd_0
                        + M * t_stage
                        + (pp - 1) * t_bwd_last)
    homogeneous_step_us = (M + pp - 1) * t_stage
    assert abs(pm.step_time_ms * 1000 - expected_step_us) / expected_step_us < 0.05, (
        f"step_time={pm.step_time_ms*1000}µs does not match heterogeneous spec "
        f"{expected_step_us}µs (homogeneous would give {homogeneous_step_us}µs)"
    )
    assert abs(expected_step_us - homogeneous_step_us) / homogeneous_step_us > 0.05, (
        "Test setup is broken: heterogeneous and homogeneous formulas are "
        "numerically too close. Increase the latency asymmetry."
    )
```

**Tolerance note** on `test_pp_routing_end_to_end` assertion 5: the bubble fraction is `1/5 = 0.2`. Heterogeneous formula (post-Item 5) yields the same numerical answer for symmetric stages, so the assertion holds both before and after Item 5 lands. Tolerance 20% absorbs rounding / the `layer_scale` multiplier.

**Why `test_pp_heterogeneous_1f1b_formula` matters**: without it, Item 5 can be merged with the heterogeneous branch never exercised — the symmetric graph drives the fallback branch only. This test forces Item 5's spec-formula code path to run and produce a number the homogeneous formula could not.

**Add `num_microbatches` field check**: `TrainingConfig` must expose `num_microbatches`. If it doesn't (grep: `grep -n "num_microbatches" python/zrt/transform/context.py`), add it as a field with default = `global_batch // micro_batch`.

---

## Items 8–9 — Deferred to Phase 3

These close precision gaps in activation memory and are listed here for traceability; they do not gate Phase 2 completion.

**Item 6**: `inflight = pp - stage_id` (currently `pp`; self-commented TODO at `training.py:181`). Requires activation memory to become per-stage rather than global.

**Item 7**: Replace static `34·h·s·L·B` coefficient with graph-derived per-layer activation bytes by walking `stitch_fwd_bwd`'s cross-graph edges (a fwd output consumed by a bwd node → saved activation → sum `mem_bytes`). Requires the memory pass to iterate nodes whose outputs participate in a cross-phase edge.

---

## Verification Protocol

After each item ships, run:
```
PYTHONPATH=python pytest tests/training/test_captured_graph_modelling.py -v 2>&1 | tail -n 80
```

Expected progression:
- **Baseline (pre-fix)**: 20/20 passing (verified on 2026-04-24 — see Current Baseline section).
- **After Item 7 test added only** (no code fixes yet): both new tests **must fail for the right reasons**:
  - `test_pp_routing_end_to_end` assertion 3 → fails (P2P not on receiver dependency path, Item 1 unlanded)
  - `test_pp_routing_end_to_end` assertion 4 → fails (no `stage_timelines_bwd` key yet, Item 4 unlanded)
  - `test_pp_heterogeneous_1f1b_formula` → fails (heterogeneous branch unlanded, Item 5 still uses homogeneous formula)
  - If either test passes at this checkpoint, **the test is not catching the bug it was designed to catch** — review assertions before landing fixes.
- **After Items 1–2**: `test_pp_routing_end_to_end` assertions 1–3 pass (P2P wired, stage/phase preserved through fusion). Assertion 4 still fails until Items 3–4. Assertion 6 passes (cross-graph edge survival).
- **After Items 3–4**: `stage_timelines_fwd`/`stage_timelines_bwd` populated; assertion 4 passes.
- **After Item 5**: `test_pp_heterogeneous_1f1b_formula` passes — proves the heterogeneous branch actually ran.
- **After Item 6**: `estimate_training_from_graphs` reports Pass-owned `step_time_ms` / `bubble_fraction`. No behavioral regression in existing pp=1 tests.

Full repo smoke:
```
PYTHONPATH=python pytest tests/ -v 2>&1 | tail -n 50
```
to confirm no regression in non-training suites (executor/scheduler tests especially, since Item 1 modifies `Timeline` / `ScheduledOp`).

---

## Critical Files (ordered by touch order)

| File | Purpose |
|---|---|
| `python/zrt/transform/parallel/pipeline_parallel.py` | Rewire stage-boundary edges through `comm.send_recv` so boundary comm is on the real dependency path |
| `python/zrt/transform/fusion/pass_.py` | Preserve `phase` / `stage_id` when building fused nodes |
| `python/zrt/executor/scheduler.py` | Add `phase` to `ScheduledOp`; add `Timeline.phase_latency()` |
| `python/zrt/transform/analysis/training.py` | Use `phase_latency` in per-stage loop; switch to heterogeneous 1F1B formula |
| `python/zrt/transform/analysis/modeller.py` | Delete duplicate step-time computation |
| `python/zrt/transform/context.py` | Verify `TrainingConfig.num_microbatches` exists; add if missing |
| `tests/training/test_pipeline_parallel.py` | Add receiver-side dependency coverage for boundary comm |
| `tests/training/test_captured_graph_modelling.py` | Extend `_captured_style_graph` with `layer` kwarg; add stitched `pp>1` e2e regression |

---

## Current Baseline (2026-04-24)

```
PYTHONPATH=python pytest tests/training/test_captured_graph_modelling.py -v 2>&1 | tail -n 80
==================== 20 passed in 1.83s ====================
```

All Phase 0 stitch tests and Phase 1 correctness tests currently green. The remediation items above are additive — existing tests should not regress.
