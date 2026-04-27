# Plan: Wire Graph-Native Training Pipeline to PipelineComposer

## Context

`_run_training_modelling` (cli.py) is the entry point for `--train --hw` graph-capture-based training estimation. Per the design doc (`docs/training_modeller_zh.md`), the intended flow is:

```
Graph Capture → stitch_fwd_bwd → Transform Pipeline
  → per-stage DAGScheduler (executor)
  → PipelineComposer (compose/) → StepResult → TrainingReport
```

**Two bugs break this flow:**

1. **`TrainingPipelinePass` duplicates pipeline formulas inline** instead of calling the validated `PipelineComposer` classes (`OneF1BComposer`, `Interleaved1F1BComposer`, `ZeroBubbleComposer`, `DualPipeComposer`, `DualPipeVComposer`). These composers are anchor-tested against real published MFU numbers. The inline formulas may diverge.

2. **`_run_training_modelling` has a broken export block** — the HTML/Chrome-trace code is trapped inside `except UnicodeEncodeError:`, references undefined variables (`output_dir`, `summary`, `fwd_tl`, `bwd_tl`, `parallel_desc`), and can never run.

The graph-native path IS the right source of truth — actual op latencies from captured aten ops are more accurate than `ModelSpec`-derived analytical formulas. No `ModelSpec` or spec-based `estimate()` is needed.

---

## Root Cause

### Bug 1 — `TrainingPipelinePass` (`python/zrt/transform/analysis/training.py`)

The pass already does the right first half:
- Runs `DAGScheduler` on each stage subgraph (lines 338–368)
- Calls `tl.phase_latency("fwd")` / `tl.phase_latency("bwd")` — **this works**: `DAGScheduler.schedule()` propagates `node.annotations["phase"]` → `ScheduledOp.phase` (scheduler.py:172), and `phase_latency()` reads `op.phase` (scheduler.py:91–96)

But the second half (lines 370–473) implements its own 1F1B/VPP/DualPipe/ZeroBubble formulas instead of delegating to the `PipelineComposer` subclasses in `python/zrt/training/compose/pipeline.py`.

### Bug 2 — `_run_training_modelling` (`python/zrt/cli.py`, lines 333–365)

Export code indented inside `except UnicodeEncodeError:` (should be after the try/except), plus undefined vars:
- `output_dir` → should be `result.output_dir`
- `summary` → no `TrainingSummary` exists in this path (only `TrainingReport`)
- `fwd_tl`, `bwd_tl` → no `Timeline` objects returned by `estimate_training_from_graphs`
- `parallel_desc` → not defined

---

## Key Interfaces (verified)

**`PipelineComposer.compose()`** — `python/zrt/training/compose/pipeline.py:36–47`
```python
def compose(
    self,
    stage_times: list[StageTime],  # per-stage fwd/bwd/dw timings in SECONDS
    M: int,                         # num microbatches
    pp: int,
    dp_ar_time: float,              # DP allreduce time in seconds
    strategy: Strategy,             # only needs pp_schedule, vpp_chunks, dp_overlap_in_bubble
) -> StepResult:
```

**`StageTime`** — `python/zrt/training/compose/stage.py:22–30`
```python
@dataclass
class StageTime:
    fwd: float = 0.0      # seconds
    bwd: float = 0.0
    bwd_dx: float = 0.0
    bwd_dw: float = 0.0
    comm_fwd: float = 0.0
    comm_bwd: float = 0.0
```

**Composer dispatch map** (from `PPSched` enum in `zrt.training.spec.strategy`):
```
"1f1b"                → OneF1BComposer
"interleaved"/"i1f1b" → Interleaved1F1BComposer
"zb"/"zero_bubble"    → ZeroBubbleComposer
"dualpipe"            → DualPipeComposer
"dualpipev"           → DualPipeVComposer
```

**Import note**: Both `python.zrt.*` and `zrt.*` resolve simultaneously when running `PYTHONPATH=python pytest` from repo root (pytest adds current dir to sys.path). Use lazy (function-level) imports inside `TrainingPipelinePass.run()`.

---

## ModelSpec-vs-Graph Coexistence Bugs

### Bug 3 — `TrainFlopsPass` double-counts backward FLOPs on stitched graphs

`TrainFlopsPass` runs on the stitched graph and for **every** node (fwd and bwd phases alike) annotates:
- `flops_fwd` = cost of that op (e.g. 2MNK for a matmul)
- `flops_dx ≈ flops_fwd`, `flops_dw ≈ flops_fwd` (ratio-derived backward estimates)

For **bwd-phase nodes** this is wrong: those nodes ARE the actual dx/dw computations captured by `loss.backward()`. Applying the ratio-based `flops_dx/dw` on top of them creates phantom FLOPs that don't exist.

**Fix in `flops_train.py::TrainFlopsPass.run()`**: gate `_calculate_grad_flops()` on node phase:
```python
phase = node.annotations.get("phase", "fwd")
is_bwd = phase in {"bwd", "backward", "train_backward"}

if is_bwd:
    # This node IS a backward op — its cost is already in flops_fwd
    dx_flops, dw_flops = 0.0, 0.0
else:
    dx_flops, dw_flops = self._calculate_grad_flops(node, fwd_flops)
```

### Bug 4 — `TrainingFlopsPass` aggregates wrong FLOPs on stitched graphs

`TrainingFlopsPass` currently sums `flops_fwd` across ALL nodes for `forward_flops` and `flops_dx+flops_dw` for `backward_flops`. For stitched graphs this is wrong:
- Bwd-phase nodes' `flops_fwd` values (cost of backward ops) get counted in `forward_flops`
- After Bug 3 fix, `flops_dx/dw` on bwd nodes = 0, so `backward_flops` = 0

**Fix in `training.py::TrainingFlopsPass.run()`**: branch on stitched vs non-stitched:
```python
if g.metadata.get("fwd_bwd_stitched"):
    # Graph-native path: each node's flops_fwd = cost of that op (fwd OR bwd)
    forward_flops = sum(
        n.annotations.get("flops_fwd", 0) for n in g.nodes.values()
        if n.annotations.get("phase", "fwd") not in {"bwd", "backward", "train_backward"}
    )
    backward_flops = sum(
        n.annotations.get("flops_fwd", 0) for n in g.nodes.values()
        if n.annotations.get("phase", "") in {"bwd", "backward", "train_backward"}
    )
else:
    # Non-stitched (fwd graph only): estimate backward via dx/dw ratios
    forward_flops = sum(n.annotations.get("flops_fwd", 0) for n in g.nodes.values())
    backward_flops = sum(
        n.annotations.get("flops_dx", 0) + n.annotations.get("flops_dw", 0)
        for n in g.nodes.values()
    )
```

Apply the same phase-filter to `recompute_flops` (only fwd-phase nodes can be recomputed).

---

## Stage 4 Analyze — Redundancy Cleanup

### Issue A — `FlopsPass` + `RooflinePass` double-compute `_fmr()`

Both call `sim._fmr(node)` independently. `RooflinePass` should read the annotations already written by `FlopsPass`.

**Fix in `passes.py::RooflinePass.run()`**: replace `flops, read_b, write_b = sim._fmr(node)` with:
```python
flops  = node.annotations.get("flops", 0)
read_b = node.annotations.get("read_bytes", 0)
write_b = node.annotations.get("write_bytes", 0)
if flops == 0 and read_b == 0:   # FlopsPass didn't run — fall back
    flops, read_b, write_b = sim._fmr(node)
```

### Issue B — `TrainFlopsPass` shadow-recomputes FLOPs

`TrainFlopsPass._calculate_fwd_flops()` reimplements simplified FLOPs formulas using op-type string matching, overriding the more comprehensive `RooflineSimulator` output already in `node.annotations["flops"]`.

**Fix in `flops_train.py::TrainFlopsPass.run()`**: replace `_calculate_fwd_flops()` call with:
```python
fwd_flops   = node.annotations.get("flops", 0)       # from FlopsPass
read_bytes  = node.annotations.get("read_bytes", 0)
write_bytes = node.annotations.get("write_bytes", 0)
```
Keep `_calculate_grad_flops(fwd_flops)` unchanged. Delete `_calculate_fwd_flops()` entirely.

### Issue C — Dead module `memory_train.py`

`python/zrt/transform/analysis/memory_train.py` defines `TrainMemoryPass` which is NOT wired into `build_pipeline()`, NOT exported from `analysis/__init__.py`, and shadows the active `TrainingMemoryPass` in `training.py`.

**Fix**: Delete `memory_train.py`.

---

## Changes Summary

### File 1: `python/zrt/transform/analysis/training.py`

**`TrainingPipelinePass.run()`** — keep per-stage DAGScheduler loop (lines 329–368) unchanged. Replace the entire inline formula block (lines 370–473) with:

```python
from python.zrt.training.compose.stage import StageTime as _StageTime
from python.zrt.training.compose.pipeline import (
    OneF1BComposer, Interleaved1F1BComposer, ZeroBubbleComposer,
    DualPipeComposer, DualPipeVComposer,
)
from python.zrt.training.spec.strategy import Strategy as _Strategy, PPSched, OptKind

_COMPOSER_MAP = {
    "1f1b": OneF1BComposer, "interleaved": Interleaved1F1BComposer,
    "i1f1b": Interleaved1F1BComposer, "zb": ZeroBubbleComposer,
    "zero_bubble": ZeroBubbleComposer, "dualpipe": DualPipeComposer,
    "dualpipev": DualPipeVComposer,
}
_PP_SCHED_ENUM = {
    "1f1b": PPSched.ONE_F_ONE_B, "interleaved": PPSched.INTERLEAVED,
    "i1f1b": PPSched.INTERLEAVED, "zb": PPSched.ZERO_BUBBLE,
    "zero_bubble": PPSched.ZERO_BUBBLE, "dualpipe": PPSched.DUALPIPE,
    "dualpipev": PPSched.DUALPIPE_V,
}

pp_schedule = ctx.training.pp_schedule if ctx.training else "1f1b"
stage_times_list = [
    _StageTime(
        fwd=stage_fwd.get(s, 0.0) / 1e6,
        bwd=stage_bwd.get(s, 0.0) / 1e6,
        bwd_dw=stage_bwd_dw.get(s, 0.0) / 1e6,
    )
    for s in range(pp)
]

# DP allreduce time — extract from dp_comm annotated nodes (existing logic, refactored)
dp_ar_time_s = _compute_dp_ar_time(g, hw, ctx) / 1e6

strategy_proxy = _Strategy(
    tp=ctx.parallel.tp if ctx.parallel else 1,
    pp=pp,
    ep=ctx.parallel.ep if ctx.parallel else 1,
    dp=ctx.parallel.dp if ctx.parallel else 1,
    cp=getattr(ctx.parallel, "cp", 1) if ctx.parallel else 1,
    micro_batch=ctx.training.micro_batch if ctx.training else 1,
    global_batch=ctx.training.global_batch if ctx.training else 32,
    pp_schedule=_PP_SCHED_ENUM.get(pp_schedule, PPSched.ONE_F_ONE_B),
    vpp_chunks=max(1, ctx.training.vpp_chunks if ctx.training else 1),
    zero_stage=ctx.training.zero_stage if ctx.training else 0,
    optimizer=OptKind(ctx.training.optimizer) if ctx.training else OptKind.ADAM,
    dp_overlap_in_bubble=ctx.training.dp_overlap_in_bubble if ctx.training else True,
)

step_result = _COMPOSER_MAP.get(pp_schedule, OneF1BComposer)().compose(
    stage_times_list, num_microbatches, pp, dp_ar_time_s, strategy_proxy
)

training_flops = g.metadata.get("training_flops", 0.0)
recompute_flops = g.metadata.get("recompute_flops", 0.0)
step_time_s = step_result.step_time
peak_flops = (hw.compute.bf16_tflops * 1e12) if hw else 1.0
mfu = training_flops / (peak_flops * step_time_s) if peak_flops * step_time_s > 0 else 0.0
hfu = (training_flops + recompute_flops) / (peak_flops * step_time_s) if peak_flops * step_time_s > 0 else 0.0

step_time_ms = step_result.step_time * 1000
per_stage_ms = max((st.fwd + st.bwd) for st in stage_times_list) * 1000 if stage_times_list else 0.0

g.metadata["pipeline_metrics"] = PipelineStepMetrics(
    step_time_ms=step_time_ms,
    per_stage_ms=per_stage_ms,
    warmup_steps=round(step_result.warmup / step_time_s * num_microbatches) if step_time_s > 0 else 0,
    cooldown_steps=round(step_result.cooldown / step_time_s * num_microbatches) if step_time_s > 0 else 0,
    steady_steps=num_microbatches,
    bubble_fraction=step_result.bubble_fraction,
    mfu=mfu,
    hfu=hfu,
)
```

Apply the same composer pattern to the **pp=1 / no stage_id else branch**: schedule the whole graph, split by `tl.phase_latency("fwd")` / `tl.phase_latency("bwd")` (falling back to `total_latency_us` for fwd when both are 0), build a single `StageTime`, call the composer.

Refactor the existing DP allreduce extraction block (lines 480–500) into `_compute_dp_ar_time(g, hw, ctx) -> float` (μs).

**`TrainingFlopsPass.run()`** — replace the flat sum with the phase-aware version shown in Bug 4 fix above.

### File 2: `python/zrt/cli.py`

Replace broken export block in `_run_training_modelling` (lines 330–365):
```python
    try:
        print(f"\n{report.summary()}")
    except UnicodeEncodeError:
        logger.info("Training summary:\n%s", report.summary())

    # Export report JSON
    try:
        import json as _json
        report_dir = result.output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        slug = _make_model_slug(model_id)
        json_path = report_dir / f"{slug}_training_report.json"
        json_path.write_text(_json.dumps(report.to_dict(), indent=2))
        logger.info("Training report written to %s", json_path)
    except Exception as exc:
        logger.warning("Report export failed: %s", exc)
```

### File 3: `python/zrt/transform/analysis/passes.py`

`RooflinePass.run()`: read `flops`/`read_bytes`/`write_bytes` from annotations (FlopsPass output); fall back to `_fmr()` only when not present.

### File 4: `python/zrt/transform/analysis/flops_train.py`

`TrainFlopsPass.run()`:
1. Replace `_calculate_fwd_flops()` call with `node.annotations.get("flops", 0)`
2. Gate `_calculate_grad_flops()` with `is_bwd` check (zero out for bwd-phase nodes)
3. Delete `_calculate_fwd_flops()` method

### File 5: `python/zrt/transform/analysis/memory_train.py`

Delete this file entirely.

---

## Critical Files

| File | Change |
|------|--------|
| `python/zrt/transform/analysis/training.py` | `TrainingPipelinePass`: composer wiring; `TrainingFlopsPass`: phase-aware aggregation |
| `python/zrt/cli.py` | Fix broken export block |
| `python/zrt/transform/analysis/passes.py` | `RooflinePass`: read from FlopsPass annotations |
| `python/zrt/transform/analysis/flops_train.py` | Use FlopsPass output; zero bwd-phase dx/dw; delete `_calculate_fwd_flops()` |
| `python/zrt/transform/analysis/memory_train.py` | **Delete** |

**Read-only references:**
- `python/zrt/training/compose/pipeline.py` — `PipelineComposer`, `StepResult`, concrete composers
- `python/zrt/training/compose/stage.py` — `StageTime`
- `python/zrt/training/spec/strategy.py` — `Strategy`, `PPSched`, `OptKind`
- `python/zrt/executor/scheduler.py` — `Timeline.phase_latency()`, `DAGScheduler`
- `python/zrt/transform/context.py` — `TrainingConfig`

---

## Verification

```bash
# Full training suite (should stay at 202 passed)
PYTHONPATH=python pytest tests/training/ -q 2>&1 | tail -n 20

# Anchor regression — strict MFU gate on GPT-3 175B
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -v

# Smoke test: pp=1
python -m python.zrt hf_models/llama3_8b --layers 2 --train --hw nvidia_h100_sxm --tp 1 --pp 1

# Smoke test: pp=2 (exercises per-stage path)
python -m python.zrt hf_models/llama3_8b --layers 4 --train --hw nvidia_h100_sxm --tp 1 --pp 2
```

Expected: `output/graph/<slug>/reports/<slug>_training_report.json` written, `step_time_ms` and `mfu` non-zero.
