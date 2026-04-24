# Session Progress

## 当前阶段：Phase 4 — 已完成

参考计划：`/Users/sky/.claude/plans/based-on-the-above-bright-hopcroft.md`
补充计划：`/Users/sky/.claude/plans/details-of-the-content-lively-stonebraker.md`

## 所有子项完成状态

| 子项 | 状态 | 说明 |
|------|------|------|
| 4.0/4.1/4.2 spec 路径 Composer | ✅ 完成 | `compose/pipeline.py` 4 个 Composer 类 |
| 4.1 VPP 测试（spec 路径） | ✅ 完成 | `test_interleaved_1f1b.py` |
| 4.2 DualPipe 测试（spec 路径） | ✅ 完成 | `test_dualpipe.py` |
| 4.3 EP 负载不均衡 | ✅ 完成 | `compose/stage.py::ep_imbalance_factor` |
| 4.4 搜索 / Pareto | ✅ 完成 | `search/space.py` + `estimator.py` |
| 4.5 Anchor 验证 | ✅ 完成 | `anchor/validate.py` + 3 个 YAML + `test_anchors.py` |
| 4.6 Chrome Trace | ✅ 完成 | `training/trace/exporter.py` + `report/chrome_trace.py` + `report/summary.py` chrome_trace 字段 + CLI `--trace` |
| **图路径调度分派** | ✅ 完成 | `training.py:285-298` + `modeller.py:307-343` + `test_graph_schedule.py` |

## 完成的变更（2026-04-24，第二轮）

### 变更 1：图路径调度分派（CRITICAL）

- **`python/zrt/transform/context.py`**：`TrainingConfig` 新增 `vpp_chunks: int = 1`
- **`python/zrt/transform/analysis/training.py:285-298`**：替换硬编码 1F1B 为按 `ctx.training.pp_schedule` 分派（interleaved / dualpipev / dualpipe / 1f1b）
- **`python/zrt/transform/analysis/modeller.py:307-343`**：unified 路径直接读取 `pipeline_metrics.step_time_ms`，不再重新计算
- **`tests/training/test_graph_schedule.py`**：5 个图路径测试（VPP/DualPipe/DualPipeV/bubble_fraction/pp1）全部通过

### 变更 2：Chrome Trace 报告集成（MEDIUM）

- **`python/zrt/report/chrome_trace.py`**：新建，`build_chrome_trace(timeline) -> dict`
- **`python/zrt/report/summary.py`**：`TrainingSummary` 新增 `chrome_trace: dict | None = None`
- **`python/zrt/training/cli.py`**：`model-training` 子命令新增 `--trace <path>` 参数

### 变更 3：Anchor YAML 文件（LOW）

- **`tests/training/anchors/gpt3_175b_megatron.yaml`**：GPT-3 175B，H100 SXM，TP8 PP1 DP64，MFU 0.52
- **`tests/training/anchors/llama3_70b_meta.yaml`**：LLaMA-3 70B，TP4 PP2 DP16，MFU 0.48
- **`tests/training/anchors/deepseek_v3.yaml`**：DeepSeek-V3，TP8 EP64 PP16，MFU 0.35
- **`tests/training/anchors/test_anchors.py`**：8 个测试（YAML 格式验证 + anchor/report 集成），全部通过

## 全量测试结果

```
252 passed, 34 warnings in 55.84s
```

零回归，无失败。

## 历史里程碑摘要

- Phase 0：`stitch_fwd_bwd()` 前向+反向图拼接（69/69 training tests pass）
- Phase 1：步骤时间公式修复 + 激活内存 + FLOPs 修复
- Phase 2：`PipelineParallelPass` + 逐阶段 `DAGScheduler` + 1F1B 公式
- Phase 3：`context_parallel.py` / `data_parallel.py` / CoC/MC2 overlap 注解
- Phase 4（完成）：spec 路径 Composer ✅；Chrome Trace ✅；图路径调度分派 ✅；EP 不均衡 ✅；搜索/Pareto ✅；Anchor 验证 ✅
