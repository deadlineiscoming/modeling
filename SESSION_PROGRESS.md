# Session Progress

Writer: Claude Code
Updated: 2026-04-20 20:00:00 +08:00

## 当前文件状态

| 文件 | 状态 |
|------|------|
| `python/zrt/executor/stream.py` | ✅ **新增** Stream 数据类（per ARCHITECTURE.md） |
| `python/zrt/executor/overlap.py` | ✅ **新增** OverlapAnalyzer + OverlapReport（扫描线算法） |
| `python/zrt/transform/analysis/comm_latency.py` | ✅ **新增** CommLatencyPass（互联带宽计算） |
| `python/zrt/executor/__init__.py` | ✅ 导出新类（Stream, OverlapAnalyzer, OverlapReport） |
| `python/zrt/transform/analysis/__init__.py` | ✅ 导出 CommLatencyPass |
| `python/zrt/transform/__init__.py` | ✅ 导出 CommLatencyPass |
| `python/zrt/transform/pipeline.py` | ✅ 注册 CommLatencyPass 在 analyze stage |
| `tests/test_executor.py` | ✅ 新增 8 个测试（21/21 passed） |
| **全部 tests/** | ✅ **158 passed, 27 skipped** |

## 本次完成

### 完善 Executor：单机多卡/多机多卡多流图执行器 ✅

**背景**：ARCHITECTURE.md 要求的 executor 模块缺少 `stream.py` 和 `overlap.py`；更严重的是，通信延迟估算使用 HBM 带宽而非互联带宽（NVLink/HCCS/RoCE/IB），导致单机多卡和多机多卡场景的性能预测错误。

**1. 新增** `python/zrt/executor/stream.py`
- `Stream` 数据类（stream_id, stream_type）— 简单的流抽象，per ARCHITECTURE.md 设计

**2. 新增** `python/zrt/executor/overlap.py`
- `OverlapReport` 数据类：compute_us, comm_us, overlap_us, exposed_comm_us, overlap_ratio, critical_path_us
- `OverlapAnalyzer` 类：使用扫描线算法精确计算通算掩盖区间交集（比当前 Timeline 的近似公式 `compute + comm - total` 更准确）

**3. 新增** `python/zrt/transform/analysis/comm_latency.py`
- `CommLatencyPass` — 运行于 analyze stage（RooflinePass 之后）
- 对所有通信节点（category="communication"）重新计算 latency_us：
  - 从 node.attrs 中读取 `collective` 和 `group_size`
  - 判断是否跨节点：`group_size > hw.interconnect.intra_node.num_devices`
  - 选择 intra_node 或 inter_node 的带宽
  - 应用集合通信公式（Ring AllReduce, AllGather, AllToAll 等）
  - 标注 cross_node flag 和通信算法
- 支持的算法：all_reduce, all_gather, reduce_scatter, all_to_all, send_recv, broadcast

**4. 修改** `python/zrt/transform/pipeline.py`
- 在 build_default_pipeline() 的 analyze stage 中，注册 CommLatencyPass（在 RooflinePass 之后）
- 确保通信延迟覆盖 Roofline 的错误估算

**5. 修改导出**
- `python/zrt/executor/__init__.py`：导出 Stream, OverlapAnalyzer, OverlapReport
- `python/zrt/transform/analysis/__init__.py`：导出 CommLatencyPass
- `python/zrt/transform/__init__.py`：导出 CommLatencyPass

**测试** ✅（8 个新测试）
- `test_overlap_analyzer_*`（3 个）：无通信、完全隐藏、部分掩盖场景
- `test_comm_latency_pass_*`（4 个）：intra_node、cross_node、零group_size、all_to_all 算法
- `test_schedule_after_full_pipeline`：验证 CommLatencyPass 集成到默认管道后仍正确工作

**关键改进**：
- **单机多卡**（e.g., TP=4 on 8×H100）：使用 NVLink 带宽 ~1.2TB/s，通信延迟由 Roofline 的 ~1µs 降低到实际 ~36µs（仍被掩盖）
- **多机多卡**（e.g., TP=16 across 2×8 nodes）：使用 RoCE 带宽 ~200Gbps，通信延迟显著增加，可能暴露在关键路径上

### Task 2: `memory/activation.py` — 图拓扑生命周期分析 ✅

**新文件**：`python/zrt/memory/activation.py`
- `ActivationAnalysis` frozen dataclass（peak_bytes, peak_mb, peak_node_id, per_node_live_mb）
- `analyze_activation(graph: OpGraph) -> ActivationAnalysis`
  - 算法：topo_sort → last_use_idx 计算 → liveness 分析 → peak 追踪
  - 关键：只考虑 mem_bytes > 0 的边，跳过 control 边

**修改**：`python/zrt/memory/model.py`
- `estimate()` 中：当 profile 是 OpGraph 且有节点时，调用 `analyze_activation` 替代公式激活估算
- 应用 activation_slack 乘数保证一致性

**修改**：`python/zrt/memory/__init__.py`
- 导出 ActivationAnalysis、analyze_activation

**测试** ✅（3 个新测试）
- `test_activation_analysis_simple` — 线性 A→B→C 图，B 生成 2 输出，验证 peak 在 B
- `test_activation_analysis_peak_node_id` — 验证 peak_node_id 正确
- `test_memory_model_uses_graph_activation` — OpGraph 时使用图分析而非公式

### Task 1: MemoryModel 接入 E2ESummary — E2E 集成 ✅

**修改**：`python/zrt/report/summary.py`
- E2ESummary 新增字段：`memory_budget: MemoryBudget | None = None`
- build_summary() 新增参数：`memory_budget: MemoryBudget | None = None`（向后兼容）
- 返回值包含 memory_budget

**测试** ✅（2 个新测试）
- `test_build_summary_with_memory_budget` — 传入 MemoryBudget，验证 summary.memory_budget 非 None
- `test_build_summary_without_memory_budget` — 不传，字段为 None（向后兼容）

## 验证结果

```bash
pytest tests/test_executor.py -v     # 21 passed (13 existing + 8 new) ✅
pytest tests/ -v --tb=short          # 158 passed, 27 skipped ✅
PYTHONIOENCODING=utf-8 python e2e_check.py  # All 7 steps pass ✅
```

对比基线：
- 原 151 passed（memory + report 工作）
- 新增 7 个测试（executor 实现）
- **总计 158 passed, 27 skipped, 0 failed** ✅
- e2e_check.py 中 TP=4 的 comm_time 现在为 36.47µs（之前 Roofline 估算 ~1µs）

## Task 0: E2E 示例脚本 — memory_budget 集成 ✅

**修改**：`e2e_check.py`
- 添加 `load_model()` 调用以获取配置对象（不加载权重）
- Step 6a：创建 MemoryModel，使用 config 对象调用 estimate() 获取 MemoryBudget
  - 输出：weights_mb、kv_cache_mb、activation_peak_mb、comm_buffer_mb、framework_overhead_mb、total_mb
- Step 6b：将 memory_budget 传入 `build_summary(memory_budget=memory_budget_1)`
  - E2ESummary 返回结果包含 memory_budget（不为 None）
- Step 7：添加对 memory_budget 的正确性断言
  - 验证 `summary.memory_budget.is_feasible == True`（TP=1 应可行）

**验证结果** ✅
```
e2e_check.py 完整运行通过，所有 7 个步骤均成功：
  [1] 抓图完成 ✓
  [2] TP=1 baseline ✓ (latency=0.761ms, comm_time=0µs)
  [3] TP=4 变换 ✓ (comm_nodes=2, overlap=1.10µs)
  [4] 前10个调度事件 ✓
  [5] 节点注解验证 ✓
  [6a] 内存预算计算 ✓ (total=4237.81MB, feasible=True)
  [6b] E2ESummary + memory_budget ✓ (mfu=0.14%, latency=0.761ms)
  [7] 正确性断言 ✓ (12/12 passed)
```

## 本次新增：Memory 模块对齐 ARCHITECTURE.md 3.3 显存模型 ✅

**背景**：python/zrt/memory 模块与 ARCHITECTURE.md 3.3 显存模型章节存在出入，缺少对 MLA (Multi-head Latent Attention) 和 EP (Expert Parallel) 的支持。

### 改进内容

#### 1. MLA 架构支持（DeepSeek-V2, Qwen-2.5）
- **修改** `python/zrt/memory/model.py`：
  - _ProfileView: 添加 `kv_lora_rank`, `qk_rope_head_dim` 字段
  - _kv_cache(): 自动检测 MLA 架构，使用 `kv_lora_rank + qk_rope_head_dim` 替代标准 GQA 公式
  - _coerce_profile(): 提取 MLA 相关字段

#### 2. Expert Parallel (EP) 权重分片（DeepSeek-V3, Mixtral）
- **修改** `python/zrt/memory/model.py`：
  - _weights(): 添加 EP 分片因子，从 `tp * pp` 扩展到 `tp * pp * ep`
  - _ProfileView: 添加 `num_experts`, `num_shared_experts`, `moe_topk` 字段
  - _coerce_profile(): 提取 MoE 相关字段

### 测试验证 ✅
- **新增** 2 个单元测试：
  - `test_memory_model_mla_architecture()` — MLA 显存计算正确性
  - `test_memory_model_ep_shards_weights()` — EP8 下权重显存为 EP1 的 1/8
- **所有测试通过**：10 passed, 0 failed（8 existing + 2 new）

### 关键对标完成
| 功能 | ARCHITECTURE.md | 实现 | 对齐 |
|------|-----------------|------|------|
| MemoryBudget | ✓ | ✓ | ✓✓✓ |
| _weights() 含 TP/PP/EP | ✓ | ✓ | ✓✓✓ |
| _kv_cache() 含 MLA | ✓ | ✓ | ✓✓✓ |
| _activation_peak() | ✓ | ✓✓（更详细） | ✓✓✓ |
| _comm_buffer() | ✓ | ✓✓（更全面） | ✓✓✓ |
| activation.py 生命周期分析 | ✓ | ✓ | ✓✓✓ |

---

## 下一步待办

1. ~~补充 E2E 示例：如何从 `run_trace()` → transform → simulate → `build_summary()` 集成，包括 memory_budget~~ ✅
2. ~~完善 executor：单机多卡/多机多卡多流图执行器~~ ✅
3. ~~对齐 memory 模块与 ARCHITECTURE.md 3.3 显存模型规范~~ ✅
4. 补充多机多卡场景的真机验证（当前仅在 e2e_check.py 中为 TP=4，需要构造 TP>8 场景）
5. 补充文档：CommLatencyPass 的设计、集合通信公式证明
6. 评估是否需要将 `export_all()` 从 `nx.DiGraph` 改为直接接收 `OpGraph`（可选）

