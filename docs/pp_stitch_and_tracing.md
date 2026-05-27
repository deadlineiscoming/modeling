# PP 流水线拼接与 Chrome Trace 可视化

## 概述

本项目实现了**拓扑驱动的流水线并行（Pipeline Parallelism, PP）调度拼接**，以及将拼接结果导出为 **Chrome Trace JSON** 格式进行可视化分析。

### 核心思想

训练中的 PP 调度策略（如 1F1B、DualPipe、ZeroBubble）在 **stage × microbatch 网格** 上表现为三类依赖边的组合。通过显式构建网格任务并施加三类约束边，再用 list scheduler 贪心调度，即可自然产生正确的流水线重叠效果 — 无需硬编码公式。

### 涉及文件

| 文件 | 作用 |
|---|---|
| `python/zrt/executor/pp_stitcher.py` | PP 流水线拼接核心：网格构建、三类边、list scheduling |
| `python/zrt/executor/chrome_trace.py` | Chrome Trace JSON 导出：三种视图模式 |
| `demo_pp_stitcher.py` | PP 拼接演示：多种调度策略、Gantt 图 |
| `demo_trace_export.py` | Trace 导出演示：三种 JSON 输出 |

---

## 一、Stage × Microbatch 网格调度

### 1.1 网格定义

```
             m=0      m=1      m=2      ...      m=M-1
        ┌────────┬────────┬────────┬─────┬────────┐
  s=0   │ F₀ B₀  │ F₁ B₁  │ F₂ B₂  │ ... │ FM-1 BM-1│  ← GPU 0
        ├────────┼────────┼────────┼─────┼────────┤
  s=1   │ F₀ B₀  │ F₁ B₁  │ F₂ B₂  │ ... │ FM-1 BM-1│  ← GPU 1
        ├────────┼────────┼────────┼─────┼────────┤
  s=2   │ F₀ B₀  │ F₁ B₁  │ F₂ B₂  │ ... │ FM-1 BM-1│  ← GPU 2
        ├────────┼────────┼────────┼─────┼────────┤
  s=3   │ F₀ B₀  │ F₁ B₁  │ F₂ B₂  │ ... │ FM-1 BM-1│  ← GPU 3
        └────────┴────────┴────────┴─────┴────────┘
```

- **Stage（s）**：流水线的一个物理 GPU/Device，持有模型的连续若干层
- **Microbatch（m）**：一个训练数据子切片，是调度的最小原子单元
- **GridTask**：网格中的一个单元格，表示某个 stage 上某个 mb 的 **一次 FWD** 或 **一次 BWD**

### 1.2 三大类约束边（Edge Types）

三条边共同决定整个流水线的执行时序：

```
Edge ①  F→B activation dependency
        G[s][m].fwd → G[s][m].bwd
        （同一 stage+mb 内，前向必须先完成才能开始反向）

Edge ②  cross-stage P2P
        G[s][m].fwd   → G[s+1][m].fwd    （前向激活传递）
        G[s+1][m].bwd → G[s][m].bwd      （反向梯度回传）

Edge ③  device-serial protocol
        取决于调度策略：
        - 1F1B:   warmup chain(F₀→F₁→...→Fw-1) + alternating chain(B₀→Fw→B₁→Fw+1→...)
        - DualPipe: 双流反并行链
        - ZeroBubble: bwd 拆分为 bwd_dx 和 bwd_dw 两个子阶段
```

**关键设计**：Edge ③ 必须拆分为两条**独立**的链（warmup 链和 alternating 链），二者间无连接。若将 warmup 末端直接连到 alternating 首位（如 `Fw-1 → B₀`），会导致 B₀ 被错误地推迟到所有 warmup 前向完成之后，完全消除流水线重叠。

### 1.3 1F1B 调度详解

以 pp=4, M=6 为例，各 stage 的 warmup 前向次数 `w = pp - s`：

```
Stage 0 (w=4):  Chain A:  F₀ → F₁ → F₂ → F₃
                Chain B:            B₀ → F₄ → B₁ → F₅ → B₂ → B₃ → B₄ → B₅

Stage 1 (w=3):  Chain A:  F₀ → F₁ → F₂
                Chain B:         B₀ → F₃ → B₁ → F₄ → B₂ → F₅ → B₃ → B₄ → B₅

Stage 2 (w=2):  Chain A:  F₀ → F₁
                Chain B:      B₀ → F₂ → B₁ → F₃ → B₂ → F₄ → B₃ → F₅ → B₄ → B₅

Stage 3 (w=1):  Chain A:  F₀
                Chain B:  B₀ → F₁ → B₁ → F₂ → B₂ → F₃ → B₃ → F₄ → B₄ → F₅ → B₅
```

加上 Edge ②（跨 stage 前/反向 P2P），list scheduler 自然生成：

```
GPU 0: |F₀ F₁ F₂ F₃ ---- B₀ F₄ B₁ F₅ B₂ B₃ B₄ B₅|
GPU 1: |  F₀ F₁ F₂ -- B₀ F₃ B₁ F₄ B₂ F₅ B₃ B₄ B₅  |
GPU 2: |    F₀ F₁ B₀ F₂ B₁ F₃ B₂ F₄ B₃ F₅ B₄ B₅    |
GPU 3: |      F₀ B₀ F₁ B₁ F₂ B₂ F₃ B₃ F₄ B₄ F₅ B₅  |
        └warmup┘└──── alternating ─────┘└ cooldown ┘
```

### 1.4 List Scheduling 算法

```
输入: 带三类依赖边的 GridTask 集合
算法: Kahn 拓扑排序 + 贪心最早开始

1. 构建 in_degree: 每个 task 的未就绪前驱数
2. ready_queue = [所有 in_degree=0 的 task]
3. while ready_queue:
     a. 选 start_time 最小的 task（start = max(所有前驱完成时间, 设备就绪时间)）
     b. 将 task 放入调度序列，更新 finish_time 和 device_free_time
     c. 释放所有依赖此 task 的后继（in_degree -= 1；若为 0 则入 ready_queue）
```

### 1.5 关键指标

```
step_time  = 从第一个任务开始到最后一个任务结束的总时间
warmup     = 从开始到 stage(pp-1) 开始第一个 BWD 的时间
steady     = step_time - warmup - cooldown
cooldown   = 从 stage 0 开始最后一个 BWD 到结束的时间
bubble     = step_time - M × per_stage_bottleneck

公式验证（1F1B，同构 stage）:
  step_time  = (M + pp - 1) × per_stage
  bubble     = (pp - 1) × per_stage
```

---

## 二、Chrome Trace 导出（三种视图）

### 2.1 `stitched.json` — 流水线网格视图

**pid = stage_id，tid = stage_id（每个 stage 一行）**

每个事件 = 一个 GridTask（某个 stage 上某个 mb 的一次 FWD/BWD 完整块）。

```
pid=0 (GPU 0): [F₀][F₁][F₂][F₃]        [B₀][F₄][B₁][F₅][B₂][B₃][B₄][B₅]
pid=1 (GPU 1):    [F₀][F₁][F₂]      [B₀][F₃][B₁][F₄][B₂][F₅][B₃][B₄][B₅]
pid=2 (GPU 2):       [F₀][F₁]   [B₀][F₂][B₁][F₃][B₂][F₄][B₃][F₅][B₄][B₅]
pid=3 (GPU 3):          [F₀][B₀][F₁][B₁][F₂][B₂][F₃][B₃][F₄][B₄][F₅][B₅]
```

附带即时事件（ph="i"）标注 warmup/cooldown 分界线。

**适用场景**：全局宏观视角——看 PP 气泡分布、warmup/steady/cooldown 三段占比、跨 stage 的 FWD/BWD 级联。

### 2.2 `per_stage.json` — 单卡算子细节视图

**pid = stage_id，tid = stream_id（compute=0, TP comm=1, EP comm=2, ...）**

把 stitched 的每个大矩形"展开"成 DAGScheduler 记录的**具体算子序列**。每个 microbatch 的 fwd/bwd 算子按其对应的网格块 `fwd_base`/`bwd_base` 做时间平移。

```
pid=0 (GPU 0):
  tid=0 (compute): [matmul][attn][matmul][ffn]...[matmul_bwd][attn_bwd]...
  tid=1 (TP comm):    [all_reduce]                   [all_reduce]     ...
  tid=2 (EP comm):        [a2a_fwd]                  [a2a_bwd]       ...
```

所有 microbatch 共享同一组物理 stream 行（不按 mb 拆行），mb 号保存在 `args.mb` 中。

**适用场景**：微观视角——看 TP/EP 通信与计算的 overlap、通信气泡、算子粒度、单卡瓶颈分析。

### 2.3 `combined.json` — 两层叠加视图

**pid 0..pp-1 = 网格（同 stitched），pid pp..2*pp-1 = 细节（同 per_stage）**

网格在上方 pids，细节在下方 pids，同一 stage 垂直对齐。通过折叠/展开 pid 分组可在宏观和微观之间切换。

**适用场景**：在一个 trace 文件中同时呈现两种视角，适合做完整的端到端分析报告。

### 2.4 Chrome Trace 事件格式

```json
{
  "traceEvents": [
    {
      "ph": "X",
      "name": "▼ FWD [c] s0 m0",
      "cat": "compute",
      "pid": 0,
      "tid": 0,
      "ts": 0.0,
      "dur": 198.0,
      "args": {"phase": "fwd", "mb": 0, "stage": 0, "dep_count": 0}
    }
  ]
}
```

- `ph="X"`：完整事件（有 duration）
- `ph="i"`：即时事件（标注 warmup/cooldown 分界）
- `pid`：进程 = Stage/GPU
- `tid`：线程 = 流（compute/comm stream）
- `ts`/`dur`：起始时间和持续时长（微秒）

---

## 三、使用示例

### 3.1 PP 流水线拼接

```python
from python.zrt.executor.pp_stitcher import PPStitcher

stitcher = PPStitcher(
    stage_fwd_us={0: 100, 1: 80, 2: 120, 3: 90},   # 每 stage 前向耗时
    stage_bwd_us={0: 200, 1: 160, 2: 240, 3: 180},  # 每 stage 反向耗时
    pp=4, M=8,                                        # stage 数、microbatch 数
    p2p_latency_us=3,                                 # 单次跨 stage 传输
    schedule="1f1b",                                  # 调度策略
)
result = stitcher.stitch()
print(result.summary())
```

### 3.2 Chrome Trace 导出

```python
from python.zrt.executor.chrome_trace import ChromeTraceExporter

exporter = ChromeTraceExporter(time_unit="us")

# 模式 1: 纯网格
exporter.export_stitched(result, "stitched.json")

# 模式 2: 算子细节
exporter.export_per_stage(timelines, "per_stage.json", M=result.M, pp_stitched=result)

# 模式 3: 组合
exporter.export_combined(result, timelines, "combined.json")
```

### 3.3 运行 Demo

```bash
# PP 调度拼接演示
python demo_pp_stitcher.py

# Trace 导出演示
python demo_trace_export.py
# 输出: demo_trace/stitched.json, per_stage.json, combined.json
# 在 Chrome 中打开 chrome://tracing，加载任一 JSON 即可查看
```

---

## 四、设计决策与注意事项

1. **TP/EP/CP 特性已内嵌**：DAGScheduler 产出的 per-stage Timeline 已经包含了 TP all_reduce、EP all_to_all 等通信算子的耗时。PPStitcher 只负责跨 stage 的流水线编排，不重复计算 intra-stage 并行。

2. **List scheduler 非确定性**：当多个 task 具有相同的 start_time 时，出现顺序由 Python dict 遍历顺序决定。如果需要确定性的输出顺序，应在 grid 构建时对 task 做全排序。

3. **Bubble 公式**：`step_time - M × per_stage_bottleneck` 在异构 stage（各 stage 的 fwd+bwd 不同）时可能低估实际气泡，因为快 stage 的等待时间不仅取决于瓶颈 stage。此时应参考 stitched 视图中的直观空白段。

4. **去重**：`_chain_on_device` 中添加边时必须检查 `prev not in tasks[tid].dependencies`，避免 activation dep 和 chain dep 产生重复边 → in_degree 错误 → 任务永久挂起。