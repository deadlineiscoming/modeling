# `--estimate-config` 命令完整业务逻辑追踪

> 命令：`python -m python.zrt --estimate-config python/zrt/training/configs/deepseek_v4_pro_3d_h100.yaml`

本文档逐步追踪该命令的完整执行路径，从 CLI 入口到最终报告输出，附带每一步的关键代码。

---

## 全局流程概览

```
CLI 解析 → 加载 YAML 配置 → 构建三层 Spec 对象 → 校验 → 构建 IR 图
    → 插入集合通信 → 计算算子 FLOPs → 计算阶段时间 → 流水线编排
    → 内存估算 → MFU/HFU 计算 → 组装报告 → 输出
```

---

## Step 1：CLI 入口

**文件**: `python/zrt/cli.py:43-226`

```python
def main() -> None:
    parser = argparse.ArgumentParser(...)
    mode_group.add_argument("--estimate-config", metavar="YAML", ...)
    args = parser.parse_args()

    if args.estimate_config:
        _run_estimate(args.estimate_config, args.output)
        return
```

`--estimate-config` 是一个互斥模式标志，命中后直接调用 `_run_estimate()` 并提前返回，**不触发任何模型加载或图抓取**。

---

## Step 2：Spec 加载

**文件**: `python/zrt/cli.py:543-556`

```python
def _run_estimate(config_path: str, output_path: str | None) -> None:
    from python.zrt.training.io.config_loader import load_specs
    from python.zrt.training.search.estimator import estimate
    from python.zrt.training.search.report import report_summary, report_to_json

    model, system, strategy = load_specs(config_path)
    report = estimate(model, system, strategy)

    if output_path:
        report_to_json(report, output_path)
    else:
        print(report_summary(report))
```

### 2.1 YAML 配置文件内容

**文件**: `python/zrt/training/configs/deepseek_v4_pro_3d_h100.yaml`

```yaml
model: deepseek_v4_pro          # 引用 configs/models/deepseek_v4_pro.yaml

system:
  hw: nvidia_h100_sxm           # 引用 hardware/configs/nvidia_h100_sxm.yaml
  nodes: 8
  gpus_per_node: 8
  host_mem_gb: 2048

strategy:
  tp: 8
  cp: 1
  pp: 4
  ep: 64
  dp: 2
  micro_batch: 1
  global_batch: 512
  pp_schedule: dualpipev
  zero_stage: 1
  recompute:
    per_layer:
      moe: ["attn"]
  tp_overlap: none
  optimizer: muon
  muon_config:
    ns_steps: 10
    rotation: true
    adam_param_types: ["embed", "lm_head", "router", "bias"]
    muon_param_fraction: 0.85
```

### 2.2 配置解析流程

**文件**: `python/zrt/training/io/config_loader.py:20-29`

```python
def load_specs(config_path: str | Path) -> tuple[ModelSpec, SystemSpec, Strategy]:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = _resolve_model(cfg["model"])    # → 加载 configs/models/deepseek_v4_pro.yaml
    system = _parse_system(cfg["system"])   # → 加载 hardware/configs/nvidia_h100_sxm.yaml
    strategy = _parse_strategy(cfg["strategy"])

    return model, system, strategy
```

#### 2.2.1 模型解析

**文件**: `python/zrt/training/io/config_loader.py:74-146`

`_resolve_model("deepseek_v4_pro")` 将字符串引用解析为文件路径 `configs/models/deepseek_v4_pro.yaml`，然后调用 `_parse_model()` 构造 `ModelSpec`。

**DeepSeek-V4-Pro 模型关键参数**：

| 字段 | 值 | 含义 |
|------|-----|------|
| `hidden` | 7168 | 隐藏维度 |
| `num_heads` | 128 | 注意力头数 |
| `num_kv_heads` | 1 | MQA：单个 KV 头 |
| `head_dim` | 512 | 每头维度 |
| `vocab` | 129280 | 词表大小 |
| `seq_len` | 4096 | 序列长度 |
| `layers` | `[moe]*61+[mtp]*1` | 61 个 MoE 层 + 1 个 MTP 层 |
| `num_experts` | 384 | 总专家数 |
| `top_k` | 6 | 每个token激活6个专家 |
| `hc_mult` | 4 | Hyper-Connection 倍数 |
| `q_lora_rank` | 1536 | Q 低秩投影 |
| `o_groups` | 16 | O 投影分组数 |
| `routed_expert_dtype` | fp4 | 路由专家权重使用 FP4 |
| `compress_ratios` | [128,128,4,128,4,...] | 每层压缩比（CSA=4, HCA=128） |
| `n_hash_routed_layers` | 3 | 前三层使用 hash routing |

层重复语法解析（`_parse_layers`）支持 `[moe]*61+[mtp]*1` 格式，展开为 62 个 `LayerKind` 元素的列表。

#### 2.2.2 系统解析

**文件**: `python/zrt/training/io/config_loader.py:148-180`

```python
def _parse_system(d: dict) -> SystemSpec:
    from zrt.hardware import registry as hw_registry
    hw = hw_registry.load(d["hw"])    # 加载 nvidia_h100_sxm 硬件规格

    gpu = GPU(
        name=hw.name,
        flops_bf16=hw.compute.bf16_tflops,       # BF16 算力 (TFLOPS)
        flops_fp8=hw.compute.fp8_tops,            # FP8 算力
        hbm_gb=hw.memory.capacity_gb,             # HBM 容量
        hbm_bw_gbps=hw.memory.hbm_bandwidth_gbps, # HBM 带宽
    )
    nets = [
        NetTier(scope="intra_node", ...),   # NVLink 域内
        NetTier(scope="inter_node", ...),    # InfiniBand 跨节点
    ]
    return SystemSpec(gpu=gpu, nets=nets, nodes=8, gpus_per_node=8)
```

#### 2.2.3 策略解析

**文件**: `python/zrt/training/io/config_loader.py:183-233`

```python
def _parse_strategy(d: dict) -> Strategy:
    return Strategy(
        tp=8, cp=1, pp=4, ep=64, dp=2,
        micro_batch=1, global_batch=512,
        pp_schedule=PPSched.DUALPIPE_V,
        zero_stage=1,
        recompute=RecomputePolicy(per_layer={"moe": {"attn"}}),
        optimizer=OptKind.MUON,
        muon_config=MuonConfig(ns_steps=10, rotation=True, muon_param_fraction=0.85),
    )
```

**并行维度**: TP=8 × CP=1 × PP=4 × EP=64 × DP=2 = 4096，但实际 world_size = 8×8 = 64。

`Strategy.validate()` 确保并行维度乘积等于 world_size。

---

## Step 3：校验

**文件**: `python/zrt/training/search/estimator.py:25-34`

```python
def estimate(model, system, strategy) -> TrainingReport:
    strategy.validate(model, system)       # 维度乘积校验
    warnings = ir_validate(model, system, strategy)  # IR 约束校验
```

### 3.1 Strategy 校验

`Strategy.validate()` 检查：
- `tp * cp * pp * ep * dp == system.world_size`
- 各维度 > 0

### 3.2 IR 校验

**文件**: `python/zrt/training/ir/validate.py:10-125`

检查项包括：
- `num_layers % pp == 0`（PP 层数均分）
- `num_heads % tp == 0`（TP 头数均分）
- `num_experts % ep == 0`（EP 专家数均分）
- CP 类型相关约束（Ulysses 头数、Ring 序列长度）
- 跨节点通信警告

---

## Step 4：构建 IR 图

**文件**: `python/zrt/training/search/estimator.py:37`

```python
graph = build_graph(model, strategy)
```

### 4.1 build_graph 主流程

**文件**: `python/zrt/training/ir/builders.py:983-1051`

```python
def build_graph(model: ModelSpec, strategy: Strategy) -> Graph:
    all_ops: list[Op] = []
    layer_index: dict[int, tuple[int, int]] = {}

    # 1. Embedding
    all_ops.append(_embed_op(model.vocab, h, s, act_dtype))

    # 2. HC expansion (hc_mult=4 → 扩展 x→(x, hc_mult, h))
    if hc_mult > 1:
        all_ops.append(_hc_expand_op(s, h, hc_mult, act_dtype))

    # 3. Transformer blocks
    for i, lk in enumerate(model.layers):
        start = len(all_ops)
        if lk == LayerKind.MOE:
            block_ops = _moe_block(model=model, ...)
        elif lk == LayerKind.MTP:
            block_ops = _mtp_block(model=model, ...)
        all_ops.extend(block_ops)
        layer_index[i] = (start, len(all_ops))

    # 4. HC head mix-down
    if hc_mult > 1:
        all_ops.append(_mhc_head_op(s, h, hc_mult, act_dtype))

    # 5. Final LN + lm_head
    all_ops.append(_final_ln_op(h, s, act_dtype))
    all_ops.append(_lm_head_op(model.vocab, h, s, act_dtype))

    graph = Graph(ops=all_ops, collectives=[], layer_index=layer_index)

    # 6. 分片 + 插入集合通信
    shard = ShardPlan(strategy)
    insert_collectives(graph, model, strategy)

    return graph
```

### 4.2 V4 MoE 层算子构建

V4 的 MoE 层结构（HC=true 时）：

```
mhc_pre_attn → ln1 → wq_a → q_norm → wq_b → q_rsqrt_norm → wkv → kv_norm
→ comp_wkv → comp_wgate → comp_pool       ← KV 压缩器（CSA/HCA）
→ idx_wq_b → idx_weights → idx_comp_wkv → idx_comp_wgate → idx_comp_pool → idx_score_topk  ← Indexer（仅CSA）
→ rope → sparse_attn / hca_attn          ← 注意力核心
→ wo_a → wo_b                             ← 分组输出投影
→ mhc_post_attn → mhc_pre_ffn
→ ln2
→ router → topk_select                    ← MoE 路由
→ shared_up_proj → shared_gate_proj → shared_swiGLU → shared_down_proj  ← 共享专家
→ routed_expert_ffn                        ← 路由专家（fwd_multiplier = 3×top_k = 18）
→ expert_agg                               ← 专家聚合
→ mhc_post_ffn
```

**关键代码**: `_build_v4_attn()` 在 `builders.py:212-330` 构建 V4 注意力子块，按 `get_layer_cp_type()` 分发 CSA/HCA/SWA 路径。

**每个 Op 携带**:
- `name`: 算子唯一名称（如 `L5.wq_a`）
- `kind`: 算子类型（`matmul`、`attn_core`、`ln`、`swiglu` 等）
- `inputs`/`outputs`: `Tensor` 列表，含 `shape_logical` 和 `shape_local`
- `meta`: 计算参数（如 `m, n, k` 对于 matmul，`b, s, heads, head_dim` 对于注意力）
- `layer_id`/`layer_kind`: 所属层索引和类型

### 4.3 分片与集合通信插入

**文件**: `python/zrt/training/ir/shard.py:54-87`

```python
def insert_collectives(graph, model, strategy):
    shard = ShardPlan(strategy)
    collectives = []

    if shard.has_tp:
        _insert_tp_collectives(graph, shard, model, collectives)  # AG/RS 对
    if shard.has_cp:
        _insert_cp_collectives(graph, shard, model, collectives)  # A2A/P2P
    if shard.has_ep:
        _insert_ep_collectives(graph, shard, model, collectives)  # A2A 对

    graph.collectives.extend(collectives)
```

对于 V4-Pro 的配置 (TP=8, EP=64)：

**TP 集合通信**（每层）：
- AG（AllGather）在 QKV 投影前 → 聚合序列分片
- RS（ReduceScatter）在 O 投影后 → 散射归约
- AG 在 FFN up_proj 前 → 聚合序列分片
- RS 在 FFN down_proj 后 → 散射归约

**EP 集合通信**（仅 MoE 层）：
- A2A 在 routed_expert_ffn 前 → 路由 token 到专家所在 rank
- A2A 在 routed_expert_ffn 后 → 收回专家计算结果

**TP 分片修改**（`_apply_tp_sharding`）:
- Column-parallel 算子：`n_local = n // tp`（如 wq_a, wq_b, wkv）
- Row-parallel 算子：`k_local = k // tp`（如 wo_a, wo_b）
- Attention：`heads_local = heads // tp`
- Memory-bound 算子：`bytes_fwd //= tp`

**EP 分片修改**（`_apply_ep_sharding`）:
- routed_expert_ffn：`fwd_multiplier *= experts_per_rank / num_experts`
- Router 输出形状：`num_experts → experts_per_rank`

---

## Step 5：计算总训练 FLOPs

**文件**: `python/zrt/training/search/estimator.py:40`

```python
total_flops = total_training_flops(graph, model, strategy)
```

### 5.1 per-op FLOPs 计算

**文件**: `python/zrt/training/models/flops.py:27-52`

```python
def op_cost(op: Op, model: ModelSpec) -> OpCost:
    if op.kind == "matmul":    return _matmul_cost(op)
    if op.kind == "attn_core": return _attn_cost(op, model)
    if op.kind == "mhc_pre":   return _mhc_pre_cost(op)
    if op.kind == "mhc_post":  return _mhc_post_cost(op)
    if op.kind in ("ln", "softmax", "rope", "swiglu", "add"):
        return _memory_bound_cost(op)
    ...
```

**关键公式**：

| 算子类型 | Forward FLOPs | Backward (dx) | Backward (dw) | 绑定类型 |
|---------|--------------|---------------|---------------|---------|
| **MatMul** | `2 × m × n_local × k_local × fwd_multiplier` | = fwd | = fwd | compute |
| **Standard Attn** | `2 × b × s² × heads × head_dim` | 2.5 × fwd | 0 | compute |
| **CSA Attn** | `2 × b × s × (topk + swa) × heads × d` | 2.5 × fwd | 0 | compute |
| **HCA Attn** | `2 × b × s × (s/ratio + swa) × heads × d` | 2.5 × fwd | 0 | compute |
| **LayerNorm** | `bytes_fwd = s × h × 2` | 1.5 × bytes | 0 | memory |
| **SwiGLU** | `bytes_fwd = s × ffn × 3` | 1.5 × bytes | 0 | memory |

**Total training FLOPs**:

```python
total = Σ(fwd_flops + dx_flops + dw_flops) × num_microbatches
```

其中 `num_microbatches = global_batch / (micro_batch × dp) = 512 / (1 × 2) = 256`

---

## Step 6：流水线步时计算

**文件**: `python/zrt/training/search/estimator.py:43`

```python
step_result: StepResult = pipeline_step_time(graph, model, system, strategy)
```

### 6.1 主流程

**文件**: `python/zrt/training/compose/schedules.py:354-427`

```python
def pipeline_step_time(graph, model, system, strategy) -> StepResult:
    pp = strategy.pp           # 4
    M = strategy.num_microbatches()  # 256

    # 1. PP stage 分配
    stage_ids = _assign_stages(model, strategy)  # 贪心装箱

    # 2. 逐 stage 计算时间
    for s in range(pp):
        stage_ops = graph.ops_for_stage(stage_ids[s])
        stage_colls = [匹配该 stage 的集合通信]
        st = stage_time(stage_ops, stage_colls, model, system, strategy)
        stage_times.append(st)

    # 3. DP AllReduce 时间
    comm_times = total_comm_time(graph, model, system, strategy)
    dp_ar_time = comm_times["dp_grad_reduce"]

    # 4. 根据调度策略组合流水线
    composer = DualPipeVComposer()
    step = composer.compose(stage_times, M, pp, dp_ar_time, strategy)

    # 5. 优化器时间
    step.optimizer_time = _compute_optimizer_time(...)
    step.optimizer_comm = _compute_optimizer_comm_time(...)

    # 6. 内存估算
    step.memory = memory_breakdown(graph, model, system, strategy)

    # 7. MFU/HFU（在加优化器时间之前计算）
    step.mfu = compute_mfu(...)
    step.hfu = compute_hfu(...)

    # 8. 加上优化器时间
    step.step_time += step.optimizer_time + step.optimizer_comm

    return step
```

### 6.2 阶段时间计算

**文件**: `python/zrt/training/compose/stage.py:76-180`

```python
def stage_time(stage_ops, stage_collectives, model, system, strategy) -> StageTime:
    t_fwd = t_bwd_dx = t_bwd_dw = 0.0

    for op in stage_ops:
        cost = op_cost(op, model)
        if cost.bound == "compute":
            fwd_t = op_to_time(cost.fwd_flops, 0, "compute", system, ...)
            dx_t  = op_to_time(cost.dx_flops, 0, "compute", system, ...)
            dw_t  = op_to_time(cost.dw_flops, 0, "compute", system, ...)
        else:
            fwd_t = op_to_time(0, cost.fwd_bytes, "memory", system, ...)
            ...
        t_fwd += fwd_t; t_bwd_dx += dx_t; t_bwd_dw += dw_t

    # 重计算：在 backward 前重跑 forward 的部分算子
    recompute_t = _recompute_time(stage_ops, model, system, strategy, ...)
    t_bwd_dx += recompute_t

    # 集合通信时间
    for c in stage_collectives:
        ct = collective_time(c, group_size, tier)
        t_comm_fwd += ct * 0.5  # TP/EP 的通信均分到 fwd/bwd
        t_comm_bwd += ct * 0.5

    # EP 负载不均衡因子
    if strategy.ep > 1:
        imb = ep_imbalance_factor(num_experts, ep, topk)
        t_fwd = t_fwd * (1-ep_frac) + t_fwd * ep_frac * imb

    return StageTime(fwd=t_fwd+t_comm_fwd, bwd=t_bwd_dx+t_bwd_dw+t_comm_bwd, ...)
```

### 6.3 Roofline 模型

**文件**: `python/zrt/training/compose/stage.py:52-73`

```python
def op_to_time(flops, bytes_, bound, system, gpu_name, dtype) -> float:
    if bound == "compute" and flops > 0:
        peak = gpu.flops_bf16 * 1e12        # TFLOP/s → FLOP/s
        eff = achieved_flops_efficiency(gpu_name, dtype, flops)
        return flops / (peak * eff)

    elif bound == "memory" and bytes_ > 0:
        bw = gpu.hbm_bw_gbps * 1e9 / 8      # GB/s → bytes/s
        eff = achieved_bandwidth_efficiency(gpu_name, bytes_)
        return bytes_ / (bw * eff)
```

`achieved_flops_efficiency` 和 `achieved_bandwidth_efficiency` 来自 perf_tables，根据 GPU 型号和算子规模返回实际可达效率。

### 6.4 通信模型

**文件**: `python/zrt/training/models/comm.py:14-46`

Alpha-beta 模型：

```python
def collective_time(c, group_size, tier) -> float:
    alpha = tier.latency_us * 1e-6         # 链路延迟 (s)
    bw_bytes = tier.bw_gbps * 1e9 / 8      # 带宽 (bytes/s)
    beta = 1.0 / bw_bytes                  # 每字节传输时间

    if c.kind == "AG":  return (N-1) * (alpha + S/N * beta)
    if c.kind == "RS":  return (N-1) * (alpha + S/N * beta)
    if c.kind == "A2A": return (N-1) * (alpha + S/N * beta)
    if c.kind == "P2P": return rounds * (alpha + S * beta)
```

**拓扑选择**（`tier_for_group`）:
- TP (group_size=8 ≤ 8 GPUs/node) → intra_node (NVLink)
- EP (group_size=64 > 8) → inter_node (InfiniBand)
- DP (group_size=2 ≤ 8) → intra_node

### 6.5 DualPipe-V 流水线编排

**文件**: `python/zrt/training/compose/schedules.py:246-291`

```python
class DualPipeVComposer(PipelineComposer):
    def compose(self, stage_times, M, pp, dp_ar_time, strategy) -> StepResult:
        V = strategy.vpp_chunks  # 默认 1
        t_stage_max = max(st.fwd + st.bwd for st in stage_times)

        bubble = (pp - 1) / (2.0 * V) * t_stage_max
        warmup = bubble / 2.0
        cooldown = bubble / 2.0
        steady = M * t_stage_max

        # DP AR 可隐藏在 bubble 中
        if strategy.dp_overlap_in_bubble:
            hidden = min(bubble, dp_ar_time)
            dp_exposed = dp_ar_time - hidden

        step = warmup + steady + cooldown + dp_exposed
        bubble_frac = bubble / step

        return StepResult(step_time=step, bubble_fraction=bubble_frac, ...)
```

**公式**: bubble = `(pp-1) / (2V)` × t_stage，比标准 1F1B 的 `(pp-1) × max(t_fwd) + (pp-1) × max(t_bwd)` 更优。

### 6.6 EP 负载不均衡

**文件**: `python/zrt/training/compose/stage.py:32-49`

```python
def ep_imbalance_factor(num_experts, ep, topk) -> float:
    experts_per_gpu = num_experts / ep  # 384/64 = 6
    factor = 1 + (topk / experts_per_gpu) * sqrt(log(experts_per_gpu))
    # ≈ 1 + (6/6) * sqrt(log(6)) ≈ 1 + 1 × 1.258 ≈ 2.258
    return max(factor, 1.0)
```

仅应用于 EP 并行部分的计算时间（routed_expert_ffn）。

---

## Step 7：内存估算

**文件**: `python/zrt/training/models/memory.py:43-115`

```python
def memory_breakdown(graph, model, system, strategy) -> MemBreakdown:
    P = _params_on_rank(model, strategy)

    # 权重：区分 FP4 专家权重和 BF16 其他权重
    use_fp4 = model.routed_expert_dtype == "fp4"
    if use_fp4:
        P_expert = _routed_expert_params_on_rank(model, strategy)
        expert_weight_bytes = P_expert * 0.5 + (P_expert / 32) * 2  # FP4 + block scale
        weights = expert_weight_bytes + (P - P_expert) * BF16_BYTES

    grads = P * FP32_BYTES
    opt_state = _optimizer_state_bytes(P, model, strategy)  # Muon: P×(12-0.85×4)B

    # ZeRO-1: 仅分片优化器状态
    opt_state //= dp

    activations, hc_overhead = _activation_memory(model, strategy, layer_ids)
    comm_buffers = _comm_buffer_memory(model, strategy)

    return MemBreakdown(weights, grads, opt_state, activations, comm_buffers)
```

### 7.1 参数量计算

```python
def _params_on_rank(model, strategy):
    # Dense 层: 4h² + 2h×ffn per layer
    # MoE 层: (num_experts + n_shared) × 2h×moe_ffn per layer
    # V4: 384 routed + 1 shared expert per MoE layer

    total = dense_params + moe_params

    total //= tp                # TP 切分
    # EP 仅切分 routed expert 参数
    routed_after_ep = routed_params // ep
    # PP: 只保留本 stage 的层
    total = non_embed * (layers_per_stage / total_layers) + embed / pp

    return total
```

### 7.2 优化器状态

```python
def _optimizer_state_bytes(P, model, strategy):
    # Muon: P × (12 - f_muon × 4)
    # f_muon = 0.85 → 每参数 12 - 3.4 = 8.6 字节
    return int(P * (12 - 0.85 * 4))
```

### 7.3 激活内存

```python
def _activation_memory(model, strategy, layer_ids):
    # Korthikanti 系数: dense=10, moe=14, mtp=12
    # HC overhead: (hc_mult-1) × s × h × 2
    # TP SP + CP 分片: total_seq_shard = tp × cp

    layer_act = s × h × bytes × coeff + hc_overhead
    layer_act //= (tp × cp)
    total_act = Σ layer_act × micro_batch × pp_in_flight
```

### 7.4 通信缓冲区

```python
def _comm_buffer_memory(model, strategy):
    # TP AG/RS: 4 × s × h_tp × bytes × layers × micro_batch
    # EP A2A: 4 × s × h_tp × bytes × n_moe × micro_batch
```

---

## Step 8：MFU / HFU 计算

**文件**: `python/zrt/training/compose/schedules.py:516-555`

### 8.1 MFU（Model FLOPs Utilization）

```python
def compute_mfu(model, strategy, system, step_time) -> float:
    P = model.effective_params_for_flops()  # MoE: 只计算 top_k 激活的专家
    tokens = global_batch × seq_len          # 512 × 4096
    model_flops = 6.0 × P × tokens          # 6P 规则
    peak = gpu.flops_bf16 × 1e12 × world_size
    return model_flops / (peak × step_time)
```

`effective_params_for_flops()` 对 MoE 模型只计算 `top_k/num_experts` 比例的路由专家参数，而非全部专家。

### 8.2 HFU（Hardware FLOPs Utilization）

```python
def compute_hfu(model, strategy, system, step_time, graph) -> float:
    model_flops = 6.0 × P × tokens
    rc_overhead = recompute_overhead_flops(graph, model, strategy)
    return (model_flops + rc_overhead) / (peak × step_time)
```

HFU 包含重计算的额外 FLOPs，所以 `HFU ≥ MFU`。

### 8.3 重计算开销

**文件**: `python/zrt/training/models/flops.py:267-301`

```python
def recompute_overhead_flops(graph, model, strategy) -> float:
    # 配置: per_layer = {"moe": {"attn"}}
    # 意味着: 在 MoE 层的 backward 前，重跑 attn 类算子的 forward

    for op in graph.ops:
        lk = model.layers[op.layer_id].value     # "moe"
        cats = policy.get(lk)                      # {"attn"}
        op_cats = _op_recompute_categories(op)     # attn_core → {"attn"}

        if op_cats & cats:  # 匹配！
            extra += op_cost(op, model).fwd_flops  # 额外的 forward FLOPs

    return extra × num_microbatches
```

---

## Step 9：优化器时间

**文件**: `python/zrt/training/compose/schedules.py:456-497`

```python
def _compute_optimizer_time(model, system, strategy) -> float:
    P = params_on_rank_after_tp_pp_zer3(model, strategy)
    peak = gpu.flops_bf16 * 1e12

    if strategy.optimizer == "muon":
        K = 10  # Newton-Schulz 迭代步数
        f_muon = 0.85
        flops = muon_optimizer_step_flops(P, K, hidden, f_muon)
        eff = achieved_flops_efficiency(gpu.name, BF16, flops)
        return flops / (peak * eff)
```

### Muon 优化器通信

**文件**: `python/zrt/training/models/comm.py:109-172`

```python
def optimizer_comm_time(model, system, strategy) -> dict:
    # ZeRO-1 + Muon:
    #   AllGather: 收集完整 Muon 参数 → P_muon × 4B
    #   ReduceScatter (if rotation): 分发更新梯度 → P_muon × 4B
    P_muon = int(P * 0.85)
    comm_bytes = P_muon * 4  # FP32 master copy
    ag_time = collective_time(AG, dp, tier)
    rs_time = collective_time(RS, dp, tier)  # rotation=true
```

---

## Step 10：组装报告

**文件**: `python/zrt/training/search/estimator.py:57-68`

```python
return TrainingReport(
    step_time_ms=step_result.step_time * 1000,
    mfu=step_result.mfu,
    hfu=step_result.hfu,
    memory=step_result.memory,
    per_stage=step_result.per_stage,
    total_flops=total_flops,
    warnings=warnings,
    config_summary={
        "model": "hidden=7168, layers=62, heads=128",
        "system": "NVIDIA H100 SXM x 64",
        "strategy": "TP=8 CP=1 PP=4 EP=64 DP=2",
        ...
    },
    bubble_fraction=step_result.bubble_fraction,
    schedule_name="dualpipev",
)
```

---

## Step 11：输出

**文件**: `python/zrt/cli.py:552-556`

```python
if output_path:
    report_to_json(report, output_path)   # JSON 文件
else:
    print(report_summary(report))          # 控制台文本
```

**控制台输出格式** (`python/zrt/training/search/report.py:47-89`):

```
============================================================
Training Estimation Report
============================================================
  Model:    hidden=7168, layers=62, heads=128
  System:   NVIDIA H100 SXM x 64
  Strategy: TP=8 CP=1 PP=4 EP=64 DP=2
  Microbatches: 256

  Step time:  XXX.X ms
  MFU:        XX.X%
  HFU:        XX.X%
  Memory:
    weights:     XX.XX GB
    grads:       XX.XX GB
    opt_state:   XX.XX GB
    activations: XX.XX GB
    comm_buf:    XX.XX GB
    TOTAL:       XX.XX GB
  Per-stage times:
    Stage 0: fwd=XX.XXms  bwd=XX.XXms  total=XX.XXms
    Stage 1: ...
============================================================
```

---

## 总结：关键路径与核心公式

### 数据流

```
YAML → ModelSpec + SystemSpec + Strategy
                          ↓
              build_graph(ModelSpec, Strategy)
                          ↓
              Graph{ops[], collectives[], layer_index{}}
                          ↓
     ┌────────────────────┼────────────────────┐
     ↓                    ↓                    ↓
total_training_flops   stage_time()      memory_breakdown()
(flops.py)            (stage.py)          (memory.py)
     ↓                    ↓                    ↓
  Σ op_cost()      roofline: flops/peak   MemBreakdown
  × M              roofline: bytes/bw       ↓
                   + comm (alpha-beta)     weights + grads
                   + EP imbalance          + opt_state + act
                   + recompute             + comm_buffers
                          ↓
              pipeline_step_time()
              (schedules.py)
                          ↓
              DualPipeVComposer.compose()
              → StepResult{step_time, bubble_fraction, ...}
                          ↓
              compute_mfu() / compute_hfu()
              → mfu, hfu
                          ↓
              TrainingReport → print / JSON
```

### 核心公式速查

| 指标 | 公式 |
|------|------|
| **MatMul FLOPs** | `2 × m × n_local × k_local × fwd_multiplier` |
| **CSA Attn FLOPs** | `2 × b × s × (index_topk + swa_window) × heads × head_dim` |
| **HCA Attn FLOPs** | `2 × b × s × (s/compress_ratio + swa_window) × heads × head_dim` |
| **Compute 时间** | `flops / (peak_tflops × 1e12 × efficiency)` |
| **Memory 时间** | `bytes / (hbm_bw × 1e9/8 × efficiency)` |
| **AG/RS 通信** | `(N-1) × (α + S/N × β)` |
| **A2A 通信** | `(N-1) × (α + S/N × β)` |
| **P2P 通信** | `rounds × (α + S × β)` |
| **EP 不均衡因子** | `1 + (topk / experts_per_gpu) × √(ln(experts_per_gpu))` |
| **MFU** | `6P_eff × tokens / (world_size × peak × step_time)` |
| **HFU** | `(6P_eff × tokens + recompute_flops) / (world_size × peak × step_time)` |
| **DualPipe-V bubble** | `(pp-1) / (2V) × t_stage_max` |
| **Step time** | `bubble + M × t_stage + dp_exposed + opt_time + opt_comm` |
| **Muon opt state** | `P × (12 - f_muon × 4)` bytes |
| **FP4 expert 权重** | `P_expert × 0.5 + (P_expert/32) × 2` bytes |
