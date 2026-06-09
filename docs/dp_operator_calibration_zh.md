# DP 并行算子影响分析与抓图校准记录

本文面向抓图路径的端到端校准，重点回答 DP 并行是否正确影响算子、通信、流水线和最终报表口径。结论先行：

- DP 不应改写普通前向/反向计算算子的 activation shape 或 dtype；它主要改变每个 DP rank 的 microbatch 数、ZeRO 分片、梯度同步通信和通信暴露尾巴。
- 当前 DeepSeek-V4 抓图融合规则库存是可解释的：注册 23 条，训练激活 21 条，训练显式禁用 `kv_compressor` 和 `sparse_indexer`。HC/MHC 前向融合已清理干净，但训练全图仍存在注意力 mask/cache、MoE routing/quant、反向 bookkeeping 等小算子，需要结合手撕模型和实测决定是否继续融合。
- 反向 `dx/dw` 已在训练建模侧有字段和估算；ZeroBubble trace path 已显式拆成 `bwd_dx`/`bwd_dw`，DualPipe/DualPipeV 的 spec/composer path 会用 `bwd_dw` 降低 bubble，但当前 trace/e2e 仍有 gap：`--pp-schedule dualpipe` 报表仍落到 `schedule_name=1f1b`。这属于 PP/DualPipe 依赖项，不在 DP part 内直接修复。
- “跳 layer” 的主要处理已经在 PP 非 layer 节点分配和 stage fallback 中实现；Excel 报表已有重复/代表层显示，HTML 报表还未镜像这套 layer range 标签。

## 证据来源

本次分析基于以下代码与已有输出，不修改 `hf_models/`：

| 主题 | 代码/输出 |
|---|---|
| 融合规则加载 | `python/zrt/transform/fusion/loading/rule_set_initializer.py`, `python/zrt/transform/fusion/configs/deepseek_v4.yaml`, `python/zrt/transform/fusion/rules/*.yaml` |
| DP 通信插入 | `python/zrt/transform/parallel/data_parallel.py` |
| TP/EP/CP/PP shape 与通信 | `python/zrt/transform/parallel/tensor_parallel.py`, `expert_parallel.py`, `context_parallel.py`, `pipeline_parallel.py`, `comm_inserter.py` |
| dx/dw 与流水线 | `python/zrt/transform/analysis/training.py`, `python/zrt/executor/pp_stitcher.py`, `python/zrt/training/compose/stage.py` |
| 跳 layer 与重复层 | `python/zrt/transform/parallel/pipeline_parallel.py`, `python/zrt/transform/exporter.py`, `tests/transform/test_layer_display_map.py` |
| e2e 输出 | `output/dp_compare_ddp_quant_fp4_after_hc_rules`, `output/hc_calib_bs2_1f1b`, `output/hc_calib_bs1_dualpipe` |

## 1. 融合规则库存与融合前后状态

### 规则库存

DeepSeek-V4 规则由 `_common.yaml` 和 `deepseek_v4.yaml` 两部分组成：

| 来源 | 注册规则数 | 说明 |
|---|---:|---|
| `_common.yaml` | 7 | `rms_norm`, `rms_norm_nn`, `rms_norm_inline`, `rms_coef`, `cross_entropy`, `dropout`, `add_norm` |
| `deepseek_v4.yaml` | 16 | DeepSeek-V4 专有规则，如 `linear`, `rotary_emb`, `moe_gate`, `hc_pre_attn`, `hc_post_attn`, `hc_head`, `sparse_attention_kernel` |
| 合计 | 23 | 当前 loader 实测注册数 |

按 `deepseek_v4.yaml` 的 phase config：

| phase | 激活规则数 | 关键差异 |
|---|---:|---|
| training | 21 | 禁用 `kv_compressor`, `sparse_indexer`；保留 `sparse_attention_kernel` |
| inference | 21 | 启用 `kv_compressor`, `sparse_indexer`；不启用训练态 `cross_entropy`, `dropout` |

这符合当前设计意图：训练态不要把 KV-cache 写入和 sparse indexer 这类有状态/路由逻辑折叠掉，以免破坏 autograd 链；纯计算子图可以融合成分析级算子。

### 融合前后对比

以 `output/dp_compare_ddp_quant_fp4_after_hc_rules` 为例，原始 DOT 仅用于观察模式分布，转换后精确数量以 JSON `nodes` 为准：

| 图 | 节点数 | 唯一 op label/type | 观察 |
|---|---:|---:|---|
| raw train_forward DOT | 1240 | 71 | `final_norm`, MoE expert quant/mm, `attn.score`, `hc.pre_*` 等小算子较多 |
| raw train_backward DOT | 1309 | 67 | `hc.pre_*`, `hc.post_*`, MoE backward/routing、`scalar_tensor` 等反向碎片较多 |
| transformed JSON, bs=1 | 1808 | 102 左右 | 已融合计算子图，同时插入 DP/TP/EP/PP 通信、优化器和训练分析节点 |
| transformed JSON, bs=2 | 1830 | 约 102 | batch 放大后额外节点很少，主要影响 FLOPs/latency 而非结构 |

注意：raw forward/backward 与 transformed unified 不能直接用节点数相减判断融合收益，因为 transform 会插入通信、重计算、优化器和并行标注节点。更可靠的判断是看融合后 op_type 分布和 leftover 小算子是否仍在同一模块语义内。

融合后主要 fused op 计数如下：

| fused op_type | bs=1 数量 | bs=2 数量 | 说明 |
|---|---:|---:|---|
| `linear` | 117 | 118 | 通用线性子图，仍需关注 backward fragment 缺少 `weight` role 的 warning |
| `rms_norm` | 23 | 23 | 标准 RMSNorm 融合 |
| `column_parallel_linear` | 8 | 8 | TP column rule |
| `row_parallel_linear` | 4 | 4 | TP row rule，后续会插入 TP all-reduce |
| `hc_pre` | 8 | 8 | HC pre attention/FFN 已按真实 `[B,S,HC*H]` shape 校准 |
| `hc_post` | 8 | 8 | HC post attention/FFN |
| `hc_head` | 1 | 1 | HC head |
| `sparse_attention_kernel` | 4 | 4 | 当前仍使用统一有效长度 proxy |
| `moe_gate` | 3 | 3 | MoE gate |
| `rms_norm_inline` | 4 | 4 | inline RMSNorm |
| `parallel_embedding` | 1 | 1 | embedding |

### leftover 小算子判断

当前可以认为“前向 HC/MHC 语义块”已经干净：e2e transformed graph 中 `hc_pre=8`, `hc_post=8`, `hc_head=1`，前向 HC scope 下没有应被 HC rule 吃掉的小算子残留。

补充核验：`temp_output/ep_domain_report_tp8_ep8_dp8_hcpost_fix/deepseek_v4_train_forward.json` 中 `hc_post=8`，其中 `HCPostFfn` scope 下有 4 个前向 `hc_post` fused node；同 scope 剩余的 `aten.mul.Tensor`、`aten.sum.dim_IntList`、`aten.add.Tensor` 是 `bwd_op_*`，属于反向 `dx/dw` 后续处理范围，不应算作前向 HC post 融合失败。当前工作树的规则命名仍以 `hc_post_attn` 为主，文档和报表侧建议后续把 attn/ffn 子类来源显式写入 fused node attrs，便于审计。

仍需校准的 leftover 分三类：

| leftover 类别 | 是否建议立即融合 | 原因 |
|---|---|---|
| attention mask/cache/bookkeeping，例如 `attn.empty`, `attn.copy_`, `attn.arange`, `attn.scalar_tensor` | 暂缓 | 部分是缓存写入、mask 构造或 shape bookkeeping，实测延迟可能很低，但贸然融合会掩盖数据依赖 |
| MoE routing/quant 碎片，例如 `clamp`, `where`, `amax`, `abs`, `div` | 有条件 | 可先与手撕 MoE expert FP4/FP8 quant 模型对齐，再决定是否形成 `moe_quant_group` 之类规则 |
| backward grad fragments，例如 generic `linear` warning 中缺少 `weight` role 的片段 | 暂缓 | 训练态不能只为减少节点数而破坏 `dw/dx` 区分，建议先补角色识别或添加反向专用规则 |

当前最大融合 gap 是 `sparse_attention_kernel`：它已经把连续 `bmm -> mul -> softmax -> bmm -> bmm` 计算段融合，但 FLOPs/有效长度仍是统一 proxy，没有区分 HCA/CSA/SWA 的稀疏窗口和压缩策略。这个 gap 后续应与手撕 attention 公式和实测 kernel 时间对齐。

## 2. 并行策略对 shape、dtype、FLOPs、通信量的影响

### 总览

| 策略 | 对输入/输出 shape 的影响 | dtype 影响 | FLOPs 影响 | 通信量影响 |
|---|---|---|---|---|
| DP | 不改变单个计算算子的 activation shape；每 rank 处理 `global_batch / dp` 对应的 microbatch 数 | 不改变 dtype | 总集群 FLOPs 不变；单 rank 执行的 microbatch 数减少 | backward 插入梯度同步；ZeRO-0 all-reduce，ZeRO-1/2 reduce-scatter；ZeRO-3 当前 DP pass 跳过，交给 FSDP/ZeRO 路径 |
| TP | column linear 输出 last dim `/ tp`；row linear 输入 last dim `/ tp`，输出 shape 保持 | 不改变 dtype | 单 rank GEMM FLOPs 按切分降低；全局 FLOPs 基本不变 | row parallel 后插入 TP all-reduce；部分 CoC/MC2 可隐藏 |
| EP | 图上 activation shape 通常保持 `(B,S,H)`；local expert 数为 `num_experts / ep` | 不改变 dtype | 每 rank expert GEMM 按 local expert/token 分布变化 | MoE block 前后插入 dispatch/combine all-to-all，量约与 `B*S*topk*H*dtype_bytes` 成正比 |
| CP/SP | 将 seq_len 相关维度按 `cp` 切分 | 不改变 dtype | attention/序列相关计算按本地 seq 缩小 | Ulysses all-to-all、Ring P2P 或 Hybrid 组合通信 |
| PP | 不改变算子 tensor shape；只给 layer/node 分配 stage | 不改变 dtype | 单 stage 只执行被分配的 layer | stage 边界插入 send/recv；流水线 schedule 决定暴露与掩盖 |
| Quant | 不是并行策略；按 quant profile 改 dtype/bytes/FLOPs | 会改变权重/激活/通信 dtype | 取决于 FP8/FP4/BF16 规则 | 影响通信 bytes 和 kernel cost |

### DP 的校准公式和检查点

DP 的关键量是每个 rank 的 microbatch 数：

```text
M = global_batch / (micro_batch * dp)
```

梯度同步按 bucket 统计参数梯度 bytes：

```text
all_reduce_time ≈ 2 * (dp - 1) / dp * bucket_bytes / bandwidth + latency_terms
reduce_scatter_time ≈ (dp - 1) / dp * bucket_bytes / bandwidth + latency_terms
```

当前实现中的校准点：

- `dp=1` 时不插入 DP 通信。
- `zero_stage=0` 插入 `comm.all_reduce`。
- `zero_stage=1/2` 插入 `comm.reduce_scatter`。
- `zero_stage>=3` 当前 `DataParallelPass` 直接跳过，避免和 ZeRO/FSDP 路径重复建模。
- `dp_bucket_mode="layer"` 会把每层 grad reduce 串进 backward 依赖链，更保守。
- `dp_bucket_mode="ddp"` 会按 backward-ready 顺序构造 bucket side branch，并在 optimizer 前插入 `ddp_wait_all_buckets`；最终暴露量来自最后一个 bucket 超出 backward compute 结束的 tail。

DP 不应该改写计算算子的 shape/dtype；如果发现 fused linear、RMSNorm、attention kernel 的 input/output shape 随 DP 直接变化，应优先检查 batch/microbatch 参数是否被误解释成 tensor shape 改写。

### wiki 版 shape 规则

| 算子/模块 | TP | EP | CP/SP | PP | DP |
|---|---|---|---|---|---|
| `q_proj/k_proj/v_proj/gate_proj/up_proj/w1/w3` | 输出 hidden/intermediate dim `/tp` | 无直接 shape 改写 | seq 维可 `/cp` | stage 分配 | 不改 |
| `o_proj/down_proj/w2` | 输入 hidden/intermediate dim `/tp`，输出保持全量，后接 all-reduce | 无直接 shape 改写 | seq 维可 `/cp` | stage 分配 | 不改 |
| attention score/context | head 或 hidden 已受 TP 上游影响 | 无直接 shape 改写 | seq 维切分，按 CP 类型插入通信 | stage 分配 | 不改 |
| MoE gate | 通常保持 `(B,S,E)` 或 topk 相关 shape | local expert 数 `/ep`，tokens 经 A2A 重分布 | seq 维可 `/cp` | stage 分配 | 不改 |
| MoE experts | TP/quant 可影响 expert GEMM 内维 | 每 rank 只持有 local experts | seq/tokens 可随 CP 本地化 | stage 分配 | 不改 |
| loss/optimizer | TP/DP/ZeRO 影响聚合和分片 | expert 参数按 EP group | 通常不切 seq | 通常最后 stage 或 optimizer stage | ZeRO 改内存/同步，不改 activation |

## 3. 反向 `dw/dx` 与 DualPipe/通信掩盖

当前系统已经有三层 `dw/dx` 支撑：

1. 训练 FLOPs pass 会从节点 annotation 中区分 `flops_dx` 和 `flops_dw`。
2. `TrainingPipelinePass` 会估算 `stage_bwd_dw`，并写入 `stage_timelines_bwd_dw` metadata。
3. `PPStitcher` 在 `schedule="zb"` 时会把反向任务拆成 `bwd_dx` 和 `bwd_dw`：`bwd_dx` 负责跨 stage 反传依赖，`bwd_dw` 是本地权重梯度计算，可浮动填 bubble。

ZeroBubble 这条 trace path 的语义是清楚的：

- `fwd[m] -> bwd_dx[m]` 保证 activation dependency。
- `bwd_dx[m] -> bwd_dw[m]` 保证本地梯度链。
- 跨 stage P2P 只发生在 `bwd_dx`，不发生在 `bwd_dw`。
- `bwd_dw` 可独立漂移，用来填充 pipeline bubble。

DualPipe/DualPipeV 的现状要分开看：

| 路径 | 当前状态 | 校准结论 |
|---|---|---|
| spec/composer path | `StageTime` 有 `bwd_dx`/`bwd_dw` 字段；DualPipe/DualPipeV 公式用 `W=bwd_dw` 降低 cooldown/bubble；`pp_overlap` 可把部分 PP P2P 视为被 W stream 掩盖 | 可以用于手撕模型对齐 |
| trace/e2e path | `stage_bwd_dw` 有估算，但 DualPipe stitcher 仍调度整体 `bwd`，不像 ZB 那样显式生成 `bwd_dx/bwd_dw` task | PP/DualPipe 侧依赖项；DP 校准文档只记录风险，不在本 part 修复 |
| CLI/e2e 报表 | 已尝试 `--pp-schedule dualpipe`，但输出仍是 `schedule_name=1f1b` | PP/DualPipe/report 侧依赖项；当前端到端 DualPipe 校准不可视为通过 |

因此，用户关心的“通过 DualPipe 掩盖相关 `dw`”在公式/手撕路径是有表达的，但抓图端到端路径还不能宣称完全正确。建议下一步先修复 schedule 透传，再把 DualPipe trace task 也拆成 `bwd_dx`/`bwd_dw`，这样通信暴露和 W stream 掩盖才能被 Chrome trace 和报表直接验证。

## 4. 最终重计算与通信计算口径

当前报表口径可以按下式检查：

```text
total_comm_volume_ms ≈ exposed_comm_ms + hidden_comm_ms
dp_total_ms          ≈ dp_exposed_ms + dp_hidden_ms
compute_time_ms      = fwd_compute_ms + bwd_compute_ms + recompute_compute_ms
```

已有 e2e 输出如下：

| 输出 | batch | 请求 schedule | 报表 schedule | step ms | total comm ms | exposed comm ms | hidden comm ms | DP total/exposed/hidden ms | recompute compute ms | bwd compute ms |
|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|
| `dp_compare_ddp_quant_fp4_after_hc_rules` | 1 | 1f1b | 1f1b | 1560.072 | 12.503 | 9.181 | 3.322 | 0.312 / 0.105 / 0.207 | 74.523 | 1041.287 |
| `hc_calib_bs2_1f1b` | 2 | 1f1b | 1f1b | 2579.404 | 12.882 | 8.654 | 4.228 | 0.621 / 0.050 / 0.571 | 147.817 | 1932.657 |
| `hc_calib_bs1_dualpipe` | 1 | dualpipe | 1f1b | 1560.072 | 12.503 | 9.181 | 3.322 | 0.312 / 0.105 / 0.207 | 74.523 | 1041.287 |

检查结论：

- batch 从 1 到 2 后，重计算和 backward compute 近似按 batch 放大，符合预期。
- DP 通信总量从 0.312 ms 到 0.621 ms 近似翻倍，但暴露尾巴下降，说明当前 DDP bucket timeline tail 的掩盖逻辑在发挥作用。
- `total_comm_volume_ms = exposed + hidden` 在已有输出中闭合。
- `dualpipe` 输出与 1f1b 完全相同，且 `schedule_name=1f1b`，这是明确的 PP/DualPipe 依赖 gap；DP part 不直接修复，只在校准结论中标注风险。
- `fwd_compute_ms=0` 出现在这批训练 trace 报表中，不能解释为前向没有计算；它更像 phase attribution/reporting gap，后续应把 train_forward 的 compute 正确归入 step report。

## 5. 验证矩阵与特殊公式

建议后续校准按“先手撕，再抓图，再实测”三步走：

| 维度 | 建议组合 | 目的 |
|---|---|---|
| batch | `micro_batch=1/2/4`，保证 `global_batch % (micro_batch * dp) == 0` | 验证 FLOPs、重计算、DP bucket bytes 是否线性或按预期变化 |
| DP | `dp=1/2/4/8` | `dp=1` 应无 DP comm；`dp>1` 检查 all-reduce/reduce-scatter 与 exposed tail |
| ZeRO | `zero_stage=0/1/2/3` | 校准 all-reduce、reduce-scatter、ZeRO/FSDP 路径边界 |
| DP bucket | `layer` vs `ddp`, `dp_bucket_cap_mb=1/25/128` | 对比保守串行和 DDP reducer overlap |
| PP | `pp=1/2/4`, schedule=`1f1b/zb/dualpipe/dualpipev` | 验证 stage 分配、P2P、bubble 和 `dw` 掩盖 |
| TP/EP/CP | `tp=1/2/8`, `ep=1/8/64`, `cp=1/2` | 验证 shape split、A2A、CP ring/Ulysses 通信 |
| MoE wave | `ep_num_waves=1/2/4/8` | 校准多 wave EP 对 A2A 的隐藏收益 |

常用手撕公式：

```text
1F1B homogeneous bubble fraction ≈ (PP - 1) / (M + PP - 1)
VPP/interleaved bubble fraction ≈ (PP - 1) / (M * V + PP - 1)
```

EP 多 wave overlap 的当前实现等价于：

```text
comm_per_wave    = comm_time / K
gemm_per_wave    = gemm_time / K
exposed_per_wave = max(comm_per_wave - gemm_per_wave, 0)
exposed_total    = comm_per_wave + (K - 1) * exposed_per_wave
saved            = max(0, comm_time - exposed_total)
```

当 GEMM 足够覆盖通信时，最多只能隐藏 `(K-1)/K` 的通信；例如 `K=4` 最多隐藏 75%，`K=8` 最多隐藏 87.5%。这条公式适合与 DeepEP dispatch/combine 的 wave overlap 做一阶对齐。

## 6. 跳 layer 与重复 layer 报表处理

### 跳 layer

当前 PP pass 已经有非 layer 节点分配逻辑：

- embedding 类节点放 stage 0。
- final norm / lm head 放最后 stage。
- 其他 forward 非 layer 节点按 stage 负载贪心分配。
- backward 非 layer 节点按各 stage forward 负载比例分散，避免全部落到 stage 0。

训练分析侧还有 fallback：如果 traced layer 覆盖不到所有 PP stage，或出现严重 stage imbalance，会用 layer profile/typical layer 信息做均匀或按层类型的分摊。这能缓解“抓了典型层，但报表 stage 跳层/空层”的问题。

已有测试覆盖：

- `tests/training/test_pipeline_parallel.py::TestNonLayerNodeDistribution::test_stage0_no_longer_absorbs_all_nonlayer_fwd`
- `tests/training/test_pipeline_parallel.py::TestNonLayerNodeDistribution::test_backward_nodes_spread_across_mutiple_stages`
- `tests/training/test_captured_graph_modelling.py` 中的 captured graph stage 分配和 P2P 依赖测试

### 重复 layer / 黄区报表

Excel 侧已有 `_build_layer_display_map(graph)`，会把 typical layer 映射成代表范围，例如把某个典型层显示成“该层代表哪些真实层、步长、层数”。`_write_transformed_ops_sheet` 和 `_write_fwd_bwd_ops_sheet` 已经使用这个 map，避免报表黄区重复 layer 时失去真实覆盖范围。

当前 gap 是 HTML 报表还没有镜像 Excel 的 represented layer range 标签。建议把 `_build_layer_display_map()` 的结果提升为报告通用 metadata，让 Excel/HTML/JSON 都使用同一套 layer display 字段。

## 7. 本次校验结果

本次只按 DP 校准边界做校验和修复，不修改 PP/DualPipe runtime 代码。

| 校验项 | 方法 | 结果 |
|---|---|---|
| 融合规则库存 | 调用 fusion registry loader：`initialize_rules("deepseek_v4")` 后统计 active rules | 通过：注册 23 条，训练 active 21 条，`kv_compressor` / `sparse_indexer` 被训练配置禁用 |
| 融合后算子数量 | 解析 `output/dp_compare_ddp_quant_fp4_after_hc_rules/deepseek_v4_train_forward.json` 和 `output/hc_calib_bs2_1f1b/deepseek_v4_train_forward.json` | 通过：bs=1 为 1808 nodes / 181 fused nodes，bs=2 为 1830 nodes / 182 fused nodes；主要 fused op 与文档表一致 |
| DP 通信节点 | 解析 transformed JSON 中 `annotations.dp_comm` 节点，并补 graph-native 回归测试 | 通过：PP4-DP4-ZeRO1 输出均为 `comm.reduce_scatter`；当前 pipeline 下 `dp=1` 的 `dp_total/exposed/hidden` 全为 0 |
| 通信闭合 | 扫描现有 `*training_report.json`，检查 `total_comm_volume_ms = exposed_comm_ms + hidden_comm_ms`、`dp_total_ms = dp_exposed_ms + dp_hidden_ms` | 通过：DP4/DP8 现有报告闭合；bs=1 到 bs=2 的 DP total 约从 0.312 ms 到 0.621 ms，符合 batch 放大趋势 |
| shape/dtype 归因 | 新增只改变 `dp=1->4`、保持 TP/EP/PP/batch 不变的 graph-native 对照 | 通过：普通 compute op 的 `op_type`、input/output shape、input/output dtype 均不随 DP 改变 |
| ZeRO/FSDP | 窄测 `zero_stage=0/2` collective、DP skip、DDP tail、ZeRO metadata memory，并新增 ZeRO-3 graph-native 校验 | 通过：ZeRO-0 all-reduce、ZeRO-2 reduce-scatter、DP bucket exposed tail、ZeRO metadata memory 通过；ZeRO-3/FSDP 插入 all-gather/reduce-scatter，且 report 中 `dp_total/exposed/hidden` 与总通信闭合 |
| 重计算 | 窄测 recompute policy 对 activation memory 的影响，并检查报告字段 | 通过但有报表 gap：recompute memory/compute 字段存在且 batch 放大趋势合理；现有训练报表仍有 `fwd_compute_ms=0` 的 phase attribution gap |
| DualPipe 依赖 | 解析 `output/hc_calib_bs1_dualpipe` | 依赖未闭合：graph metadata 是 `dualpipe`，report 仍是 `1f1b`；trace task 只有 `fwd/bwd`，未显式拆 `bwd_dx/bwd_dw`，按外部依赖记录 |
| 跳 layer/重复 layer | 运行 Excel layer display map 窄测 | 通过：Excel fwd/bwd ops sheet 会应用 represented layer range；HTML/JSON 仍未共享该展示字段 |

此前 DP 校验已覆盖融合配置、layer display、recompute memory、DP bucket tail 等 13 条窄测。本轮针对 DP-owned gap 补充修复后，重点重跑以下 11 条：

```text
tests/training/test_captured_graph_modelling.py::test_modeller_dp1_reports_zero_dp_comm_fields
tests/training/test_captured_graph_modelling.py::test_zero3_fsdp_comm_is_reported_as_dp_comm_breakdown
tests/training/test_captured_graph_modelling.py::test_dp_degree_does_not_change_compute_op_shape_or_dtype
tests/training/test_data_parallel.py::TestDPZero0::test_all_reduce_created_per_layer_by_default
tests/training/test_data_parallel.py::TestDPZero2::test_reduce_scatter_for_zero2
tests/training/test_data_parallel.py::TestDPOverlap::test_dp_skip_when_dp1
tests/training/test_data_parallel.py::TestDPGroupIdx::test_ddp_bucket_mode_step_result_uses_timeline_tail
tests/training/test_data_parallel.py::TestDPGroupIdx::test_ddp_bucket_tail_exposes_only_comm_after_backward_end
tests/training/test_data_parallel.py::TestDPZero2::test_reduce_scatter_has_lower_modeled_time_than_all_reduce
tests/training/test_phase1_bugfixes.py::test_activation_memory_reads_zero_metadata
tests/training/test_report_field_parity.py::TestPerStrategyCommFields::test_dp_total_nonzero
```

本轮补充修复后的结果：11 targeted passed；随后完整运行 `tests/training/test_data_parallel.py`，36 passed。

注意：`tests/training/test_data_parallel.py::TestDPZero2::test_reduce_scatter_for_zero3` 当前测试名容易误导，它没有断言 `dp_comm` 节点数量；实际 `DataParallelPass` 对 `zero_stage>=3` 会跳过，交给 `ZeroFSDPPass`。因此 ZeRO-3 仍需要单独 e2e/graph 校验，而不能把这个单测通过理解为 ZeRO-3 已闭环。
本轮已新增 `test_zero3_fsdp_comm_is_reported_as_dp_comm_breakdown` 作为 ZeRO-3/FSDP 的 graph-native 闭环校验。

## 8. 当前 gap 与后续动作

| 优先级 | gap | 建议动作 |
|---|---|---|
| 外部依赖 | `--pp-schedule dualpipe` e2e 报表仍显示 `schedule_name=1f1b` | PP/DualPipe/report 侧修复项；DP part 不直接修改 |
| 外部依赖 | DualPipe trace path 未显式拆 `bwd_dx/bwd_dw` task | PP/DualPipe trace 侧修复项；DP part 只记录它会影响通信掩盖解释 |
| 外部依赖 | graph-path `test_zero_bubble_uses_dw_split_to_reduce_dualpipe_bubble` 当前会暴露 `pipeline_metrics` 缺失 | PP schedule regression；DP part 不直接修改 |
| P1 | `fwd_compute_ms=0` 报表归因异常 | 检查 train_forward/train_backward unified metadata 的 phase 聚合 |
| P1 | `sparse_attention_kernel` 有效长度 proxy 未区分 HCA/CSA/SWA | 对照手撕 sparse attention 公式，为不同 layer kind 写 FLOPs/bytes 分支 |
| P1 | backward generic `linear` warning | 补 grad-role 识别或反向专用融合规则，避免把 `dw/dx` 混掉 |
| P2 | HTML 缺 represented layer range | 复用 Excel `_build_layer_display_map()` 到 HTML/JSON |
| P2 | MoE routing/quant leftover 是否融合 | 先按 FP4/FP8 expert hand model 做数量级对齐，再决定规则 |

## 参考资料

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)：DualPipe 与 DeepSeek-V3 训练系统背景。
- [DeepEP 官方仓库](https://github.com/deepseek-ai/DeepEP)：MoE expert parallel dispatch/combine 通信库与 overlap 背景。
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)：1F1B/interleaved pipeline parallelism 背景。
- [PyTorch DistributedDataParallel 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)：DDP gradient bucket 与 reducer 行为背景。
