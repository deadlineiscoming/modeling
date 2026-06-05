# HC 算子端到端校准说明

当前分析基于分支 `main_up_MHC_fix`，日期 `2026-06-05`。本文面向抓图路径的 HC/MHC 算子校准，覆盖融合规则、融合前后状态、手撕 IR 对齐、并行 shape/dtype/FLOPs/通信、反向 dx/dw、DualPipe 掩盖、重计算以及跳 layer/重复 layer 报表处理。

结论分两类：

- 已确认实现行为：来自当前源码和 focused 脚本统计。
- 待校准 gap：需要继续用真实抓图、手撕模型和硬件实测对齐。

## 0. 当前分支相对上一轮的关键差异

| 项 | 当前分支结论 |
| --- | --- |
| V4 注册规则数 | `_common.yaml` 7 条 + `deepseek_v4.yaml` 16 条，共 23 条 |
| V4 training active rules | 21 条，禁用 `kv_compressor`、`sparse_indexer`，但 `sparse_attention_kernel` 仍在 training 中启用 |
| V4 inference active rules | 21 条，包含 V4 16 条；common 侧不包含 training-only 的 `cross_entropy/dropout` |
| Global training fallback | `training_default.yaml` 会禁用 `kv_compressor/sparse_indexer/sparse_attention_kernel`，但只有没有模型专属 config 时才使用 |
| Tensor shape | 训练 IR 同时记录 `shape_logical` 和 `shape_local` |
| DualPipe dw 掩盖 | 只有 `strategy.pp_overlap=True` 时，`bwd_dw` 才作为 PP P2P/recompute 的 hide window；普通 `pp_hidden_ms` 还需结合 trace 判断 |
| 重计算掩盖 | DualPipe/DualPipeV 下，`bwd_dw` hide window 会先分配给 PP P2P，再给 recompute residual |

## 1. 融合规则与融合前后状态

### 1.1 规则数量

当前规则加载顺序是：

1. 总是加载 `python/zrt/transform/fusion/rules/_common.yaml`。
2. 如果 `model_id` 可解析为模型 slug，则加载对应 `rules/<slug>.yaml`，DeepSeek-V4 对应 `deepseek_v4.yaml`。
3. fusion config 优先级是 explicit path、`configs/<model_slug>_<phase>.yaml`、`configs/<model_slug>.yaml`、`configs/<phase>_default.yaml`。

| 场景 | 注册规则 | active rules | 说明 |
| --- | ---: | ---: | --- |
| V4 training | 23 | 21 | 使用 `configs/deepseek_v4.yaml`，禁用 `kv_compressor/sparse_indexer` |
| V4 inference | 23 | 21 | 使用 `configs/deepseek_v4.yaml`，V4 规则全启用；common 的 CE/dropout 不属于 inference |
| 无模型 training fallback | 7 | 7 | 只加载 common；fallback 中禁用 V4 专属规则是 no-op |
| 无模型 inference fallback | 7 | 5 | common 的 `cross_entropy/dropout` 不属于 inference |

V4 active HC 相关规则：

| 规则 | op_type | phase | pattern | 融合前算子数 | 当前判断 |
| --- | --- | --- | --- | ---: | --- |
| `hc_pre_attn_raw` | `hc_pre` | training/inference | raw ordered regex | 11 | 高优先级，直接吃掉 RMS + routing + Sinkhorn 相关小算子 |
| `hc_pre_attn` | `hc_pre` | training/inference | `rms_coef` + ordered regex | 8 | 第二轮兜底，消费 pass-1 形成的 `rms_coef` |
| `hc_post_attn` | `hc_post` | training/inference | ordered regex | 4 | 吃掉 post/comb residual mixer |
| `hc_head` | `hc_head` | training/inference | `target_class: HCHead` | class subtree | 依赖 `HCHead` wrapper 被正确挂到 capture module |
| `sparse_attention_kernel` | `sparse_attention_kernel` | training/inference | ordered regex | 5 | 当前 V4 training 可融合，不等于 fallback training default |

### 1.1.1 各规则实际含义

当前 DeepSeek-V4 抓图路径实际加载 `_common.yaml` 7 条通用规则和 `deepseek_v4.yaml` 16 条模型专属规则。training phase 使用 `configs/deepseek_v4.yaml`，只禁用 `kv_compressor` 与 `sparse_indexer`。

| 规则 | op_type | training active | 实际含义 |
| --- | --- | --- | --- |
| `rms_norm` | `rms_norm` | 是 | 通用 RMSNorm 6-op 分解：`pow/square -> mean -> add -> rsqrt -> mul -> mul`。 |
| `rms_norm_nn` | `rms_norm` | 是 | `torch.nn.RMSNorm` 内置分解，语义同 RMSNorm，覆盖不同 module/class 命名。 |
| `rms_norm_inline` | `rms_norm_inline` | 是 | 无 weight mul 的内联 RMS 片段，用于 attention/HC 子图里只需要 normalize 系数的情况。 |
| `rms_coef` | `rms_coef` | 是 | 只融合 `pow/square -> mean -> add -> rsqrt`，作为第二轮 `hc_pre_attn` 的前置片段。 |
| `cross_entropy` | `cross_entropy` | 是 | 训练 loss 的 `log_softmax + nll_loss` 融合。当前 V4 causal LM 抓图主链路通常不是 HC 关注点。 |
| `dropout` | `dropout` | 是 | 训练 dropout 的 mask 生成与乘法融合。V4 当前配置下不是主要瓶颈。 |
| `add_norm` | `add_norm` | 是 | 虚拟 Add+Norm composer，用于 residual add 后接 Norm 的组合，不是 V4 HC 的主规则。 |
| `dsv4_rms_norm` | `rms_norm` | 是 | DeepSeek-V4 自定义 RMSNorm；优先覆盖 V4 `RMSNorm` class。 |
| `parallel_embedding` | `parallel_embedding` | 是 | embedding gather；world size > 1 时语义包含可选 all-reduce。 |
| `linear` | `linear` | 是 | 通用 `Linear`/fp4/fp8/bf16 GEMM，使用 DAG signature 容忍量化 kernel 的 op 顺序差异。当前 backward 图上会偶发匹配到线性梯度片段，存在 `weight` role warning，见 gap 表。 |
| `column_parallel_linear` | `column_parallel_linear` | 是 | 列并行线性层，输出 hidden_out 分片；TP>1 时影响输出局部 shape 和后续通信。 |
| `row_parallel_linear` | `row_parallel_linear` | 是 | 行并行线性层，输入 hidden 分片，输出通常需要 all-reduce/规约。 |
| `rotary_emb` | `rotary_emb` | 是 | RoPE 复数乘相关子图，覆盖 `view_as_complex/mul/view_as_real` 等。 |
| `kv_compressor` | `kv_compressor` | 否 | V4 KV compressor class-only 融合；training 禁用，因为模块有 KV state/buffer 写入，保留 raw autograd 链更安全。 |
| `sparse_indexer` | `sparse_indexer` | 否 | V4 sparse indexer class-only 融合；training 禁用，原因同 compressor，避免破坏 backward。 |
| `mla_sparse_attn` | `mla_sparse_attn` | 是 | Attention class-level 语义规则，用于标识 V4 MLA/sparse attention 大块；是否命中取决于完整 module bucket。 |
| `moe_gate` | `moe_gate` | 是 | MoE router：linear + sigmoid/softmax + topk/gather。 |
| `moe_expert_swiglu` | `moe_expert_swiglu` | 是 | 单 Expert 的 SwiGLU FFN：3 个 GEMM + silu + mul。EP/GroupedMM 后要注意 local expert 计数。 |
| `hc_pre_attn_raw` | `hc_pre` | 是 | HC pre 原始 11-op 形态：RMS 系数、routing linear、Sinkhorn split、weighted sum 一次融合。修复后以 capture 中 flattened `[B,S,HC*H]` 输入、`post/comb` 输出推导 HC 维。 |
| `hc_pre_attn` | `hc_pre` | 是 | 第二轮兜底：当 pass-1 已先形成 `rms_coef` 时，继续融合后续 routing/Sinkhorn/weighted sum。 |
| `hc_post_attn` | `hc_post` | 是 | HC post residual mixer：`post*x + comb*residual`，输出 `[B,S,HC,H]`。 |
| `hc_head` | `hc_head` | 是 | LM head 前的 HC mix-down：flatten 后做 RMS + linear/sigmoid + weighted sum，输出 `[B,S,H]`。 |
| `sparse_attention_kernel` | `sparse_attention_kernel` | 是 | `sparse_attn` kernel 的 `bmm/mul/softmax/bmm/bmm` 连续片段；当前仍使用统一 proxy，未区分 HCA/CSA/SWA effective length。 |

### 1.2 融合前后对比口径

每次真实抓图校准建议保存两类计数：

| 指标 | raw graph | fused graph | 检查 |
| --- | --- | --- | --- |
| HC pre | `pow/square -> mean -> add -> rsqrt -> mm -> mul -> softmax -> sigmoid -> softmax -> mul -> sum` | 1 个 `hc_pre` | 同一 scope 内不应残留 `rms_coef`、softmax/sigmoid、sum |
| HC post | `mul -> mul -> sum -> add` | 1 个 `hc_post` | 同一 scope 内不应残留 post/comb mixer 小算子 |
| HC head | `HCHead` subtree | 1 个 `hc_head` | 查 `fused_by_rule` 和 `source_op_ids` |
| sparse attention | `bmm/matmul -> mul -> softmax -> bmm/matmul -> bmm/matmul` | 1 个 `sparse_attention_kernel` | 与 HCA/CSA/SWA 的 effective length 分支对齐 |

融合后节点需要重点查：

- `annotations["fused_by_rule"]`
- `annotations["source_op_ids"]`
- `annotations["fused_from"]`
- `annotations["num_sub_ops"]`

当前 fuser 命中后会删除被替换的 source nodes。若仍有小算子残留，常见原因是：

1. 对应 rule 未 active，例如非 V4 fallback training 中 `sparse_attention_kernel` 会被禁用。
2. sliding-window 只命中连续片段，前后未匹配小算子保留。
3. module bucket 被 class/scope 拆散，导致 ordered regex 跨 bucket 失败。
4. 同一个 fused node 的 `io_roles` 过于简化，命中了 pattern 但 shape/FLOPs 推导仍不准。

### 1.3 修复后融合现状

本轮修复后，HC forward 侧已经从“只看 fused count”推进到“多输出语义和 FLOPs 公式可对齐”：

| 项 | 修复后 capture 融合规则 | 手撕 IR 对齐 |
| --- | --- | --- |
| HC pre 输入 | capture 中实际是 flatten 后的 `[B,S,HC*H]` | 手撕 IR 逻辑输入仍可表示为 `[S,HC,H]`，二者通过 flatten 对齐 |
| HC pre 输出 | fused node 语义保留 `output/post/comb` | 对齐 `x_pre: (S,H)`、`hc_post: (S,HC)`、`hc_comb: (S,HC,HC)` |
| HC pre FLOPs | `linear + Sinkhorn + weighted HC reduction`，其中 `mix_hc=(2+HC)*HC` | 与 `hc_split_sinkhorn` kernel 公式一致，实测系数仍待拟合 |
| HC post 输出 | 从 `[B,S,HC,H]` 输出推导 `HC/H` | 对齐手撕 IR 的 HC residual stream |
| HC head | 从 `activation.shape[-1] // output.shape[-1]` 推 `HC` | 对齐尾部 HC mix-down |
| sparse kernel | 仍使用统一 `topk_kv_proxy = S//4` | 仍需按 HCA/CSA/SWA effective length 拆分或补 annotation |

真实 e2e 抓图验证：`train_forward` 捕获 1240 个 raw ops，`train_backward` 捕获 1309 个 raw ops，stitched graph 共 2549 个节点。融合后 forward 侧 HC scope 中：

| scope | fused 数量 | forward 小算子残留 |
| --- | ---: | ---: |
| `hc_pre` | 8 | 0 |
| `hc_post` | 8 | 0 |
| `hc_head` | 1 | 0 |

backward 侧仍保留 autograd 展开的 `pow/mul/sum/sigmoid_backward` 等小算子；当前没有对 backward HC 子图再做独立融合。因此反向校准应以 `flops_dx/flops_dw`、recompute、通信掩盖口径为主，不应把 backward 残留小算子误判为 forward fusion 失败。

## 2. 手撕 IR 基线

当前 `deepseek_v4_pro` spec 统计：

| 项 | 值 |
| --- | ---: |
| 层数 | 62 |
| hidden | 7168 |
| seq_len | 4096 |
| `hc_mult` | 4 |
| `hc_sinkhorn_iters` | 20 |
| HCA 层 | 31 |
| CSA 层 | 30 |
| SWA/MTP 层 | 1 |
| 总 ops | 1854 |

关键算子数量：

| kind | count |
| --- | ---: |
| `hc_expand` | 1 |
| `mhc_pre` | 124 |
| `mhc_post` | 124 |
| `mhc_head` | 1 |
| `hca_attn` | 31 |
| `sparse_attn` | 30 |
| `swa_attn` | 1 |
| `compressor_pool` | 91 |
| `indexer_topk` | 30 |

对齐关系：

| 抓图 fused op | 手撕 IR op | 校准说明 |
| --- | --- | --- |
| `hc_pre` | `mhc_pre` | count 要按每层 attention/FFN 两个 pre 对齐；FLOPs 要补 Sinkhorn 与多输出语义 |
| `hc_post` | `mhc_post` | count 要按每层 attention/FFN 两个 post 对齐 |
| `hc_head` | `mhc_head` | 全局尾部 mix-down |
| 无直接 capture 对应 | `hc_expand` | 手撕 IR 显式建模全局 HC 扩展 |
| `sparse_attention_kernel` | `hca_attn/sparse_attn/swa_attn` | 需要按 layer compression ratio 拆分 |

## 3. 并行策略对 shape、dtype、FLOPs、通信量的影响

### 3.1 base shape

未切分时代表 shape：

| op | logical/local input | logical/local output | dtype |
| --- | --- | --- | --- |
| `hc_expand` | `(4096,7168)` | `(4096,4,7168)` | BF16 |
| `mhc_pre` | `(4096,4,7168)` | `(4096,7168)`, `(4096,4)`, `(4096,4,4)` | BF16 output + FP32 post/comb |
| `mhc_post` | `(4096,7168)`, `(4096,4,7168)`, `(4096,4)` | `(4096,4,7168)` | BF16 main path |
| `hca_attn` | q `(4096,65536)`, k/v `(4096,512)` | `(4096,65536)` | BF16 |
| `sparse_attn` | q `(4096,65536)`, k/v `(4096,512)` | `(4096,65536)` | BF16 |
| `indexer_topk` | q `(4096,8192)`, kv `(1024,128)` | `(4096,1024)` | BF16 in current Tensor metadata |
| `mhc_head` | `(4096,4,7168)` | `(4096,7168)` | BF16 |

真实抓图中，HC 源码入口和 aten capture 入口存在一个形态差异：`hc_pre/hc_head` 在源码里先接收 `[B,S,HC,H]`，但进入 `F.linear` 前已经 `flatten(2)`，因此 fusion rule 实际看到的 activation 是 `[B,S,HC*H]`。当前规则按以下口径推导：

| fused op | capture sem_io | 推导 |
| --- | --- | --- |
| `hc_pre` | activation `[B,S,HC*H]`，output `[B,S,H]`，post `[B,S,HC]`，comb `[B,S,HC,HC]` | `HC = post.shape[-1]`，`H = output.shape[-1]`，`mix_hc=(2+HC)*HC` |
| `hc_post` | output `[B,S,HC,H]` | `HC = output.shape[-2]`，`H = output.shape[-1]` |
| `hc_head` | activation `[B,S,HC*H]`，output `[B,S,H]` | `HC = activation.shape[-1] // output.shape[-1]` |

### 3.2 `tp=4, cp=2` shape

当前实现的关键 local shape：

| op | local shape | 说明 |
| --- | --- | --- |
| HC family | `S_local = 4096/(4*2) = 512` | `hc_expand/mhc_pre/mhc_post/mhc_head` 都按 TP*CP 切 token 维 |
| attention core | `S_local = 4096/2 = 2048`, `heads = 128/4 = 32` | q local `(2048,16384)` |
| compressor | input/output local `(2048,512)`，meta `d_local=128` | sequence 按 CP，compute 维按 TP |
| indexer | q local `(2048,2048)`, `ih_local=16` | output local `(2048,1024)` |
| PP/EP | 不改变上述 HC/attention local shape | 主要改变 stage 分配和 MoE A2A |

这个分支的测试 `tests/training/test_mhc_sharding.py` 明确要求 HC family 的 `meta["s"]` 和 tensor leading local dim 等于 `seq/(tp*cp)`。这是一条需要与 runtime 实际张量分布重点核对的规则。

### 3.3 策略影响速查

| 策略 | shape/FLOPs | 通信 |
| --- | --- | --- |
| TP | matmul col/row 切分；attention heads 切分；compressor `d_local`；indexer `ih_local`；HC 当前按 token special case 切到 `S/(TP*CP)` | QKV/O/FFN 插入 AG/RS，TP overlap 支持 MC2/CoC |
| CP | attention token dim按 CP；V4 默认可走 compressed CP | Ulysses A2A、Ring P2P、Hybrid，或 V4 compressed CP 的 stage1 P2P + stage2 AG |
| EP | routed experts 切分，不直接改变 HC shape | routed expert 前后 A2A，payload 不预除以 EP |
| PP | 不改变单 op shape | stage P2P、bubble、warmup/cooldown 和 schedule 选择 |
| DP | 不改变单卡 op shape | grad reduce/optimizer comm，可被 bubble 或 bucket overlap 掩盖 |

V4 compressed CP 公式口径：

```text
CSA stage1: m=4 boundary P2P, 含 KV / weights / indexer_k
HCA stage1: m'=128 boundary P2P, 含 KV / compress weights
stage2: compressed KV allgather
SWA-only: 跳过 compressed CP 通信
```

## 4. 反向 dx/dw 与 DualPipe 掩盖

当前 `OpCost` 区分 fwd、dx、dw：

| op | dx | dw | 说明 |
| --- | --- | --- | --- |
| `mhc_pre` | 有 | 有 | `dw_cube_flops = fwd_lin`；dx 约 `2.5*fwd` |
| `mhc_post` | 有 | 无 | 当前只计激活反传 |
| `mhc_head` | 有 | 有 | `dw_cube_flops = fwd_lin` |
| attention | 有 | 无 | attention kernel 无参数 dw |
| matmul/FFN/MoE | 有 | 有 | TP bwd overlap 统计 dx+dw |

`stage_time()` 会累计：

- `bwd_dx`
- `bwd_dw`
- `bwd = bwd_dx + bwd_dw + native exposed comm`
- `recompute`，单独保留，不直接塞进 `bwd_dx`

DualPipe/DualPipeV 公式：

```text
F = max_stage_fwd
B = max_stage_bwd
W = max_stage_bwd_dw
F&B = max(F, B)
DualPipe bubble = (PP/2 - 1) * (max(F&B - 2W, floor) + max(B - W, floor))
DualPipeV bubble = DualPipe bubble / V
```

PP P2P 与 recompute 的 **dw 专属掩盖** 有一个门控：

```text
is_dual = schedule in {dualpipe, dualpipev} and strategy.pp_overlap
```

只有 `pp_overlap=True` 时，`bwd_dw` 才作为 hide window：

1. 先隐藏 `2 * pp_p2p_per_direction`。
2. 剩余 hide budget 再隐藏 recompute。
3. 未隐藏的 PP P2P 进入 `pp_exposed_ms`。
4. 已隐藏的 PP P2P 进入 `pp_hidden_ms`。
5. recompute 的未隐藏部分进入 `recompute_critical_ms`，原始量保留在 `recompute_raw_mag_ms`。

这点很关键：DualPipe schedule 本身减少 bubble，但 PP P2P 是否被 `bwd_dw` 隐藏，取决于 `pp_overlap`，不是默认免费开启。注意：报告中的 `pp_hidden_ms` 也可能包含普通 timeline/bubble 中被隐藏的 PP 通信；判断 DualPipe-dw 掩盖时要同时看 `schedule_name`、`pp_overlap` 配置和 trace。

## 5. 重计算与最终结果检查

重计算类别中 HC 对应 `"hc"`，覆盖：

- `mhc_pre`
- `mhc_post`
- `mhc_head`

memory 侧也有 HC 特判：当 recompute 类别包含 `"hc"` 或 `"full"` 时，会释放 HC residual stream activation；否则只增加 recompute time 而不降内存会误判为必然更差。

最终结果建议检查这些字段：

| 字段 | 检查 |
| --- | --- |
| `compute_time_ms` | 应等于 `fwd_compute_ms + bwd_compute_ms + recompute_critical_ms` |
| `exposed_comm_ms` | 应等于 TP/CP/EP/PP/DP exposed 之和 |
| `hidden_comm_ms` | 应等于 DP/TP/EP/PP hidden 之和 |
| `total_comm_ms` | 应等于 exposed + hidden |
| `recompute_critical_ms` | 只表示关键路径残留 |
| `recompute_raw_mag_ms` | 表示未考虑隐藏前的原始重计算量 |
| `pp_hidden_ms` | 表示被 timeline/bubble 或 dw hide window 掩盖的 PP 通信；DualPipe-dw 掩盖需额外确认 schedule/pp_overlap |
| `mfu/hfu` | MFU 排除 recompute，HFU 包含 recompute |

核心不变量：

```text
step_time = pipeline_time + optimizer_time + optimizer_comm
pipeline_time = compute_time + exposed_comm + bubble
compute_time = fwd_compute + bwd_compute + recompute_critical
hidden_comm = dp_hidden + tp_hidden + ep_hidden + pp_hidden
total_comm = exposed_comm + hidden_comm
```

## 6. 验证矩阵

建议先跑 spec-based estimate，再接真实抓图 raw/fused dump。

| 目标 | batch / strategy | 核查 |
| --- | --- | --- |
| 单卡基线 | `tp=1, cp=1, pp=1, ep=1, dp=1`, `micro_batch=1/2` | HC/MHC 数量、dtype、FLOPs 与手撕公式 |
| TP/CP shape | `tp=4, cp=2` | HC `S/(TP*CP)`、attention `S/CP`、heads/TP |
| V4 compressed CP | `cp=2/4/8`, `cp_kind=compressed` | CSA/HCA stage1/stage2 bytes、SWA skip |
| EP 多 wave | `ep=8/64/384`, `ep_overlap=True`, `mega_moe_waves=K` | A2A payload、hidden/exposed 守恒 |
| 1F1B | `pp=4/8`, `pp_schedule=1f1b` | bubble、PP P2P exposed/hidden 守恒 |
| DualPipe | `pp=4/8`, `pp_schedule=dualpipe/dualpipev` | `bwd_dw` 降 bubble；`pp_overlap=True` 时查 `pp_hidden_ms` |
| 重计算 | `recompute={"moe":{"hc"}}` 或 `full` | `recompute_raw_mag_ms` 与 `recompute_critical_ms` 区分 |
| batch sweep | 固定策略下扫 `micro_batch=1/2/4` 和不同 `global_batch` | microbatches `M = global_batch/(micro_batch*dp)` 对 warmup/steady/cooldown 的影响 |

本轮抓图 e2e 验证使用同一条主命令参数族：`hf_models/deepseek_v4 --train --hw nvidia_h100_sxm --hidden 7168 --layers 4 --seq-len 128 --global-batch 32 --micro-batch 8 --dp 4 --pp 4 --tp 1 --recompute-policy full --optimizer adam --zero-stage 1 --dp-ddp-buckets --dp-bucket-cap-mb 25`。

| 变体 | `schedule_name` | step | FLOPs | HC fused 摘要 | 通信/重计算 |
| --- | --- | ---: | ---: | --- | --- |
| `batch-size=1, pp_schedule=1f1b` | `1f1b` | 1560.1 ms | 159.38T | `hc_pre=8/1.47G`，`hc_post=8/293.6M`，`hc_head=1/44.0M` | exposed 9.18 ms，hidden 3.32 ms，recompute critical 19.19 ms/raw 26.92 ms |
| `batch-size=2, pp_schedule=1f1b` | `1f1b` | 2579.4 ms | 318.71T | HC FLOPs 约 2 倍：`hc_pre=2.94G`，`hc_post=587.2M`，`hc_head=88.1M` | exposed/hidden 通信数量随 batch 增长，HC shape/FLOPs 线性通过 |
| `batch-size=1, pp_schedule=dualpipe` | `1f1b` | 1560.1 ms | 159.38T | 与 1F1B 相同 | CLI 传入 dualpipe，但 report 仍落为 `1f1b`，这是当前 pipeline gap |

PowerShell 对 stderr 中的 transformers deprecation warning 会把命令包装成 `NativeCommandError`，但报告、Excel、JSON、HTML 和 trace 均已导出。另有 generic `linear` 规则在 backward 线性梯度片段上出现 `weight` role warning，当前不影响 HC forward 规则，但应单独收敛。

参考公式：

```text
1F1B bubble 近似: (PP - 1) * (F + B)
DualPipe bubble: (PP/2 - 1) * (F&B + B - 3W)
DualPipeV bubble: DualPipe bubble / V
wave overlap saved 近似: min(comm, compute * (K - 1) / K)
```

参考链接：

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeek-AI DualPipe](https://github.com/deepseek-ai/DualPipe)
- [Megatron Core Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)

## 7. 跳 layer 与重复 layer 报表处理

当前分支有三层处理：

1. `LayerProfile` 对 V4 按 compression ratio 和 hash/topk 拆成 HCA_HASH、HCA_TOPK、CSA_HASH、CSA_TOPK、SWA_HASH、SWA_TOPK，`typical_indices` 选每类第一层。
2. `PipelineParallelPass` 只有在有完整 metadata cost 或显式 full assignment 时才用真实 `layer_id` 索引 full-model assignment；否则对代表层 trace 使用 compact trace partition，避免 `[0,2,3,4]` 被当成连续层导致跳 stage。
3. Excel 报表有 `_build_layer_display_map()`，会把同一类型代表层展示成覆盖范围，例如 V4 topk HCA 可显示为 `3 (3-59, step 2, 29层)`，避免黄区报表里重复 layer 看不清来源。

HTML report 侧目前仍保留每个 traced layer 独立 block，并按 `layer_scale` 设置 `repeat` 和 scaled `total_ms`；它不会跨 HCA/CSA/SWA 合并。这个策略对 V4 是安全的，因为不同 compression ratio 的层不应被折成一个平均 block。

后续可增强：

- HTML block label 增加 layer type 和 represented range。
- raw/fused op dump 增加 `layer_display_map` 字段。
- 对 HC pre/post fused node 输出多输出 shape 诊断，避免只显示单 output 造成误读。

## 8. 当前 gap 清单

| gap | 当前状态 | 后续动作 |
| --- | --- | --- |
| HC pre capture shape/FLOPs | 已修复：`hc_pre_attn_raw/hc_pre_attn` 暴露 `output/post/comb`，按真实 capture `[B,S,HC*H]` + `post/comb` 推 `HC/H` | 对真实 fused node 的 `source_op_ids`、输入输出 TensorMeta、公式系数继续实测校准 |
| HC pre Sinkhorn | 已修复：capture rule 计入 routing linear、Sinkhorn 迭代和 weighted HC reduction | 先与手撕 IR 对齐，再拟合实测系数 |
| HC post shape | 已修复：`hc_post_attn` 从 `[B,S,HC,H]` 输出推导 `hc_dim`，FLOPs 覆盖 current-state mix 与 residual mix | 检查真实抓图中 role 选择是否稳定 |
| sparse/HCA effective length | capture rule 使用统一 `S//4` proxy | 按 HCA/CSA/SWA 拆 rule 或补 annotation |
| training config 差异 | V4 config 与 fallback config 对 `sparse_attention_kernel` 行为不同 | 文档和 CLI 输出需显示实际使用的 fusion config 路径 |
| HC TP special case | 当前实现固定 `S/(TP*CP)` | 与目标 runtime 张量分布确认 |
| DualPipe P2P hide | 需要 `pp_overlap=True` | 验证配置、报表和 trace 都显式展示该开关 |
| CLI schedule 落地 | `--pp-schedule dualpipe` e2e 后 report 仍为 `schedule_name=1f1b` | 检查 trace-mode TrainingPipelinePass 到 report 的 schedule 参数传递 |
| backward generic linear warning | backward 图中 `linear` rule 偶发缺 `weight` role，出现 memory formula warning | 限制 linear rule phase/scope，或给 backward 线性片段补更稳的 fallback IO |
| 黄区重复 layer | Excel 有 range label，HTML 仍按 traced layer block 展示 | HTML 增加 type/range label |

## 9. 本轮修复记录

2026-06-05 在 `main_up_MHC_fix` 上完成以下低风险修复：

- `python/zrt/transform/fusion/rules/deepseek_v4.yaml`：修正 `hc_pre_attn_raw`、`hc_pre_attn`、`hc_post_attn`、`hc_head` 的 HC 维度推导和 FLOPs/bytes 公式；`hc_pre` 融合节点保留 `y/post/comb` 三个外部输出，并按真实 capture 中 flatten 后的 `[B,S,HC*H]` 形态推导。
- `tests/test_fusion_pass.py`：新增 `test_dsv4_hc_pre_raw_preserves_y_post_comb_outputs`，固定回归断言 `hc_pre` 三输出、`hidden_out` 和 Sinkhorn 后 FLOPs 量级。
- 已验证：`tests/test_fusion_pass.py`、`tests/transform/fusion/test_dsv4_rules.py`、`tests/training/test_mhc_sharding.py`、`tests/training/test_dualpipe.py`、`tests/IT/test_mhc.py`。
- 已补充 e2e：`batch-size=1/2 + 1F1B` 和 `batch-size=1 + dualpipe`。其中 batch 维线性通过；dualpipe 参数未反映到最终 `schedule_name`，已列为 gap。
