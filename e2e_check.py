from python.zrt.graph import run_trace_phases, load_model
from python.zrt.transform import (
    build_default_pipeline, TransformContext,
    ParallelConfig, StreamConfig,
)
from python.zrt.executor import DAGScheduler
from python.zrt.simulator import SimulatorHub
from python.zrt.report import build_summary
from python.zrt.memory import MemoryModel
import python.zrt.hardware.registry as hw_registry

model_id = "Qwen/Qwen2.5-7B-Instruct"

# Step 1: 抓图（Qwen2.5-7B 无需授权，最快）
result = run_trace_phases(
    model_id=model_id,
    num_layers=4,
    batch_size=1,
    seq_len=128,
    phases=("prefill",),
)
raw_graph, fused_capture_graph = result.graphs["prefill"]

# 获取配置对象用于内存估算（仅读取 config，不加载权重）
_, config, _ = load_model(model_id, num_hidden_layers=4)
print(f"\n[1] 抓图完成: {raw_graph}")
print(f"    fused (capture): {fused_capture_graph}")

# Step 2: 单卡 baseline（tp=1）
hw = hw_registry.load("nvidia_h100_sxm")
ctx1 = TransformContext(
    hw_spec=hw,
    parallel=ParallelConfig(tp=1),
    stream_config=StreamConfig(num_compute_streams=1, num_comm_streams=1),
)
pipe = build_default_pipeline()
g1 = pipe.run(raw_graph, ctx1)
tl1 = DAGScheduler(hw_spec=hw).schedule(g1)

print(f"\n[2] TP=1 baseline:")
print(f"    nodes={g1.num_nodes()}, comm_nodes={len(g1.comm_nodes())}")
print(f"    total_latency = {tl1.total_latency_us:.2f} µs ({tl1.total_latency_ms:.3f} ms)")
print(f"    compute_time  = {tl1.compute_time_us:.2f} µs")
print(f"    comm_time     = {tl1.comm_time_us:.2f} µs")
print(f"    overlap       = {tl1.overlap_us:.2f} µs")

# Step 3: TP=4（两流：1 compute + 1 comm）
ctx4 = TransformContext(
    hw_spec=hw,
    parallel=ParallelConfig(tp=4),
    stream_config=StreamConfig(num_compute_streams=1, num_comm_streams=1),
)
g4 = pipe.run(raw_graph, ctx4)
tl4 = DAGScheduler(hw_spec=hw).schedule(g4)

print(f"\n[3] TP=4:")
print(f"    nodes={g4.num_nodes()}, comm_nodes={len(g4.comm_nodes())}")
print(f"    total_latency = {tl4.total_latency_us:.2f} µs ({tl4.total_latency_ms:.3f} ms)")
print(f"    compute_time  = {tl4.compute_time_us:.2f} µs")
print(f"    comm_time     = {tl4.comm_time_us:.2f} µs")
print(f"    overlap       = {tl4.overlap_us:.2f} µs  ← 通算掩盖收益")

# Step 4: 查看前10个调度事件
print(f"\n[4] 前10个调度事件（按执行顺序）:")
for op in tl4.scheduled_ops[:10]:
    print(f"    [{op.stream_type:7s} stream {op.stream_id}] "
          f"{op.op_type:40s}  {op.start_us:8.2f} → {op.end_us:8.2f} µs")

# Step 5: 节点注解验证（FlopsPass + RooflinePass + StreamAssignPass 产出）
print(f"\n[5] 节点注解验证（g1 前5个节点，pipeline 应注入 latency_us / flops / bound / stream_id）:")
sample_nodes = list(g1.topo_sort())[:5]
missing_annot = []
for node in sample_nodes:
    lat   = node.annotations.get("latency_us", None)
    flops = node.annotations.get("flops", "N/A")
    bound = node.annotations.get("bound", "N/A")
    sid   = node.annotations.get("stream_id", "N/A")
    print(f"    {node.op_type:35s}  lat={str(lat):>10}µs  flops={str(flops):>12}  bound={bound}  stream={sid}")
    if lat is None:
        missing_annot.append(node.op_type)
assert not missing_annot, f"以下节点缺少 latency_us 注解: {missing_annot}"
print(f"    ✓ 所有采样节点均含 latency_us / flops / bound / stream_id 注解")

# Step 6a: 计算 memory_budget（基于 config + MemoryModel）
print(f"\n[6a] 内存预算计算:")
mem_model = MemoryModel()
memory_budget_1 = mem_model.estimate(
    profile=config,  # 使用 config 对象包含所需的模型参数
    hw_spec=hw,
    parallel=ParallelConfig(tp=1),
    batch_size=1,
    seq_len=128,
)
print(f"    weights_mb={memory_budget_1.weights_mb:.2f}")
print(f"    kv_cache_mb={memory_budget_1.kv_cache_mb:.2f}")
print(f"    activation_peak_mb={memory_budget_1.activation_peak_mb:.2f}")
print(f"    comm_buffer_mb={memory_budget_1.comm_buffer_mb:.2f}")
print(f"    framework_overhead_mb={memory_budget_1.framework_overhead_mb:.2f}")
print(f"    total_mb={memory_budget_1.total_mb:.2f} / {memory_budget_1.capacity_mb:.2f}")
print(f"    is_feasible={memory_budget_1.is_feasible}")

# Step 6b: E2ESummary（SimulatorHub + build_summary，包含 memory_budget）
print(f"\n[6b] E2ESummary（TP=1 prefill Qwen2.5-7B on H100）:")
hub = SimulatorHub.default()
sim_results_1 = hub.simulate_graph(g1, hw)
summary = build_summary(
    model="Qwen2.5-7B-Instruct",
    hardware="nvidia_h100_sxm",
    phase="prefill",
    batch_size=1,
    seq_len=128,
    graph=g1,
    sim_results=sim_results_1,
    timeline=tl1,
    hw_spec=hw,
    parallel_desc="TP1",
    memory_budget=memory_budget_1,  # ← 传入 memory_budget
)
print(summary)

# Step 7: 正确性断言
print(f"\n[7] 正确性断言:")
assert tl1.comm_time_us == 0.0,       f"TP=1 不应有通信时间，实际 {tl1.comm_time_us}"
assert len(g1.comm_nodes()) == 0,     f"TP=1 不应有 comm 节点，实际 {len(g1.comm_nodes())}"
assert len(g4.comm_nodes()) > 0,      f"TP=4 应有 comm 节点，实际 0"
assert tl1.overlap_us >= 0.0,         f"TP=1 overlap 不应为负"
assert tl4.overlap_us >= 0.0,         f"TP=4 overlap 不应为负"
assert summary.latency_ms > 0,        f"summary latency 应为正"
assert summary.mfu >= 0.0,            f"MFU 应为非负"
assert summary.ttft_ms is not None,   "prefill 阶段应有 TTFT"
assert summary.tpot_ms is None,       "prefill 阶段不应有 TPOT"
assert summary.memory_budget is not None,  "summary 应含 memory_budget"
assert summary.memory_budget.is_feasible, f"TP=1 内存应可行，总计 {summary.memory_budget.total_mb:.2f}MB"
assert raw_graph.num_nodes() > fused_capture_graph.num_nodes(), \
    f"融合后节点数 {fused_capture_graph.num_nodes()} 应小于原始 {raw_graph.num_nodes()}"
assert g1.num_nodes() > 0,            "transform 后图不应为空"
print(f"    ✓ TP=1: comm_time=0µs, comm_nodes=0")
print(f"    ✓ TP=4: comm_nodes={len(g4.comm_nodes())} > 0")
print(f"    ✓ overlap ≥ 0  (TP=1: {tl1.overlap_us:.2f}µs, TP=4: {tl4.overlap_us:.2f}µs)")
print(f"    ✓ summary: latency={summary.latency_ms:.3f}ms, mfu={summary.mfu:.4f}, ttft={summary.ttft_ms:.3f}ms")
print(f"    ✓ memory_budget: total={summary.memory_budget.total_mb:.2f}MB, feasible={summary.memory_budget.is_feasible}")
print(f"    ✓ raw_graph.nodes={raw_graph.num_nodes()} > fused_capture.nodes={fused_capture_graph.num_nodes()}")
