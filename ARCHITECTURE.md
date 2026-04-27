# ZRT-Sim Architecture V2 — LLM Performance Modeling & Simulation

---

## 1. 总体架构

### 1.1 设计理念

| 原则 | 含义 |
|------|------|
| **图驱动** | 所有分析围绕 OpGraph IR，它是贯穿全系统的唯一数据结构 |
| **硬件×软件栈正交** | 硬件规格（910B / H100）与软件栈（MindIE / vLLM）独立定义，笛卡尔组合 |
| **先切后融** | 先做并行切分 → 再在子图内融合，保证融合规则不跨切分边界 |
| **显存一等公民** | 显存模型独立于执行仿真，参与寻优剪枝和可行性判断 |
| **仿真器可插拔** | 统一接口 + 优先级回退链，从 Roofline 到 Tiling 级仿真自动适配 |
| **无卡运行** | 全流程基于 FakeTensor，不需要真实硬件和模型权重 |
| **模块独立可服务化** | 每个核心模块可独立使用、独立对外暴露 API |

### 1.2 系统分层

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Application Layer                               │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐           │
│  │ Orchestrator│ │Config      │ │HW Compare  │ │Bottleneck  │           │
│  │ (全流程编排)│ │Search(寻优)│ │(硬件对比)   │ │Analysis    │           │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘           │
├────────┴──────────────┴──────────────┴───────────────┴──────────────────┤
│                          Core Layer                                     │
│                                                                         │
│  ┌────────┐  ┌────────┐  ┌────────────┐  ┌──────────┐  ┌────────────┐  │
│  │ Model  │  │ Graph  │  │  Graph     │  │  Graph   │  │  Report    │  │
│  │Manager │─→│Capture │─→│ Transform  │─→│ Executor │─→│ Generator  │  │
│  └────────┘  └────────┘  └────────────┘  └──────────┘  └────────────┘  │
│                               │               │                         │
│                          ┌────┴────┐    ┌─────┴──────┐                  │
│                          │ Memory  │    │  Simulator  │                  │
│                          │ Model   │    │  Hub        │                  │
│                          └─────────┘    └────────────┘                  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                        Foundation Layer                                  │
│  ┌────────┐  ┌───────────┐  ┌───────────┐  ┌────────────┐              │
│  │  IR    │  │ Hardware  │  │ Software  │  │   Comm     │              │
│  │(OpGraph│  │ Registry  │  │ Stack     │  │   Model    │              │
│  │ types) │  │           │  │ Registry  │  │            │              │
│  └────────┘  └───────────┘  └───────────┘  └────────────┘              │
├─────────────────────────────────────────────────────────────────────────┤
│                        Extension Layer                                   │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                        │
│  │ Serving    │  │ Training   │  │ Calibration │                        │
│  │ Simulator  │  │ Estimator  │  │ (校准)      │                        │
│  └────────────┘  └────────────┘  └────────────┘                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 数据流

```
HF model_id / 本地路径
        │
        ▼
  ┌─────────────┐
  │ ModelManager │─── patches/compat 注入
  │ (FakeTensor) │─── 输出 ModelProfile (结构化参数)
  └──────┬──────┘
         │ model + fake_mode
         ▼
  ┌─────────────┐
  │ GraphCapture │─── TorchDispatchMode + ModuleTracker
  │              │─── prefill / decode 两阶段
  └──────┬──────┘
         │ Raw OpGraph (IR)
         ▼
  ┌─────────────────────────────────────────────────┐
  │              Graph Transform Pipeline             │
  │                                                   │
  │  Stage 1: Parallel Split                          │
  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌───────────────┐  │
  │  │ TP │→│ EP │→│ SP │→│ PP │→│ Comm Inserter │  │
  │  └────┘ └────┘ └────┘ └────┘ └───────────────┘  │
  │               ↓                                   │
  │  Stage 2: Fusion (在子图内)                        │
  │  ┌───────────────────────┐                        │
  │  │ SoftwareStack.rules() │                        │
  │  └───────────────────────┘                        │
  │               ↓                                   │
  │  Stage 3: Optim Passes                            │
  │  ┌──────┐ ┌────────┐ ┌─────┐ ┌──────┐            │
  │  │Quant │ │  EPLB  │ │ MTP │ │Recomp│            │
  │  └──────┘ └────────┘ └─────┘ └──────┘            │
  │               ↓                                   │
  │  Stage 4: Analysis                                │
  │  ┌──────┐ ┌────────────┐                          │
  │  │FLOPs │ │  Roofline  │                          │
  │  └──────┘ └────────────┘                          │
  └────────────────────┬────────────────────────────┘
                       │ Transformed OpGraph(s)
             ┌─────────┴──────────┐
             ▼                    ▼
      ┌─────────────┐     ┌────────────┐
      │MemoryModel  │     │SimulatorHub│─── Roofline / Regression / ProfileDB / Tiling
      │(可行性检查)  │     │(算子耗时)   │
      └──────┬──────┘     └─────┬──────┘
             │                  │ dict[node_id → SimResult]
             │                  ▼
             │           ┌─────────────┐
             │           │ DAGScheduler│─── 多流调度 + 通算掩盖
             │           │ + CommModel │
             │           └──────┬──────┘
             │                  │ Timeline
             ▼                  ▼
      ┌──────────────────────────────┐
      │        Report Generator       │
      │  Excel │ HTML │ ONNX │ JSON  │
      └──────────────────────────────┘
```

---

## 2. Foundation Layer

### 2.1 计算图 IR

IR 是全系统的数据总线。设计要点：
- 与 PyTorch/ONNX 类型解耦
- 原生支持层次化视图（算子级 → 模块级 → 层级 → 阶段级）
- 可序列化（JSON）
- 平台无关，平台信息通过 annotations 附加

```
zrt/
└── ir/
    ├── __init__.py
    ├── types.py           # DType, DeviceType, TensorMeta
    ├── node.py            # OpNode
    ├── edge.py            # Edge (data / control)
    ├── graph.py           # OpGraph + 图操作 API
    ├── hierarchy.py       # GraphHierarchy: scope 树 + 聚合
    ├── utils.py           # 拓扑排序, 依赖分析, 子图提取
    └── serde.py           # JSON 序列化/反序列化
```

#### 核心类型

```python
@dataclass(frozen=True)
class TensorMeta:
    """tensor 的元信息（不持有真实数据）"""
    id: str
    shape: tuple[int, ...]
    dtype: DType                 # fp16 / bf16 / fp32 / int8 / ...
    memory_bytes: int            # shape * dtype.itemsize

@dataclass
class OpNode:
    """单个算子节点"""
    id: str
    op_type: str                 # "aten.mm", "aten.layer_norm", "comm.all_reduce", ...
    inputs: list[TensorMeta]
    outputs: list[TensorMeta]
    attrs: dict[str, Any]        # 算子属性 (groups, eps, ...)
    scope: str                   # 模块路径 "model.layers.0.self_attn.q_proj"
    category: str                # "compute" | "communication" | "memory"

    # 由 Transform 阶段按需填充
    annotations: dict[str, Any] = field(default_factory=dict)

@dataclass
class Edge:
    src: str                     # src node id
    src_idx: int                 # output index
    dst: str                     # dst node id
    dst_idx: int                 # input index
    tensor: TensorMeta | None    # None = control edge

@dataclass
class OpGraph:
    name: str
    phase: str                   # "prefill" / "decode"
    nodes: dict[str, OpNode]
    edges: list[Edge]
    metadata: dict[str, Any]     # model, batch_size, seq_len, ...

    # ── 图操作 ──
    def topo_sort(self) -> list[OpNode]: ...
    def predecessors(self, node_id: str) -> list[str]: ...
    def successors(self, node_id: str) -> list[str]: ...
    def subgraph(self, node_ids: set[str]) -> "OpGraph": ...
    def insert_after(self, ref_id: str, new_node: OpNode, edges: list[Edge]): ...
    def replace_subgraph(self, old_ids: set[str], new_node: OpNode): ...
    def clone(self) -> "OpGraph": ...

    # ── 层次化视图 ──
    @cached_property
    def hierarchy(self) -> "GraphHierarchy": ...
```

#### 层次化视图

从 scope 字符串自动构建树，支持任意粒度的指标聚合：

```python
@dataclass
class HierNode:
    """层次化节点，可以是一个模块、一层、或整个模型"""
    scope: str                          # "model.layers.0.self_attn"
    children: list["HierNode"]
    leaf_node_ids: list[str]            # 所属的 OpNode IDs
    _metrics: dict[str, float] = field(default_factory=dict)

class GraphHierarchy:
    """scope 树，从 OpGraph 自动构建"""

    def __init__(self, graph: OpGraph):
        self.root = self._build_tree(graph)

    def at_depth(self, depth: int) -> list[HierNode]:
        """depth=0 → 整图, 1 → embed/layers/lm_head, 2 → 各层, 3 → attn/mlp/norm"""
        ...

    def aggregate(self, node: HierNode, metric: str,
                  values: dict[str, float]) -> float:
        """递归求和子树的某个指标（latency, flops, memory_bytes, ...）"""
        if node.leaf_node_ids:
            return sum(values.get(nid, 0) for nid in node.leaf_node_ids)
        return sum(self.aggregate(c, metric, values) for c in node.children)

    def find(self, scope_pattern: str) -> list[HierNode]:
        """glob 匹配 scope，如 'model.layers.*.mlp'"""
        ...
```

**使用场景**：
```python
# 报表中生成"模块级耗时分解"
latency_map = {r.op_node_id: r.latency_us for r in sim_results}
for module in graph.hierarchy.at_depth(3):      # attn / mlp / norm 级别
    total = graph.hierarchy.aggregate(module, "latency", latency_map)
    print(f"{module.scope}: {total:.1f} us")
```

### 2.2 硬件注册表

纯硬件规格，不含任何软件/框架信息。

```
zrt/
└── hardware/
    ├── __init__.py
    ├── spec.py              # HardwareSpec, ComputeSpec, MemorySpec, InterconnectSpec
    ├── registry.py          # load(name) → HardwareSpec
    └── configs/
        ├── ascend_910b.yaml
        ├── ascend_910c.yaml
        ├── nvidia_a100_80g.yaml
        ├── nvidia_h100_sxm.yaml
        ├── nvidia_h800.yaml
        └── ...
```

```yaml
# hardware/configs/ascend_910b.yaml
name: "Ascend 910B"
vendor: "huawei"
device_type: "npu"

compute:
  # 峰值算力（理论值），仿真器可通过效率系数折算
  fp16_tflops: 320
  bf16_tflops: 320
  fp32_tflops: 160
  int8_tops: 640
  int4_tops: 1280

memory:
  capacity_gb: 64
  hbm_bandwidth_gbps: 1600
  l2_cache_mb: 48
  # 多级带宽（仿真器可按数据量选择命中哪一级）
  tiers:
    - { name: "L2",  bandwidth_gbps: 6400, capacity_mb: 48 }
    - { name: "HBM", bandwidth_gbps: 1600 }

interconnect:
  intra_node:
    type: "HCCS"
    unidirectional_bw_gbps: 56     # 单链路单向
    num_links: 7                    # 910B 每卡 7 条 HCCS link
    latency_us: 3
    topology: "full_mesh"
    num_devices: 8
  inter_node:
    type: "RoCE"
    bandwidth_gbps: 200
    latency_us: 5
```

```python
@dataclass
class HardwareSpec:
    name: str
    vendor: str
    device_type: str                 # "npu" | "gpu" | "cpu"
    compute: ComputeSpec
    memory: MemorySpec
    interconnect: InterconnectSpec

    def peak_flops(self, dtype: DType) -> float:
        """返回 FLOPs/s"""
        ...

    def hbm_bandwidth(self) -> float:
        """返回 bytes/s"""
        ...

@dataclass
class InterconnectSpec:
    intra_node: LinkSpec
    inter_node: LinkSpec

@dataclass
class LinkSpec:
    type: str
    bandwidth_gbps: float     # 总有效双向带宽（由 registry 从 yaml 计算）
    latency_us: float
    topology: str
    num_devices: int = 1
```

### 2.3 软件栈注册表

**与硬件正交**——同一硬件可跑不同栈，同一栈可跑不同硬件。

```
zrt/
└── stacks/
    ├── __init__.py
    ├── base.py              # SoftwareStack ABC
    ├── registry.py          # load(name) → SoftwareStack
    ├── mindie/
    │   ├── __init__.py
    │   ├── fusion_rules.py  # MindIE 支持的融合规则
    │   ├── op_mapping.py    # aten op → MindIE 算子映射
    │   └── optim_caps.py    # 支持的优化能力声明
    ├── vllm/
    │   ├── __init__.py
    │   ├── fusion_rules.py
    │   ├── op_mapping.py
    │   └── optim_caps.py
    ├── tensorrt_llm/
    │   └── ...
    └── generic/             # 理想化基线（不做融合、标准通信）
        └── ...
```

```python
class SoftwareStack(ABC):
    """软件栈接口——声明框架的能力，不感知具体硬件参数"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def fusion_rules(self) -> list["FusionRule"]:
        """此栈支持的融合规则列表"""
        ...

    @abstractmethod
    def op_mapping(self, aten_op: str) -> str | None:
        """aten op → 此栈的内核名（用于查 profiling 数据）
        返回 None 表示直接使用 aten 语义（不做映射）"""
        ...

    @abstractmethod
    def supported_parallel(self) -> set[str]:
        """支持的并行策略 {'tp', 'ep', 'sp', 'pp', 'dp'}"""
        ...

    @abstractmethod
    def supported_optimizations(self) -> set[str]:
        """支持的优化项 {'flash_attn', 'eplb', 'mtp', 'kv_quant', ...}"""
        ...

    @abstractmethod
    def supported_dtypes(self) -> set[DType]:
        """支持的计算数据类型"""
        ...
```

**组合使用**：
```python
# 用户配置：hardware = "ascend_910b", stack = "mindie"
hw = hardware_registry.load("ascend_910b")    # 纯硬件参数
stack = stack_registry.load("mindie")         # 纯软件能力

# 后续各模块接收 (hw, stack) 二元组
transform_ctx = TransformContext(hw_spec=hw, stack=stack, ...)
sim_hub.simulate(node, hw_spec=hw, stack=stack)
```

### 2.4 通信模型

独立模块，不是简单的 `bytes / bandwidth`。

```
zrt/
└── comm/
    ├── __init__.py
    ├── model.py             # CommModel: 通信耗时估算
    ├── algorithms.py        # Ring / Tree / RecursiveHalving 等集合通信算法
    └── topology.py          # 拓扑感知路由（节点内 vs 跨节点）
```

```python
@dataclass
class CommEstimate:
    latency_us: float
    algorithm: str              # "ring" | "tree" | "recursive_halving"
    is_cross_node: bool
    effective_bw_gbps: float    # 实际有效带宽
    num_segments: int           # 分段数（用于流水线重叠建模）
    segment_latency_us: float   # 单段耗时

class CommModel:
    """
    通信耗时估算。
    输入：集合通信类型 + 数据量 + 设备数 + 拓扑。
    输出：耗时估算（含算法选择和分段信息）。
    """

    def estimate(self,
                 collective: str,        # "all_reduce" | "all_gather" | "reduce_scatter"
                                         # | "all_to_all" | "send_recv"
                 data_bytes: int,
                 group_size: int,         # 参与的设备数
                 hw_spec: HardwareSpec,
                 cross_node: bool = False,
                 algorithm: str = "auto",
                 ) -> CommEstimate:

        link = hw_spec.interconnect.inter_node if cross_node \
               else hw_spec.interconnect.intra_node

        if algorithm == "auto":
            algorithm = self._select_algorithm(collective, data_bytes, group_size)

        if collective == "all_reduce" and algorithm == "ring":
            # Ring AllReduce: 2*(n-1)/n * D / BW + 2*(n-1) * lat
            n = group_size
            bw = link.bandwidth_gbps * 1e9 / 8   # bytes/s
            volume = 2 * (n - 1) / n * data_bytes
            latency_us = volume / bw * 1e6 + 2 * (n - 1) * link.latency_us
            # 分段：大数据量可切 S 段做流水
            num_segments = max(1, data_bytes // (4 * 1024 * 1024))  # 每段 ~4MB
            ...

        elif collective == "all_to_all":
            # A2A: 每对设备交换 D/(n*n) 数据，总耗时取决于最慢的链路
            ...

        return CommEstimate(
            latency_us=latency_us,
            algorithm=algorithm,
            is_cross_node=cross_node,
            effective_bw_gbps=...,
            num_segments=num_segments,
            segment_latency_us=latency_us / num_segments,
        )

    def _select_algorithm(self, collective, data_bytes, group_size) -> str:
        """基于经验的算法选择：
        - 小数据量（<256KB）：Tree（低延迟）
        - 大数据量（>1MB）：Ring（高带宽利用）
        - 中间：RecursiveHalving
        """
        if data_bytes < 256 * 1024:
            return "tree"
        elif data_bytes > 1024 * 1024:
            return "ring"
        return "recursive_halving"
```

---

## 3. Core Layer

### 3.1 模型管理器

```
zrt/
└── model/
    ├── __init__.py
    ├── loader.py            # load_model(model_id, num_layers) → (model, config, fake_mode)
    ├── compat.py            # transformers 版本 shim
    ├── patches.py           # 运行时 monkey-patch (MoE, Indexer, ...)
    ├── registry.py          # 本地模型注册表 (model_type → hf_models/ 路径)
    └── profile.py           # ModelProfile: 从 config.json 提取结构化参数
```

```python
@dataclass
class ModelProfile:
    """从 config.json 解析的结构化模型参数"""
    model_type: str                   # "llama", "qwen2", "deepseek_v3", ...
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    # MoE
    num_experts: int | None = None
    num_shared_experts: int | None = None
    moe_topk: int | None = None
    # MLA
    kv_lora_rank: int | None = None
    q_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    # MTP
    num_mtp_heads: int | None = None

    @classmethod
    def from_config(cls, config) -> "ModelProfile":
        """从 HF PretrainedConfig 对象自动提取，字段不存在则 None"""
        ...

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def is_moe(self) -> bool:
        return self.num_experts is not None and self.num_experts > 1

    @property
    def is_gqa(self) -> bool:
        return self.num_kv_heads < self.num_attention_heads

    def param_count(self, dtype_bytes: int = 2) -> int:
        """估算总参数量（不含 embedding 共享的重复计算）"""
        ...
```

### 3.2 图抓取

```
zrt/
└── capture/
    ├── __init__.py
    ├── tracer.py            # capture(model, phase, ...) → OpGraph
    ├── dispatch.py          # RecordingDispatch: aten op 拦截 → OpNode 列表
    ├── tracker.py           # ModuleTracker: forward hook → scope 路径
    ├── input_builder.py     # 按 phase 构建 FakeTensor 输入
    └── graph_builder.py     # op records → OpGraph (建边、建层次)
```

```python
def capture(model, fake_mode, profile: ModelProfile,
            phase: str,            # "prefill" | "decode"
            batch_size: int = 1,
            seq_len: int = 128,
            ) -> OpGraph:
    """
    在 FakeTensorMode 下执行一次 forward，拦截全部 aten 算子，
    构建 OpGraph。

    prefill 和 decode 在同一 fake_mode 中执行：
    prefill 产生 KV cache (FakeTensor)，decode 直接消费。
    """
    inputs = InputBuilder.build(profile, phase, batch_size, seq_len, fake_mode)
    with RecordingDispatch() as recorder, ModuleTracker(model) as tracker:
        model(**inputs)
    return GraphBuilder.build(
        records=recorder.records,
        scopes=tracker.scopes,
        name=f"{profile.model_type}_{phase}",
        phase=phase,
        metadata={"batch_size": batch_size, "seq_len": seq_len},
    )
```

### 3.3 显存模型

独立模块，不依赖执行仿真。在两个场景使用：
1. **寻优剪枝**：快速判断配置是否可行，不需要跑完整仿真
2. **报表输出**：生成显存分解（权重/KV/激活/通信 buffer）

```
zrt/
└── memory/
    ├── __init__.py
    ├── model.py             # MemoryModel: 各项显存估算
    ├── budget.py            # MemoryBudget: 分解结果
    └── activation.py        # 基于图的激活显存分析（生命周期）
```

```python
@dataclass
class MemoryBudget:
    weights_mb: float
    kv_cache_mb: float
    activation_peak_mb: float
    comm_buffer_mb: float
    framework_overhead_mb: float    # 框架 + 碎片估算（固定比例）
    total_mb: float
    capacity_mb: float
    is_feasible: bool               # total <= capacity

    def breakdown(self) -> dict[str, float]:
        """返回各项占比，用于报表"""
        ...

class MemoryModel:
    """显存估算器——纯公式计算，不依赖图执行"""

    def estimate(self,
                 profile: ModelProfile,
                 hw_spec: HardwareSpec,
                 parallel: "ParallelConfig",
                 quant: "QuantConfig | None" = None,
                 batch_size: int = 1,
                 seq_len: int = 4096,
                 ) -> MemoryBudget:
        w = self._weights(profile, parallel, quant)
        kv = self._kv_cache(profile, parallel, quant, batch_size, seq_len)
        act = self._activation_peak(profile, parallel, batch_size, seq_len)
        comm = self._comm_buffer(profile, parallel)
        overhead = (w + kv + act + comm) * 0.05   # 5% 碎片估算
        total = w + kv + act + comm + overhead
        capacity = hw_spec.memory.capacity_gb * 1024
        return MemoryBudget(
            weights_mb=w, kv_cache_mb=kv,
            activation_peak_mb=act, comm_buffer_mb=comm,
            framework_overhead_mb=overhead, total_mb=total,
            capacity_mb=capacity, is_feasible=total <= capacity,
        )

    def _weights(self, profile, parallel, quant) -> float:
        """权重显存 = total_params * bytes_per_param / tp / pp
        量化时 bytes_per_param 按量化位宽计算"""
        params = profile.param_count()
        bytes_per = quant.weight_bytes if quant else 2  # default bf16
        return params * bytes_per / parallel.tp / parallel.pp / (1024**2)

    def _kv_cache(self, profile, parallel, quant, bs, seq_len) -> float:
        """KV = 2 * num_layers_local * kv_heads_local * head_dim * seq * bs * dtype"""
        layers = profile.num_layers // parallel.pp
        kv_heads = profile.num_kv_heads // parallel.tp
        head_dim = profile.head_dim
        # MLA 架构：kv_lora_rank 替代 kv_heads * head_dim
        if profile.kv_lora_rank:
            kv_dim = profile.kv_lora_rank + profile.qk_rope_head_dim
        else:
            kv_dim = kv_heads * head_dim
        kv_bytes = quant.kv_bytes if (quant and quant.kv_bytes) else 2
        return 2 * layers * kv_dim * seq_len * bs * kv_bytes / (1024**2)

    def _activation_peak(self, profile, parallel, bs, seq_len) -> float:
        """激活峰值 ≈ 最大的中间 tensor 占用
        经验公式：~34 * bs * seq * h / tp  (bytes, for bf16 transformer layer)"""
        h = profile.hidden_size
        return 34 * bs * seq_len * h / parallel.tp / (1024**2)

    def _comm_buffer(self, profile, parallel) -> float:
        """通信 buffer：TP all_reduce 需要 hidden_size 大小的临时 buffer"""
        if parallel.tp <= 1:
            return 0
        return profile.hidden_size * 2 * 2 / (1024**2)  # 2 buffers, bf16
```

### 3.4 图变换

**核心改动**：分 4 个 Stage 固定顺序执行，**先切分再融合**。

```
zrt/
└── transform/
    ├── __init__.py
    ├── pipeline.py          # TransformPipeline: 4-stage 编排
    ├── base.py              # GraphPass ABC
    ├── context.py           # TransformContext
    │
    ├── parallel/            # Stage 1: 并行切分
    │   ├── __init__.py
    │   ├── tensor_parallel.py
    │   ├── expert_parallel.py
    │   ├── sequence_parallel.py
    │   ├── pipeline_parallel.py
    │   └── comm_inserter.py
    │
    ├── fusion/              # Stage 2: 算子融合
    │   ├── __init__.py
    │   ├── engine.py        # FusionEngine: 模式匹配 + 子图替换
    │   └── patterns.py      # FusionRule 基类 + 匹配框架
    │
    ├── optim/               # Stage 3: 优化 Pass
    │   ├── __init__.py
    │   ├── quantization.py  # 量化标注 (W8A8, W4A16, KV int8, ...)
    │   ├── eplb.py          # Expert-Level Load Balancing
    │   ├── shared_expert.py # 共享专家外置（和计算重叠）
    │   ├── mtp.py           # Multi-Token Prediction
    │   ├── kv_cache.py      # KV cache 优化 (PagedAttn shape 调整)
    │   └── recompute.py     # 重计算（训练场景）
    │
    └── analysis/            # Stage 4: 标注分析
        ├── __init__.py
        ├── flops.py         # FLOPs/MACs 估算
        └── roofline.py      # Roofline 标注 (compute/memory bound)
```

#### Pass 接口

```python
class GraphPass(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def run(self, graph: OpGraph, ctx: "TransformContext") -> OpGraph:
        """纯函数：输入图 → 输出新图，不修改原图"""
        ...

@dataclass
class TransformContext:
    hw_spec: HardwareSpec
    stack: SoftwareStack
    profile: ModelProfile
    phase: str
    parallel: ParallelConfig
    quant: QuantConfig | None = None
    optim_flags: set[str] = field(default_factory=set)  # {"eplb", "mtp", ...}
```

#### Pipeline 编排

```python
class TransformPipeline:
    """
    4-stage 图变换管线。

    Stage 顺序固定：Split → Fuse → Optim → Analyze
    每个 Stage 内的 Pass 按注册顺序执行。
    Pass 可声明 condition，不满足则跳过。
    """

    def __init__(self):
        self._stages: dict[str, list[tuple[GraphPass, Callable | None]]] = {
            "split": [], "fuse": [], "optim": [], "analyze": [],
        }

    def add(self, stage: str, pass_: GraphPass,
            condition: Callable[[TransformContext], bool] | None = None):
        self._stages[stage].append((pass_, condition))

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        current = graph
        for stage_name in ("split", "fuse", "optim", "analyze"):
            for pass_, cond in self._stages[stage_name]:
                if cond and not cond(ctx):
                    continue
                current = pass_.run(current, ctx)
        return current


def build_default_pipeline() -> TransformPipeline:
    pipe = TransformPipeline()

    # Stage 1: Split
    pipe.add("split", TensorParallelPass(),
             condition=lambda c: c.parallel.tp > 1)
    pipe.add("split", ExpertParallelPass(),
             condition=lambda c: c.parallel.ep > 1)
    pipe.add("split", SequenceParallelPass(),
             condition=lambda c: c.parallel.sp)
    pipe.add("split", PipelineParallelPass(),
             condition=lambda c: c.parallel.pp > 1)
    pipe.add("split", CommInserterPass())     # 在切分点插入通信节点

    # Stage 2: Fuse (融合规则来自 SoftwareStack)
    pipe.add("fuse", FusionPass())            # 内部调用 ctx.stack.fusion_rules()

    # Stage 3: Optim
    pipe.add("optim", QuantizationPass(),
             condition=lambda c: c.quant is not None)
    pipe.add("optim", EPLBPass(),
             condition=lambda c: "eplb" in c.optim_flags)
    pipe.add("optim", SharedExpertPass(),
             condition=lambda c: "shared_expert_external" in c.optim_flags)
    pipe.add("optim", MTPPass(),
             condition=lambda c: "mtp" in c.optim_flags)

    # Stage 4: Analyze
    pipe.add("analyze", FlopsPass())
    pipe.add("analyze", RooflinePass())

    return pipe
```

#### 为什么先切分再融合

```
错误顺序（先融合再切分）:

  aten.linear → aten.add → aten.rms_norm        →   fused(linear_add_rmsnorm)
  如果 linear 做 TP 需要在 linear 后 all_reduce，但融合后无法在中间插入通信

正确顺序（先切分再融合）:

  aten.linear → comm.all_reduce → aten.add → aten.rms_norm
  融合只在 all_reduce 之后的子图内进行 →   fused(add_rmsnorm)
  通信边界天然成为融合边界，规则不需要感知并行策略
```

#### Tensor Parallel 切分

```python
class TensorParallelPass(GraphPass):
    name = "tensor_parallel"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        g = graph.clone()
        tp = ctx.parallel.tp
        for node in g.topo_sort():
            rule = self._match(node, ctx.profile)
            if rule is None:
                continue
            # 修改 shape: 被切分维度 / tp
            for out in node.outputs:
                shape = list(out.shape)
                shape[rule.split_dim] //= tp
                out.shape = tuple(shape)
            for inp in node.inputs:
                if rule.input_split:
                    shape = list(inp.shape)
                    shape[rule.split_dim] //= tp
                    inp.shape = tuple(shape)
            # 标注，供后续 CommInserter 使用
            node.annotations["tp_split"] = rule
        return g

    def _match(self, node: OpNode, profile: ModelProfile) -> TPRule | None:
        """根据 scope + op_type 判断切分规则。
        利用 scope 中的关键词（q_proj, k_proj, gate_proj, ...）识别位置，
        不硬编码模型架构。"""
        scope = node.scope.lower()
        if node.op_type not in ("aten.mm", "aten.linear", "aten.addmm"):
            return None
        # Column parallel: Q/K/V/gate/up projection
        if any(k in scope for k in ("q_proj", "k_proj", "v_proj",
                                      "gate_proj", "up_proj", "w1", "w2", "w3")):
            return TPRule(split_dim=-1, comm_after=None, input_split=False)
        # Row parallel: O/down projection → 后接 all_reduce
        if any(k in scope for k in ("o_proj", "down_proj")):
            return TPRule(split_dim=-2, comm_after="all_reduce", input_split=True)
        return None
```

#### Expert Parallel 切分

```python
class ExpertParallelPass(GraphPass):
    name = "expert_parallel"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        g = graph.clone()
        ep = ctx.parallel.ep
        profile = ctx.profile
        if not profile.is_moe or ep <= 1:
            return g

        experts_per_rank = profile.num_experts // ep

        for node in g.topo_sort():
            if not self._is_expert_op(node):
                continue
            # 每个 rank 只算 experts_per_rank 个专家
            # 对 expert 维度的 shape 做缩减
            node.annotations["ep_experts_local"] = experts_per_rank
            node.annotations["ep_needs_a2a"] = True
            # token dispatch: 输入 tokens 需要 all-to-all 分发
            # token combine:  输出 tokens 需要 all-to-all 汇聚

        return g
```

#### CommInserter：在切分点插入通信节点

```python
class CommInserterPass(GraphPass):
    """
    扫描所有被 Split Pass 标注的节点，在需要的位置插入通信算子节点。
    通信算子节点的 op_type 以 "comm." 开头，category = "communication"。
    """
    name = "comm_inserter"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        g = graph.clone()
        comm_model = CommModel()

        for node in list(g.topo_sort()):
            # TP all_reduce
            tp_rule = node.annotations.get("tp_split")
            if tp_rule and tp_rule.comm_after:
                comm_node = OpNode(
                    id=f"comm_{node.id}",
                    op_type=f"comm.{tp_rule.comm_after}",
                    inputs=node.outputs.copy(),
                    outputs=node.outputs.copy(),  # shape 不变
                    attrs={"group_size": ctx.parallel.tp},
                    scope=node.scope,
                    category="communication",
                )
                # 估算通信数据量
                data_bytes = sum(t.memory_bytes for t in node.outputs)
                est = comm_model.estimate(
                    tp_rule.comm_after, data_bytes,
                    ctx.parallel.tp, ctx.hw_spec,
                )
                comm_node.annotations["comm_estimate"] = est
                g.insert_after(node.id, comm_node, ...)

            # EP all-to-all
            if node.annotations.get("ep_needs_a2a"):
                # 插入 dispatch A2A (在 expert 计算前) 和 combine A2A (在计算后)
                ...

        return g
```

### 3.5 算子仿真器 Hub

```
zrt/
└── simulator/
    ├── __init__.py
    ├── hub.py               # SimulatorHub: 路由 + 缓存
    ├── base.py              # OpSimulator ABC
    ├── result.py            # SimResult
    ├── cache.py             # content-hash 缓存
    └── backends/
        ├── __init__.py
        ├── roofline.py      # 理论公式 (priority=0, 兜底)
        ├── profile_db.py    # 真机查表 (priority=30)
        ├── regression.py    # 回归拟合 (priority=50)
        ├── tiling_sim.py    # Tiling 级仿真适配器 (priority=100)
        └── custom.py        # 用户自定义后端注册接口
```

#### SimResult

```python
@dataclass
class SimResult:
    op_node_id: str
    latency_us: float            # 算子总耗时
    compute_us: float            # 计算部分
    memory_us: float             # 访存部分
    flops: int
    read_bytes: int
    write_bytes: int
    arithmetic_intensity: float  # flops / (read + write)
    bound: str                   # "compute" | "memory" | "latency"
    hw_utilization: float        # 0~1
    backend: str                 # 实际使用的后端名
    confidence: float            # 0~1
```

#### Hub 路由 + content-hash 缓存

```python
class SimulatorHub:
    def __init__(self):
        self._backends: list[OpSimulator] = []
        self._cache: dict[int, SimResult] = {}

    def register(self, backend: OpSimulator):
        self._backends.append(backend)
        self._backends.sort(key=lambda b: b.priority, reverse=True)

    def simulate(self, node: OpNode, hw: HardwareSpec) -> SimResult:
        key = self._content_hash(node, hw)
        if key in self._cache:
            return self._cache[key]
        for b in self._backends:
            if b.can_simulate(node, hw):
                r = b.simulate(node, hw)
                self._cache[key] = r
                return r
        raise RuntimeError(f"No backend can simulate {node.op_type}")

    def simulate_graph(self, graph: OpGraph, hw: HardwareSpec) -> dict[str, SimResult]:
        return {n.id: self.simulate(n, hw) for n in graph.topo_sort()}

    @staticmethod
    def _content_hash(node: OpNode, hw: HardwareSpec) -> int:
        """基于 (op_type, input shapes/dtypes, attrs, hw.name) 的稳定 hash。
        融合算子的 hash 包含子算子列表。"""
        import hashlib
        parts = [node.op_type, hw.name]
        for t in node.inputs:
            parts.extend([str(t.shape), t.dtype.name])
        parts.append(str(sorted(node.attrs.items())))
        return int(hashlib.md5("|".join(parts).encode()).hexdigest(), 16)
```

#### Roofline 后端

```python
class RooflineSimulator(OpSimulator):
    """
    Roofline Model 理论估算。零外部依赖，任何算子都能估。

    matmul:   FLOPs = 2*M*N*K, read = (M*K + K*N)*dtype, write = M*N*dtype
    layernorm: FLOPs ≈ 5*N, read/write = N*dtype
    softmax:  FLOPs ≈ 5*N, read/write = N*dtype
    elementwise: FLOPs = N, read/write = N*dtype * num_inputs
    """
    name = "roofline"
    priority = 0

    # aten op → (flops_fn, memory_fn) 的注册表
    _OP_FORMULAS: dict[str, tuple[Callable, Callable]] = {}

    @classmethod
    def register_formula(cls, op_type: str, flops_fn, memory_fn):
        cls._OP_FORMULAS[op_type] = (flops_fn, memory_fn)

    def can_simulate(self, node, hw):
        return True

    def simulate(self, node: OpNode, hw: HardwareSpec) -> SimResult:
        flops_fn, mem_fn = self._OP_FORMULAS.get(
            node.op_type, (self._default_flops, self._default_memory))

        flops = flops_fn(node)
        read_bytes, write_bytes = mem_fn(node)
        total_bytes = read_bytes + write_bytes

        peak = hw.peak_flops(node.outputs[0].dtype)
        bw = hw.hbm_bandwidth()

        compute_us = flops / peak * 1e6 if peak > 0 else 0
        memory_us = total_bytes / bw * 1e6 if bw > 0 else 0
        latency_us = max(compute_us, memory_us)

        ai = flops / total_bytes if total_bytes > 0 else float("inf")
        bound = "compute" if compute_us >= memory_us else "memory"
        util = min(1.0, flops / (latency_us * 1e-6 * peak)) if peak > 0 else 0

        return SimResult(
            op_node_id=node.id, latency_us=latency_us,
            compute_us=compute_us, memory_us=memory_us,
            flops=flops, read_bytes=read_bytes, write_bytes=write_bytes,
            arithmetic_intensity=ai, bound=bound,
            hw_utilization=util, backend=self.name, confidence=0.3,
        )
```

#### ProfileDB 后端（真机查表）

```python
class ProfileDBSimulator(OpSimulator):
    """
    精确查表：从真机 profiling 数据中查找完全匹配的记录。
    数据格式：CSV，列包含 op_type, input_shapes, dtype, hardware, latency_us, ...
    """
    name = "profile_db"
    priority = 30

    def __init__(self, data_dir: str):
        self._db = self._load(data_dir)    # dict[(op_type, shapes_key, dtype, hw) → row]

    def can_simulate(self, node, hw):
        key = self._make_key(node, hw)
        return key in self._db

    def simulate(self, node, hw):
        row = self._db[self._make_key(node, hw)]
        return SimResult(
            ..., backend=self.name, confidence=0.9,
        )
```

### 3.6 图执行器

```
zrt/
└── executor/
    ├── __init__.py
    ├── scheduler.py         # DAGScheduler: 多流 DAG 调度
    ├── stream.py            # Stream: 流抽象
    ├── timeline.py          # Timeline: 调度结果 + 查询 API
    └── overlap.py           # OverlapAnalyzer: 通算掩盖分析
```

```python
@dataclass
class StreamConfig:
    num_compute_streams: int = 1
    num_comm_streams: int = 1

    @property
    def total(self) -> int:
        return self.num_compute_streams + self.num_comm_streams

@dataclass
class ScheduledOp:
    node_id: str
    stream_id: int
    stream_type: str           # "compute" | "comm"
    start_us: float
    end_us: float

class Timeline:
    """调度结果，支持多种查询"""

    def __init__(self):
        self._ops: list[ScheduledOp] = []

    def add(self, op: ScheduledOp): ...

    @property
    def total_latency_us(self) -> float:
        return max(op.end_us for op in self._ops) if self._ops else 0

    def by_stream(self, stream_id: int) -> list[ScheduledOp]: ...
    def by_scope_prefix(self, prefix: str) -> list[ScheduledOp]: ...
    def by_type(self, stream_type: str) -> list[ScheduledOp]: ...

    def intervals(self, stream_type: str) -> list[tuple[float, float]]:
        """返回某类型流的 [(start, end), ...] 区间列表"""
        return [(op.start_us, op.end_us) for op in self._ops
                if op.stream_type == stream_type]
```

#### DAG 调度器

```python
class DAGScheduler:
    """
    仿真调度：
    1. 拓扑排序确定执行顺序
    2. 就绪队列（所有前驱完成的节点）
    3. 选流：compute 算子 → compute stream，comm 算子 → comm stream
    4. 同一流内串行，不同流并行
    5. 同步点：有数据依赖时等待

    通算掩盖的实现方式：
    - comm 算子被分配到 comm stream，和 compute stream 并行执行
    - 依赖 comm 结果的 compute 算子需要等 comm 完成（自然同步）
    - 不依赖 comm 结果的 compute 算子可以和 comm 完全重叠
    """

    def schedule(self, graph: OpGraph,
                 sim_results: dict[str, SimResult],
                 stream_config: StreamConfig,
                 ) -> Timeline:

        timeline = Timeline()
        in_degree = {n.id: len(graph.predecessors(n.id)) for n in graph.nodes.values()}
        stream_clock = [0.0] * stream_config.total
        node_end = {}   # node_id → end_time

        # 初始化就绪队列
        ready = [nid for nid, deg in in_degree.items() if deg == 0]
        heapq.heapify(ready)   # 按 node id 稳定排序

        while ready:
            node_id = heapq.heappop(ready)
            node = graph.nodes[node_id]

            # 前驱最晚完成时间
            deps_done = max((node_end[p] for p in graph.predecessors(node_id)),
                            default=0.0)

            # 选流
            if node.category == "communication":
                sid = self._pick_comm_stream(stream_config, stream_clock)
            else:
                sid = self._pick_compute_stream(stream_config, stream_clock)

            start = max(deps_done, stream_clock[sid])
            duration = sim_results[node_id].latency_us
            end = start + duration

            timeline.add(ScheduledOp(
                node_id=node_id, stream_id=sid,
                stream_type="comm" if node.category == "communication" else "compute",
                start_us=start, end_us=end,
            ))
            stream_clock[sid] = end
            node_end[node_id] = end

            # 释放后继
            for succ_id in graph.successors(node_id):
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    heapq.heappush(ready, succ_id)

        return timeline
```

#### 通算掩盖分析

```python
class OverlapAnalyzer:
    def analyze(self, timeline: Timeline) -> "OverlapReport":
        comp = timeline.intervals("compute")
        comm = timeline.intervals("comm")

        total_comp = self._sum_duration(comp)
        total_comm = self._sum_duration(comm)
        overlap = self._intersection(comp, comm)

        return OverlapReport(
            compute_us=total_comp,
            comm_us=total_comm,
            overlap_us=overlap,
            exposed_comm_us=total_comm - overlap,
            overlap_ratio=overlap / total_comm if total_comm > 0 else 1.0,
            critical_path_us=timeline.total_latency_us,
        )

    def _intersection(self, a: list[tuple], b: list[tuple]) -> float:
        """计算两组区间的重叠总长度（扫描线算法）"""
        events = []
        for s, e in a:
            events.append((s, 1, "a"))
            events.append((e, -1, "a"))
        for s, e in b:
            events.append((s, 1, "b"))
            events.append((e, -1, "b"))
        events.sort()

        overlap = 0.0
        active_a = active_b = 0
        prev_t = 0.0
        for t, delta, typ in events:
            if active_a > 0 and active_b > 0:
                overlap += t - prev_t
            if typ == "a":
                active_a += delta
            else:
                active_b += delta
            prev_t = t
        return overlap
```

### 3.7 报表生成器

```
zrt/
└── report/
    ├── __init__.py
    ├── engine.py            # ReportEngine: 编排所有 writer
    ├── excel_writer.py      # 算子级明细 (xlsx)
    ├── html_writer.py       # 交互式 timeline + 饼图 (html)
    ├── onnx_writer.py       # 计算图可视化 (onnx, Netron 查看)
    ├── json_writer.py       # 结构化数据 (json, 供外部工具消费)
    └── summary.py           # E2E 汇总指标
```

#### Excel 列设计

```
| # | op_type | scope | component | input_shapes | output_shapes | dtype |
    flops | read_bytes | write_bytes | arithmetic_intensity |
    latency_us | compute_us | memory_us | bound |
    hw_utilization | backend | confidence |
    fused_into | parallel_split | comm_type |
```

#### E2E 汇总

```python
@dataclass
class E2ESummary:
    model: str
    hardware: str
    stack: str
    phase: str
    parallel_desc: str               # "TP8-EP8-PP1"

    # 核心指标
    latency_ms: float                # 单次推理延迟
    tokens_per_sec: float            # 吞吐
    ttft_ms: float | None            # prefill 特有
    tpot_ms: float | None            # decode 特有

    # 分解
    compute_ms: float
    comm_ms: float
    exposed_comm_ms: float
    overlap_ratio: float
    memory_budget: MemoryBudget

    # 层次化分解
    by_component: dict[str, float]   # {"attention": 45.2, "mlp": 38.1, ...} (%)
    by_layer: list[float]            # 每层耗时 (ms)
    top_bottleneck_ops: list[tuple[str, float]]  # [(op_desc, latency_us), ...]
```

---

## 4. Application Layer

### 4.1 Orchestrator

```python
@dataclass
class RunConfig:
    model_id: str
    hardware: str                     # YAML 名
    stack: str                        # 软件栈名
    phases: list[str] = field(default_factory=lambda: ["prefill", "decode"])
    num_layers: int = 0               # 0 = 全量
    batch_size: int = 1
    seq_len: int = 4096
    output_len: int = 512
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    quant: QuantConfig | None = None
    optim_flags: set[str] = field(default_factory=set)
    stream_config: StreamConfig = field(default_factory=StreamConfig)
    output_dir: str = ""
    output_formats: list[str] = field(default_factory=lambda: ["excel", "json"])

@dataclass
class ParallelConfig:
    tp: int = 1
    pp: int = 1
    ep: int = 1
    dp: int = 1
    sp: bool = False

    @property
    def total_devices(self) -> int:
        return self.tp * self.pp * self.ep * self.dp

    def describe(self) -> str:
        parts = []
        if self.tp > 1: parts.append(f"TP{self.tp}")
        if self.ep > 1: parts.append(f"EP{self.ep}")
        if self.pp > 1: parts.append(f"PP{self.pp}")
        if self.dp > 1: parts.append(f"DP{self.dp}")
        if self.sp: parts.append("SP")
        return "-".join(parts) or "single"

class Orchestrator:
    def __init__(self, model_mgr, hw_registry, stack_registry,
                 simulator_hub, comm_model, memory_model,
                 pipeline, report_engine):
        self.model_mgr = model_mgr
        self.hw_reg = hw_registry
        self.stack_reg = stack_registry
        self.sim_hub = simulator_hub
        self.comm = comm_model
        self.mem = memory_model
        self.pipeline = pipeline
        self.report = report_engine

    def run(self, config: RunConfig) -> "RunResult":
        # 1. 加载
        model, profile, fake_mode = self.model_mgr.load(
            config.model_id, config.num_layers)
        hw = self.hw_reg.load(config.hardware)
        stack = self.stack_reg.load(config.stack)

        # 2. 显存可行性快速检查
        mem_budget = self.mem.estimate(
            profile, hw, config.parallel, config.quant,
            config.batch_size, config.seq_len)
        if not mem_budget.is_feasible:
            return RunResult(feasible=False, memory_budget=mem_budget)

        # 3. 图抓取
        graphs = {}
        for phase in config.phases:
            graphs[phase] = capture(model, fake_mode, profile, phase,
                                    config.batch_size, config.seq_len)
        fake_mode.__exit__(None, None, None)

        # 4. 图变换
        ctx = TransformContext(
            hw_spec=hw, stack=stack, profile=profile,
            parallel=config.parallel, quant=config.quant,
            optim_flags=config.optim_flags,
        )
        for phase in config.phases:
            ctx.phase = phase
            graphs[phase] = self.pipeline.run(graphs[phase], ctx)

        # 5. 仿真
        sim_results = {}
        for phase, g in graphs.items():
            sim_results[phase] = self.sim_hub.simulate_graph(g, hw)

        # 6. 调度
        timelines = {}
        for phase, g in graphs.items():
            timelines[phase] = DAGScheduler().schedule(
                g, sim_results[phase], config.stream_config)

        # 7. 报表
        self.report.generate(
            config=config, profile=profile, hw=hw,
            graphs=graphs, sim_results=sim_results,
            timelines=timelines, memory_budget=mem_budget)

        return RunResult(
            feasible=True, memory_budget=mem_budget,
            graphs=graphs, sim_results=sim_results,
            timelines=timelines,
        )
```

### 4.2 配置寻优

```python
@dataclass
class SearchSpace:
    """搜索空间定义"""
    tp_choices: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    pp_choices: list[int] = field(default_factory=lambda: [1, 2, 4])
    ep_choices: list[int] = field(default_factory=lambda: [1])  # MoE 模型自动填充
    sp_choices: list[bool] = field(default_factory=lambda: [False, True])
    quant_choices: list[str | None] = field(default_factory=lambda: [None, "w8a8", "w4a16"])
    batch_sizes: list[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])

class ConfigSearchEngine:
    """
    搜索策略:
    1. 生成候选配置
    2. MemoryModel 快速剪枝（不可行的直接跳过）
    3. 对可行配置跑完整仿真
    4. 按目标函数排序
    """

    def __init__(self, orchestrator: Orchestrator, memory_model: MemoryModel):
        self.orch = orchestrator
        self.mem = memory_model

    def search(self,
               model_id: str,
               hardware: str,
               stack: str,
               objective: str = "throughput",    # "throughput" | "latency"
               space: SearchSpace | None = None,
               max_trials: int = 100,
               ) -> list["SearchResult"]:

        if space is None:
            space = self._default_space(model_id)

        # 1. 枚举候选
        candidates = list(self._enumerate(space))

        # 2. 显存剪枝（毫秒级，不需要图抓取和仿真）
        hw = self.orch.hw_reg.load(hardware)
        profile = ModelProfile.from_config(
            self.orch.model_mgr.load_config(model_id))
        feasible = []
        for cfg in candidates:
            budget = self.mem.estimate(profile, hw, cfg.parallel, cfg.quant,
                                       cfg.batch_size, cfg.seq_len)
            if budget.is_feasible:
                feasible.append((cfg, budget))

        # 3. 仿真（可能很多，限制 max_trials）
        feasible = feasible[:max_trials]
        results = []
        for cfg, budget in feasible:
            run_result = self.orch.run(cfg)
            score = self._score(run_result, objective)
            results.append(SearchResult(config=cfg, result=run_result, score=score))

        # 4. 排序
        results.sort(key=lambda r: r.score, reverse=True)
        return results
```

### 4.3 硬件对比

```python
class HardwareComparator:
    """固定模型+配置，对比多个硬件"""

    def compare(self, model_id: str,
                hardware_list: list[str],
                stack: str,
                parallel: ParallelConfig,
                batch_size: int = 1,
                seq_len: int = 4096,
                ) -> "ComparisonReport":

        results = {}
        for hw_name in hardware_list:
            cfg = RunConfig(
                model_id=model_id, hardware=hw_name, stack=stack,
                parallel=parallel, batch_size=batch_size, seq_len=seq_len,
            )
            results[hw_name] = self.orchestrator.run(cfg)

        return ComparisonReport(results)

@dataclass
class ComparisonReport:
    results: dict[str, RunResult]

    def to_table(self) -> list[dict]:
        """生成对比表格数据"""
        rows = []
        for hw, r in self.results.items():
            for phase, tl in r.timelines.items():
                rows.append({
                    "hardware": hw,
                    "phase": phase,
                    "latency_ms": tl.total_latency_us / 1000,
                    "memory_gb": r.memory_budget.total_mb / 1024,
                    "feasible": r.feasible,
                    ...
                })
        return rows
```

### 4.4 瓶颈分析

```python
class BottleneckAnalyzer:
    def analyze(self, graph: OpGraph, sim_results: dict[str, SimResult],
                timeline: Timeline) -> "BottleneckReport":
        overlap = OverlapAnalyzer().analyze(timeline)

        # 算子级 Top-N
        sorted_ops = sorted(sim_results.values(),
                           key=lambda r: r.latency_us, reverse=True)
        top_ops = sorted_ops[:20]

        # compute vs memory bound 统计
        compute_bound = [r for r in sim_results.values() if r.bound == "compute"]
        memory_bound = [r for r in sim_results.values() if r.bound == "memory"]

        # 模块级分解（利用 hierarchy）
        latency_map = {r.op_node_id: r.latency_us for r in sim_results.values()}
        module_breakdown = {}
        for module in graph.hierarchy.at_depth(3):
            total = graph.hierarchy.aggregate(module, "latency", latency_map)
            module_breakdown[module.scope] = total

        return BottleneckReport(
            top_ops=top_ops,
            overlap=overlap,
            compute_bound_pct=sum(r.latency_us for r in compute_bound) /
                              sum(r.latency_us for r in sim_results.values()),
            module_breakdown=module_breakdown,
        )
```

---

## 5. Extension Layer

### 5.1 Serving 仿真

建模连续批处理下的系统级吞吐和延迟分布。输入是单次推理的 latency（来自 Executor），输出是系统级指标。

```
zrt/
└── serving/
    ├── __init__.py
    ├── request_model.py     # 请求到达模型（泊松 / trace replay）
    ├── batch_scheduler.py   # 连续批处理仿真 (continuous batching)
    ├── pd_split.py          # Prefill-Decode 分离建模
    └── metrics.py           # TTFT 分布 / TPOT 分布 / 吞吐 / SLO 达标率
```

```python
class ServingSimulator:
    """
    离散事件仿真：
    - 请求按到达模型生成
    - scheduler 按策略组 batch（continuous batching）
    - 每个 batch 的 latency 从 Executor 结果查表/插值
    - 统计 TTFT/TPOT 分布和吞吐
    """

    def simulate(self,
                 prefill_latency_fn: Callable[[int, int], float],  # (bs, seq) → ms
                 decode_latency_fn: Callable[[int], float],         # (bs) → ms
                 arrival_rate: float,       # requests/s
                 avg_input_len: int,
                 avg_output_len: int,
                 max_batch_size: int,
                 duration_s: float = 60,
                 ) -> "ServingMetrics":
        ...

@dataclass
class ServingMetrics:
    throughput_tokens_per_sec: float
    avg_ttft_ms: float
    p99_ttft_ms: float
    avg_tpot_ms: float
    p99_tpot_ms: float
    slo_attainment: float            # % 请求满足 SLO
```

### 5.2 训练估算

训练不需要构建 backward 图——用公式从 forward 估算即可。

```
zrt/
└── training/
    ├── __init__.py
    ├── estimator.py         # TrainingEstimator: forward → iteration 耗时
    ├── memory.py            # 训练显存 (master weights, optimizer states, gradients)
    └── zero.py              # ZeRO 1/2/3 显存和通信建模
```

```python
class TrainingEstimator:
    """
    从推理仿真结果估算训练性能。

    核心公式：
    - backward FLOPs ≈ 2× forward FLOPs
    - backward latency ≈ 2× forward latency（经验值）
    - optimizer step ≈ 固定开销
    - 一次 iteration = forward + backward + optimizer + allreduce(gradients)
    """

    def estimate(self,
                 forward_result: RunResult,
                 grad_accumulation_steps: int = 1,
                 zero_stage: int = 0,
                 ) -> "TrainingEstimate":

        fwd_ms = forward_result.timelines["prefill"].total_latency_us / 1000
        bwd_ms = fwd_ms * 2.0   # 经验系数
        optimizer_ms = self._estimate_optimizer(forward_result.profile)

        # ZeRO 通信
        zero_comm_ms = ZeROModel().comm_per_step(
            forward_result.profile, forward_result.parallel,
            zero_stage, forward_result.hw_spec)

        iteration_ms = (fwd_ms + bwd_ms) * grad_accumulation_steps + \
                       optimizer_ms + zero_comm_ms

        return TrainingEstimate(
            forward_ms=fwd_ms, backward_ms=bwd_ms,
            optimizer_ms=optimizer_ms, zero_comm_ms=zero_comm_ms,
            iteration_ms=iteration_ms,
            samples_per_sec=forward_result.batch_size *
                           grad_accumulation_steps / (iteration_ms / 1000),
            mfu=self._compute_mfu(forward_result),
        )
```

### 5.3 校准模块

建立 "预测 → 实测 → 修正" 的闭环。

```
zrt/
└── calibration/
    ├── __init__.py
    ├── profiler_adapter.py  # 解析各平台 profiling 数据
    │                        # (nsys / msprof / pytorch profiler)
    ├── comparator.py        # 预测值 vs 实测值逐算子对比
    ├── report.py            # 精度报告 (MAPE / 分布图 / 逐算子误差)
    └── tuner.py             # 自动调参 (调整效率系数 / 拟合模型)
```

```python
class Calibrator:
    """
    校准工作流：
    1. 用户跑一次真机 profiling → 导出 trace 文件
    2. ProfilerAdapter 解析 trace → dict[op_key → measured_latency_us]
    3. Comparator 和仿真预测逐算子对比
    4. 输出精度报告 + 可选自动调参
    """

    def calibrate(self,
                  trace_file: str,            # nsys/msprof trace 文件路径
                  sim_results: dict[str, SimResult],
                  graph: OpGraph,
                  format: str = "auto",       # "nsys" | "msprof" | "pytorch" | "auto"
                  ) -> "CalibrationReport":

        measured = ProfilerAdapter.parse(trace_file, format)
        pairs = Comparator.align(sim_results, measured, graph)

        errors = []
        for pred, actual in pairs:
            error_pct = abs(pred.latency_us - actual) / actual * 100
            errors.append(OpError(op_id=pred.op_node_id,
                                  predicted=pred.latency_us,
                                  measured=actual,
                                  error_pct=error_pct))

        mape = sum(e.error_pct for e in errors) / len(errors)
        return CalibrationReport(mape=mape, errors=errors, ...)
```

---

## 6. 完整目录结构

```
zrt/
├── __init__.py
│
├── ir/                              # 计算图 IR
│   ├── __init__.py
│   ├── types.py                     #   DType, TensorMeta
│   ├── node.py                      #   OpNode
│   ├── edge.py                      #   Edge
│   ├── graph.py                     #   OpGraph
│   ├── hierarchy.py                 #   GraphHierarchy + HierNode
│   ├── utils.py                     #   拓扑排序, 子图提取
│   └── serde.py                     #   JSON 序列化
│
├── model/                           # 模型管理
│   ├── __init__.py
│   ├── loader.py                    #   FakeTensor 加载
│   ├── compat.py                    #   transformers shim
│   ├── patches.py                   #   monkey-patch
│   ├── registry.py                  #   本地注册表
│   └── profile.py                   #   ModelProfile
│
├── capture/                         # 图抓取
│   ├── __init__.py
│   ├── tracer.py                    #   capture() 入口
│   ├── dispatch.py                  #   aten 拦截
│   ├── tracker.py                   #   scope 追踪
│   ├── input_builder.py             #   阶段输入
│   └── graph_builder.py             #   records → OpGraph
│
├── memory/                          # 显存模型
│   ├── __init__.py
│   ├── model.py                     #   MemoryModel
│   ├── budget.py                    #   MemoryBudget
│   └── activation.py                #   激活生命周期分析
│
├── transform/                       # 图变换
│   ├── __init__.py
│   ├── pipeline.py                  #   4-stage Pipeline
│   ├── base.py                      #   GraphPass ABC
│   ├── context.py                   #   TransformContext
│   ├── parallel/                    #   Stage 1
│   │   ├── tensor_parallel.py
│   │   ├── expert_parallel.py
│   │   ├── sequence_parallel.py
│   │   ├── pipeline_parallel.py
│   │   └── comm_inserter.py
│   ├── fusion/                      #   Stage 2
│   │   ├── engine.py
│   │   └── patterns.py
│   ├── optim/                       #   Stage 3
│   │   ├── quantization.py
│   │   ├── eplb.py
│   │   ├── shared_expert.py
│   │   ├── mtp.py
│   │   ├── kv_cache.py
│   │   └── recompute.py
│   └── analysis/                    #   Stage 4
│       ├── flops.py
│       └── roofline.py
│
├── simulator/                       # 算子仿真
│   ├── __init__.py
│   ├── hub.py                       #   SimulatorHub
│   ├── base.py                      #   OpSimulator ABC
│   ├── result.py                    #   SimResult
│   ├── cache.py                     #   content-hash 缓存
│   └── backends/
│       ├── roofline.py              #   理论公式
│       ├── profile_db.py            #   真机查表
│       ├── regression.py            #   回归拟合
│       ├── tiling_sim.py            #   Tiling 仿真接口
│       └── custom.py                #   用户自定义
│
├── comm/                            # 通信模型
│   ├── __init__.py
│   ├── model.py                     #   CommModel
│   ├── algorithms.py                #   集合通信算法公式
│   └── topology.py                  #   拓扑路由
│
├── executor/                        # 图执行器
│   ├── __init__.py
│   ├── scheduler.py                 #   DAGScheduler
│   ├── stream.py                    #   Stream
│   ├── timeline.py                  #   Timeline
│   └── overlap.py                   #   OverlapAnalyzer
│
├── hardware/                        # 硬件配置
│   ├── __init__.py
│   ├── spec.py                      #   HardwareSpec
│   ├── registry.py                  #   Registry
│   └── configs/                     #   YAML 文件
│       ├── ascend_910b.yaml
│       ├── ascend_910c.yaml
│       ├── nvidia_a100_80g.yaml
│       ├── nvidia_h100_sxm.yaml
│       └── ...
│
├── stacks/                          # 软件栈
│   ├── __init__.py
│   ├── base.py                      #   SoftwareStack ABC
│   ├── registry.py
│   ├── mindie/
│   ├── vllm/
│   ├── tensorrt_llm/
│   └── generic/
│
├── report/                          # 报表
│   ├── __init__.py
│   ├── engine.py
│   ├── excel_writer.py
│   ├── html_writer.py
│   ├── onnx_writer.py
│   ├── json_writer.py
│   └── summary.py
│
├── app/                             # 应用层
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── config_search.py
│   ├── hw_compare.py
│   ├── bottleneck.py
│   └── cli.py
│
├── serving/                         # 扩展：Serving 仿真
│   ├── __init__.py
│   ├── request_model.py
│   ├── batch_scheduler.py
│   └── metrics.py
│
├── training/                        # 扩展：训练估算
│   ├── __init__.py
│   ├── estimator.py
│   ├── memory.py
│   └── zero.py
│
├── calibration/                     # 扩展：校准
│   ├── __init__.py
│   ├── profiler_adapter.py
│   ├── comparator.py
│   ├── report.py
│   └── tuner.py
│
└── api/                             # 可选：REST API
    ├── __init__.py
    ├── server.py
    └── routes/
        ├── simulate.py
        ├── trace.py
        └── predict.py
```

---

## 7. 配置体系

### 7.1 运行配置 YAML

```yaml
# configs/examples/dsv3_910b_tp8.yaml
model:
  id: "deepseek-ai/DeepSeek-V3-0324"
  num_layers: 0                      # 0 = 全量 61 层

hardware: "ascend_910b"
stack: "mindie"                      # 软件栈独立指定

phases: ["prefill", "decode"]

workload:
  batch_size: 1
  seq_len: 4096
  output_len: 512

parallel:
  tp: 8
  pp: 1
  ep: 8
  dp: 1
  sp: true

optimization:
  quantization:
    weight: "int8"                   # W8A8
    activation: "int8"
    kv_cache: "int8"
  flags:
    - "eplb"
    - "shared_expert_external"
    - "mtp"

executor:
  compute_streams: 1
  comm_streams: 1

simulator:
  backends: ["profile_db", "regression", "roofline"]
  profile_data_dir: "data/profiles/ascend_910b"

output:
  dir: "output/dsv3_910b_tp8ep8"
  formats: ["excel", "html", "onnx", "json"]
```

### 7.2 集群配置

```yaml
# configs/clusters/training_cluster_a.yaml
name: "Training Cluster A"
nodes:
  count: 16
  hardware: "ascend_910b"
  devices_per_node: 8
  intra_node:
    type: "HCCS"
    bandwidth_gbps: 392
  inter_node:
    type: "RoCE"
    bandwidth_gbps: 200
    latency_us: 5
```

---

## 8. CLI 设计

```bash
# ── 核心命令 ──

# 端到端性能预测（从 YAML）
zrt predict --config configs/examples/dsv3_910b_tp8.yaml

# 端到端性能预测（CLI 参数）
zrt predict deepseek-ai/DeepSeek-V3-0324 \
    --hw ascend_910b --stack mindie \
    --tp 8 --ep 8 --sp \
    --quant w8a8 \
    --batch-size 1 --seq-len 4096

# 配置寻优
zrt search deepseek-ai/DeepSeek-V3-0324 \
    --hw ascend_910b --stack mindie \
    --objective throughput \
    --max-trials 50

# 硬件对比
zrt compare deepseek-ai/DeepSeek-V3-0324 \
    --hw ascend_910b,nvidia_h100 \
    --stack mindie,vllm \
    --tp 8 --ep 8

# ── 独立子命令 ──

# 仅抓取图
zrt capture deepseek-ai/DeepSeek-V3-0324 \
    --layers 4 --phase prefill,decode \
    -o output/dsv3_graphs

# 独立算子仿真
zrt simulate matmul \
    --shape 4096,4096,4096 --dtype bf16 \
    --hw ascend_910b

# 显存估算（秒级，不需要图抓取）
zrt memory deepseek-ai/DeepSeek-V3-0324 \
    --hw ascend_910b --tp 8 --ep 8 --quant w8a8

# 校准
zrt calibrate \
    --trace trace.nsys \
    --sim-result output/sim_results.json \
    --graph output/graph.json

# API 服务
zrt serve --port 8080
```

---

## 9. 开发路线

每个 Phase 都是可交付的里程碑，Phase 1 完成后系统即可产出有价值的结果。

```
Phase 1 ── 单卡理论建模（~3 周）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
目标：单卡推理的算子级理论性能预测
交付：Excel + JSON + ONNX 报表

  ir/          → OpGraph + 层次化视图
  model/       → FakeTensor 加载（现有代码迁移）
  capture/     → 图抓取（现有代码迁移，输出改为 IR）
  hardware/    → YAML 加载 + HardwareSpec
  stacks/      → generic/ 基线栈（无融合）
  simulator/   → Roofline 后端
  executor/    → 单流调度（串行执行）
  memory/      → 公式估算
  report/      → Excel + JSON + ONNX
  app/cli.py   → `zrt predict` 基础版

验证点：DSV3 prefill/decode 在 910B 上的理论 latency
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


Phase 2 ── 融合 + 多卡并行（~4 周）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
目标：TP/EP 并行 + 平台融合 + 通算掩盖
交付：多卡性能预测 + 通算分析

  stacks/mindie/   → MindIE 融合规则 + 算子映射
  stacks/vllm/     → vLLM 融合规则
  transform/parallel/  → TP + EP + CommInserter
  transform/fusion/    → FusionEngine (现有三阶段引擎迁移)
  comm/            → CommModel (Ring/Tree AllReduce, A2A)
  executor/        → 多流调度 + OverlapAnalyzer
  report/html      → Timeline 可视化

验证点：DSV3 TP8-EP8 910B × 8 的 prefill/decode latency
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


Phase 3 ── 优化项 + 寻优（~3 周）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
目标：量化/EPLB/MTP/SP/PP + 配置寻优
交付：优化配置建议 + 硬件对比报表

  transform/optim/     → 量化 + EPLB + SharedExpert + MTP
  transform/parallel/  → SP + PP
  app/config_search    → 寻优引擎（显存剪枝 + 仿真评估）
  app/hw_compare       → 硬件对比
  app/bottleneck       → 瓶颈分析

验证点：寻优输出的最优配置 vs 人工调优结果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


Phase 4 ── 精度提升（~3 周）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
目标：接入真机数据，提升预测精度
交付：精度报告 + 校准工具

  simulator/backends/profile_db    → 真机查表
  simulator/backends/regression    → 回归拟合
  calibration/                     → 精度对比 + 自动调参

验证点：MAPE < 15%（相对真机 profiling）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


Phase 5 ── 扩展场景（按需）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  serving/       → Continuous batching 仿真
  training/      → 训练估算 + ZeRO
  api/           → REST API 服务化
  simulator/backends/tiling_sim  → Tiling 级仿真接口
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 10. 关键设计决策汇总

| # | 决策 | 理由 |
|---|------|------|
| 1 | 硬件和软件栈正交分离 | 避免 M×N 的笛卡尔爆炸；新增组合零代码 |
| 2 | 先切分再融合 | 通信边界天然成为融合边界；融合规则不感知并行 |
| 3 | 显存模型独立 | 寻优时毫秒级剪枝不可行配置，不需要跑完整仿真 |
| 4 | IR 原生层次化视图 | 算子/模块/层/阶段级分析无需重复遍历 |
| 5 | 通信模型独立+算法选择 | 不同数据量用不同算法，跨节点 vs 节点内差异大 |
| 6 | 仿真器 content-hash 缓存 | 避免手工拼 key 的遗漏和冲突 |
| 7 | 训练用公式估算不建图 | backward ≈ 2× forward 经验准确度够用，省复杂度 |
| 8 | 校准闭环 | 仿真系统价值 = 精度，必须有量化精度和调优手段 |
| 9 | Transform 4-stage 固定顺序 | 比声明式依赖系统简单可靠，满足当前需求 |
| 10 | 每个模块可独立使用 | MemoryModel/SimulatorHub/CommModel 可单独对外 |
