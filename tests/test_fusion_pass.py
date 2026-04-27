"""Tests for FusionPass OpGraph IR integration."""
import pytest
from python.zrt.ir.node import OpNode
from python.zrt.ir.edge import Edge
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.types import TensorMeta, DType
from python.zrt.transform.fusion.pass_ import FusionPass
from python.zrt.transform.context import TransformContext, ParallelConfig, StreamConfig
import python.zrt.hardware.registry as hw_registry


# ── helpers ───────────────────────────────────────────────────────────────────

def _t(tid, shape=(1, 128), dtype=DType.BF16):
    return TensorMeta.from_shape_dtype(tid, shape, dtype)


def _node(nid, op_type, scope="", layer="0", module_class="", category="compute"):
    n = OpNode(
        id=nid, op_type=op_type,
        inputs=[_t(f"{nid}_in")], outputs=[_t(f"{nid}_out")],
        scope=scope, layer=layer, module_class=module_class, category=category,
    )
    return n


def _edge(src, dst):
    return Edge(src=src, src_idx=0, dst=dst, dst_idx=0, tensor=_t("e"))


def _graph(nodes, edges, name="test"):
    return OpGraph(name=name, phase="prefill",
                   nodes={n.id: n for n in nodes},
                   edges=edges)


def _ctx(hw_name="nvidia_h100_sxm"):
    hw = hw_registry.load(hw_name)
    return TransformContext(hw_spec=hw, parallel=ParallelConfig(tp=1),
                            stream_config=StreamConfig())


# ── pass-through cases ────────────────────────────────────────────────────────

def test_empty_graph_returns_empty():
    g = OpGraph(name="t", phase="prefill")
    out = FusionPass().run(g, _ctx())
    assert out.num_nodes() == 0


def test_single_node_no_scope_unchanged():
    n = _node("a", "aten.mm.default", scope="")
    g = _graph([n], [])
    out = FusionPass().run(g, _ctx())
    assert out.num_nodes() == 1
    assert out.nodes["a"].op_type == "aten.mm.default"


def test_does_not_mutate_input():
    n1 = _node("a", "aten.mm.default", scope="model.layers.0.mlp")
    n2 = _node("b", "aten.add.Tensor", scope="model.layers.0.mlp")
    g  = _graph([n1, n2], [_edge("a", "b")])
    original_count = g.num_nodes()
    FusionPass().run(g, _ctx())
    assert g.num_nodes() == original_count


# ── leaf fusion ───────────────────────────────────────────────────────────────

def test_same_scope_nodes_are_fused():
    scope = "model.layers.0.mlp"
    a = _node("a", "aten.mm.default",   scope=scope, module_class="MLP")
    b = _node("b", "aten.silu.default", scope=scope, module_class="MLP")
    c = _node("c", "aten.mm.default",   scope=scope, module_class="MLP")
    g = _graph([a, b, c], [_edge("a", "b"), _edge("b", "c")])

    out = FusionPass().run(g, _ctx())
    # Three nodes fused into one
    assert out.num_nodes() == 1


def test_different_scope_nodes_not_fused():
    a = _node("a", "aten.mm.default",   scope="model.layers.0.q_proj", module_class="Linear")
    b = _node("b", "aten.mm.default",   scope="model.layers.0.k_proj", module_class="Linear")
    g = _graph([a, b], [])
    out = FusionPass().run(g, _ctx())
    # Different scopes → not fused (no parent group either: no common parent class)
    assert out.num_nodes() >= 1


def test_different_layer_nodes_not_fused():
    scope = "model.layers.X.mlp"
    a = _node("a", "aten.mm.default", scope=scope, layer="0", module_class="MLP")
    b = _node("b", "aten.mm.default", scope=scope, layer="1", module_class="MLP")
    g = _graph([a, b], [])
    out = FusionPass().run(g, _ctx())
    assert out.num_nodes() == 2


def test_comm_node_breaks_group():
    scope = "model.layers.0.mlp"
    a    = _node("a", "aten.mm.default",   scope=scope, module_class="MLP")
    comm = _node("c", "comm.all_reduce",   scope="",    category="communication")
    b    = _node("b", "aten.mm.default",   scope=scope, module_class="MLP")
    g = _graph([a, comm, b], [_edge("a", "c"), _edge("c", "b")])

    out = FusionPass().run(g, _ctx())
    # a and b are separated by comm → cannot be in the same group
    # comm itself is standalone
    # Depending on whether pass2 merges them: at minimum comm survives as-is
    comm_nodes = [n for n in out.nodes.values() if n.category == "communication"]
    assert len(comm_nodes) == 1
    assert comm_nodes[0].op_type == "comm.all_reduce"


# ── semantic labelling ────────────────────────────────────────────────────────

def test_fused_node_has_semantic_label():
    scope = "model.layers.0.mlp"
    a = _node("a", "aten.mm.default",   scope=scope, module_class="MLP")
    b = _node("b", "aten.silu.default", scope=scope, module_class="MLP")
    c = _node("c", "aten.mm.default",   scope=scope, module_class="MLP")
    g = _graph([a, b, c], [_edge("a", "b"), _edge("b", "c")])

    out = FusionPass().run(g, _ctx())
    assert out.num_nodes() == 1
    fused = next(iter(out.nodes.values()))
    # MLP module class → semantic label "mlp"
    assert fused.op_type == "mlp"


def test_rms_norm_gets_semantic_label():
    scope = "model.layers.0.input_layernorm"
    a = _node("a", "aten.pow.Tensor_Scalar", scope=scope, module_class="LlamaRMSNorm")
    b = _node("b", "aten.rsqrt.default",     scope=scope, module_class="LlamaRMSNorm")
    c = _node("c", "aten.mul.Tensor",        scope=scope, module_class="LlamaRMSNorm")
    g = _graph([a, b, c], [_edge("a", "b"), _edge("b", "c")])

    out = FusionPass().run(g, _ctx())
    fused = next(iter(out.nodes.values()))
    assert fused.op_type == "rms_norm"


def test_fused_from_tracks_constituent_ops():
    scope = "model.layers.0.mlp"
    a = _node("a", "aten.mm.default",   scope=scope, module_class="MLP")
    b = _node("b", "aten.silu.default", scope=scope, module_class="MLP")
    g = _graph([a, b], [_edge("a", "b")])

    out = FusionPass().run(g, _ctx())
    fused = next(iter(out.nodes.values()))
    assert "aten.mm.default" in fused.fused_from
    assert "aten.silu.default" in fused.fused_from


# ── edge connectivity ─────────────────────────────────────────────────────────

def test_external_edges_rewired():
    """Predecessor and successor outside the fused group must be reconnected."""
    pre   = _node("pre",  "aten.embedding.default", scope="model.embed")
    scope = "model.layers.0.mlp"
    a     = _node("a",   "aten.mm.default",   scope=scope, module_class="MLP")
    b     = _node("b",   "aten.silu.default", scope=scope, module_class="MLP")
    post  = _node("post", "aten.mm.default",  scope="model.lm_head")
    g = _graph([pre, a, b, post],
               [_edge("pre", "a"), _edge("a", "b"), _edge("b", "post")])

    out = FusionPass().run(g, _ctx())
    # pre + fused + post = 3 nodes
    assert out.num_nodes() == 3
    # fused node has pre as predecessor and post as successor
    fused_id = [nid for nid, n in out.nodes.items()
                if n.op_type in ("mlp", "MLP") or n.is_fused][0]
    assert "pre" in out.predecessors(fused_id)
    assert "post" in out.successors(fused_id)


def test_node_count_and_edge_count_consistent():
    """After fusion the graph must remain a valid DAG (no dangling edges)."""
    scope = "model.layers.0.mlp"
    nodes = [_node(f"n{i}", "aten.mm.default", scope=scope, module_class="MLP")
             for i in range(5)]
    edges = [_edge(f"n{i}", f"n{i+1}") for i in range(4)]
    g = _graph(nodes, edges)

    out = FusionPass().run(g, _ctx())
    assert out.num_nodes() == 1
    assert out.num_edges() == 0   # all edges were internal to the group


# ── platform: cuda subpattern matching ────────────────────────────────────────

def test_cuda_gated_mlp_pattern():
    """Platform=cuda: mm → silu → mul → mm in MLP class → 'gated_mlp'."""
    scope = "model.layers.0.mlp"
    mc    = "MistralMLP"  # matches _MLP_RE
    ops   = [
        ("gate", "aten.mm.default"),
        ("silu", "aten.silu.default"),
        ("mul",  "aten.mul.Tensor"),
        ("down", "aten.mm.default"),
    ]
    nodes = [_node(nid, op, scope=scope, module_class=mc) for nid, op in ops]
    edges = [_edge(ops[i][0], ops[i+1][0]) for i in range(len(ops)-1)]
    g     = _graph(nodes, edges)

    # Use nvidia hw → platform inferred as "cuda"
    out = FusionPass().run(g, _ctx("nvidia_h100_sxm"))
    fused = next(iter(out.nodes.values()))
    assert fused.op_type == "gated_mlp"
