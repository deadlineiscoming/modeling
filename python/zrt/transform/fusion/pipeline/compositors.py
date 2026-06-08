"""Post-fusion compositors."""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.ir.node import OpNode
    from python.zrt.transform.fusion.core.rule import ModuleFusionRule


_ADD_NORM_RULE_NAME = "add_norm"
_HC_POST_FFN_RULE_NAME = "hc_post_ffn"


def _compose_add_norm(graph: "OpGraph") -> "OpGraph":
    """Merge adjacent Add + RMSNorm/LayerNorm nodes into AddNorm.

    Operates on already-fused nodes (not raw aten ops).  Looks for
    ``op_type`` ending in "add" followed by one ending in "norm".
    """
    from python.zrt.ir.node import OpNode

    topo = graph.topo_sort()
    if len(topo) < 2:
        return graph

    norm_types = {"rms_norm", "layer_norm", "RMSNorm", "LayerNorm"}
    add_types = {"add", "residual_add"}

    pairs: list[tuple[int, int]] = []
    for i in range(len(topo) - 1):
        a, b = topo[i], topo[i + 1]
        a_label = a.op_type.lower()
        b_label = b.op_type.lower()
        if (a_label in add_types or a_label.endswith("add")) and \
           (b_label in norm_types or b_label.endswith("norm")):
            has_edge = any(
                e.src == a.id and e.dst == b.id for e in graph.edges
            )
            if has_edge:
                pairs.append((i, i + 1))

    if not pairs:
        return graph

    used: set[str] = set()
    for add_i, norm_i in reversed(pairs):
        add_node = topo[add_i]
        norm_node = topo[norm_i]
        if add_node.id in used or norm_node.id in used:
            continue

        merged = OpNode(
            id=f"composed_{add_node.id}_{norm_node.id}",
            op_type="AddNorm",
            inputs=add_node.inputs,
            outputs=norm_node.outputs,
            scope=norm_node.scope,
            category="compute",
            module_class=norm_node.module_class,
            layer=norm_node.layer,
            component=norm_node.component,
            fused_from=add_node.fused_from + norm_node.fused_from,
            num_sub_ops=add_node.num_sub_ops + norm_node.num_sub_ops,
            fusion_level="parent",
            name=norm_node.name,
        )
        merged.annotations.update(add_node.annotations)
        merged.annotations.update(norm_node.annotations)
        merged.annotations["source_op_ids"] = (
            list(add_node.annotations.get("source_op_ids", [add_node.id]))
            + list(norm_node.annotations.get("source_op_ids", [norm_node.id]))
        )
        merged.annotations["fused_by_rule"] = _ADD_NORM_RULE_NAME
        graph.replace_subgraph({add_node.id, norm_node.id}, merged)
        used.add(add_node.id)
        used.add(norm_node.id)

    return graph


def _compose_hc_post_ffn(
    graph: "OpGraph",
    rule: "ModuleFusionRule",
) -> "OpGraph":
    """Fuse the remaining HCPostFfn diamond left by non-contiguous topo order.

    TP/EP traces can expose ``HCPostFfn`` as two same-scope fragments:
    one branch computes ``mul -> sum`` and another computes ``mul`` before
    both feed the final ``add``.  The generic bucket fuser intentionally
    keeps non-contiguous topo fragments separate, so handle this exact
    four-node diamond here without changing global bucketing semantics.
    """
    from python.zrt.transform.fusion.bucketing.call_id_bucketer import FusionGroup
    from python.zrt.transform.fusion.building.node_builder import build_fused_node

    buckets: dict[tuple[str, str, str], list[OpNode]] = defaultdict(list)
    for node in graph.topo_sort():
        if node.category == "communication":
            continue
        if node.module_class != "HCPostFfn":
            continue
        if node.op_type not in {
            "aten.mul.Tensor",
            "aten.sum.dim_IntList",
            "aten.add.Tensor",
        }:
            continue
        phase = node.annotations.get("phase")
        if phase and phase != "fwd":
            continue
        buckets[(node.scope, node.layer, str(phase or ""))].append(node)

    fuse_idx = _next_fuse_idx(graph)
    for nodes in buckets.values():
        ordered = _ordered_hc_post_ffn_diamond(graph, nodes)
        if ordered is None:
            continue

        group = FusionGroup(
            scope=ordered[0].scope,
            module_class=ordered[0].module_class,
            module_class_obj=None,
            child_ops=ordered,
            leaf_attr=ordered[0].name,
            call_id=getattr(ordered[0], "call_id", 0) or 0,
            is_full_forward=True,
        )
        replacement = build_fused_node(group, rule, graph, fuse_idx)
        if group.call_id:
            replacement.call_id = group.call_id
        graph.replace_subgraph({node.id for node in ordered}, replacement)
        fuse_idx += 1

    return graph


def _ordered_hc_post_ffn_diamond(
    graph: "OpGraph",
    nodes: list["OpNode"],
) -> list["OpNode"] | None:
    if len(nodes) != 4:
        return None

    muls = [n for n in nodes if n.op_type == "aten.mul.Tensor"]
    sums = [n for n in nodes if n.op_type == "aten.sum.dim_IntList"]
    adds = [n for n in nodes if n.op_type == "aten.add.Tensor"]
    if len(muls) != 2 or len(sums) != 1 or len(adds) != 1:
        return None

    sum_node = sums[0]
    add_node = adds[0]
    edges = {(e.src, e.dst) for e in graph.edges}
    if (sum_node.id, add_node.id) not in edges:
        return None

    mix_mul = next((m for m in muls if (m.id, sum_node.id) in edges), None)
    if mix_mul is None:
        return None

    residual_mul = next(
        (m for m in muls if m.id != mix_mul.id and (m.id, add_node.id) in edges),
        None,
    )
    if residual_mul is None:
        return None

    return [residual_mul, mix_mul, sum_node, add_node]


def _next_fuse_idx(graph: "OpGraph") -> int:
    idx = len(graph.nodes)
    while any(node_id.startswith(f"fused_{idx}_") for node_id in graph.nodes):
        idx += 1
    return idx
