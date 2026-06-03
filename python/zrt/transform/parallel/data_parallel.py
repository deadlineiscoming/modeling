"""DataParallelPass: gradient reduction communication insertion."""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from python.zrt.ir.edge import Edge
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.transform.base import GraphPass
from python.zrt.transform.context import TransformContext

logger = logging.getLogger(__name__)

_BWD_PHASES = {"bwd", "backward", "train_backward"}
_DEFAULT_BUCKET_CAP_MB = 25.0


@dataclass
class _GradEntry:
    node: OpNode
    size_bytes: int
    layer: str
    source_id: str


class DataParallelPass(GraphPass):
    """Insert gradient reduction nodes for data parallelism.

    Default behavior is the original layer-granularity DP path. It rewires the
    graph so later backward compute waits for each layer reduction; summary
    hidden/exposed accounting is controlled separately by
    ``training.dp_overlap_in_bubble``. When ``training.dp_bucket_mode == "ddp"``,
    gradients are collected in backward-ready topological order and accumulated
    into buckets capped by ``training.dp_bucket_cap_mb``.
    """

    name = "data_parallel"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        if not ctx.is_training or ctx.parallel.dp <= 1:
            return graph

        g = graph.clone()
        dp = ctx.parallel.dp
        zero_stage = ctx.training.zero_stage if ctx.training else 0

        if zero_stage >= 3:
            return g

        graph_phase = g.metadata.get("phase", "")
        if graph_phase and graph_phase not in ("train_backward", "backward"):
            return g

        entries = self._find_grad_entries(g)
        if not entries:
            return g

        collective = "all_reduce" if zero_stage == 0 else "reduce_scatter"
        dp_overlap = getattr(ctx.training, "dp_overlap_in_bubble", True) if ctx.training else True
        bucket_mode = getattr(ctx.training, "dp_bucket_mode", "layer") if ctx.training else "layer"
        if bucket_mode == "ddp":
            self._insert_ddp_buckets(g, entries, ctx, dp, collective, dp_overlap)
        else:
            self._insert_layer_buckets(g, entries, dp, collective, dp_overlap)

        return g

    def _insert_ddp_buckets(
        self,
        graph: OpGraph,
        entries: list[_GradEntry],
        ctx: TransformContext,
        dp: int,
        collective: str,
        dp_overlap: bool,
    ) -> None:
        cap_mb = getattr(ctx.training, "dp_bucket_cap_mb", _DEFAULT_BUCKET_CAP_MB) if ctx.training else _DEFAULT_BUCKET_CAP_MB
        cap_bytes = max(1, int(cap_mb * 1024 * 1024))
        pp = ctx.parallel.pp if ctx.parallel else 1
        buckets = self._build_buckets(entries, cap_bytes, split_by_layer=pp > 1)

        for bucket_idx, bucket in enumerate(buckets):
            last_node = bucket[-1].node
            bucket_bytes = sum(entry.size_bytes for entry in bucket)
            layers = sorted({entry.layer for entry in bucket}, key=self._layer_sort_key)
            owner_layer = bucket[-1].layer if pp <= 1 else layers[0]

            comm_node = OpNode(
                id=f"comm_grad_reduce_bucket_{bucket_idx}",
                op_type=f"comm.{collective}",
                inputs=[],
                outputs=[],
                attrs={
                    "group_size": dp,
                    "collective": collective,
                    "role": "dp_grad_reduce",
                    "bucket_bytes": bucket_bytes,
                    "bucket_cap_mb": cap_mb,
                    "bucket_index": bucket_idx,
                    "bucket_param_count": len(bucket),
                    "bucket_ready_node": last_node.id,
                    "bucket_layers": layers,
                    "bucket_source_ids": [entry.source_id for entry in bucket],
                },
                scope=f"data_parallel.grad_reduce.bucket_{bucket_idx}",
                layer=owner_layer if pp > 1 else "",
                category="communication",
            )
            comm_node.annotations["inserted_by"] = "data_parallel_pass"
            comm_node.annotations["dp_comm"] = True
            comm_node.annotations["phase"] = "bwd"
            if dp_overlap:
                comm_node.annotations["overlap_in_bubble"] = True

            self._add_side_branch(graph, last_node, comm_node)

            scale_node = OpNode(
                id=f"grad_scale_bucket_{bucket_idx}",
                op_type="aten.div.Scalar",
                attrs={"divisor": dp, "role": "dp_grad_average", "bucket_index": bucket_idx},
                layer=owner_layer if pp > 1 else "",
                category="communication",
            )
            scale_node.annotations["inserted_by"] = "data_parallel_pass"
            scale_node.annotations["phase"] = "bwd"
            scale_node.annotations["dp_comm_postprocess"] = True
            self._add_side_branch(graph, comm_node, scale_node)

    def _insert_layer_buckets(
        self,
        graph: OpGraph,
        entries: list[_GradEntry],
        dp: int,
        collective: str,
        dp_overlap: bool,
    ) -> None:
        layer_groups: dict[str, list[_GradEntry]] = defaultdict(list)
        for entry in entries:
            layer_groups[entry.layer].append(entry)

        for group_idx, (layer_key, bucket) in enumerate(
            sorted(layer_groups.items(), key=lambda kv: self._layer_sort_key(kv[0]))
        ):
            last_node = bucket[-1].node
            bucket_bytes = sum(entry.size_bytes for entry in bucket)

            comm_node = OpNode(
                id=f"comm_grad_reduce_layer_{layer_key}",
                op_type=f"comm.{collective}",
                inputs=[],
                outputs=[],
                attrs={
                    "group_size": dp,
                    "collective": collective,
                    "role": "dp_grad_reduce",
                    "bucket_bytes": bucket_bytes,
                    "dp_grad_group_idx": group_idx,
                    "bucket_ready_node": last_node.id,
                    "bucket_layers": [layer_key],
                },
                scope=f"data_parallel.grad_reduce.layer_{layer_key}",
                category="communication",
            )
            comm_node.annotations["inserted_by"] = "data_parallel_pass"
            comm_node.annotations["dp_comm"] = True
            comm_node.annotations["phase"] = "bwd"
            if dp_overlap:
                comm_node.annotations["overlap_in_bubble"] = True
            else:
                comm_node.annotations["blocking_comm"] = True

            self._insert_after(graph, last_node, comm_node)

            scale_node = OpNode(
                id=f"grad_scale_layer_{layer_key}",
                op_type="aten.div.Scalar",
                attrs={"divisor": dp, "role": "dp_grad_average"},
                category="compute",
            )
            scale_node.annotations["inserted_by"] = "data_parallel_pass"
            scale_node.annotations["phase"] = "bwd"
            self._insert_after(graph, comm_node, scale_node)

    def _find_grad_entries(self, graph: OpGraph) -> list[_GradEntry]:
        topo_index = {node.id: idx for idx, node in enumerate(graph.topo_sort())}
        param_nodes = {
            nid
            for nid, n in graph.nodes.items()
            if n.annotations.get("is_param") and n.annotations.get("phase") == "fwd"
        }

        if param_nodes:
            entries: list[_GradEntry] = []
            for edge in graph.edges:
                if edge.src not in param_nodes or not edge.dst.startswith("bwd_"):
                    continue
                bwd_node = graph.nodes.get(edge.dst)
                if bwd_node is None:
                    continue
                size_bytes = self._edge_or_node_bytes(edge, graph.nodes[edge.src])
                if size_bytes <= 0:
                    continue
                entries.append(_GradEntry(
                    node=bwd_node,
                    size_bytes=size_bytes,
                    layer=bwd_node.layer if bwd_node.layer else "0",
                    source_id=edge.src,
                ))
            return sorted(entries, key=lambda entry: topo_index.get(entry.node.id, len(topo_index)))

        entries = []
        for node in graph.topo_sort():
            if not self._is_backward_node(node):
                continue
            size_bytes = self._node_output_bytes(node)
            if size_bytes <= 0:
                continue
            entries.append(_GradEntry(
                node=node,
                size_bytes=size_bytes,
                layer=node.layer if node.layer else "0",
                source_id=node.id,
            ))
        return entries

    @staticmethod
    def _build_buckets(
        entries: list[_GradEntry],
        cap_bytes: int,
        *,
        split_by_layer: bool = False,
    ) -> list[list[_GradEntry]]:
        buckets: list[list[_GradEntry]] = []
        current: list[_GradEntry] = []
        current_bytes = 0
        current_layer: str | None = None

        for entry in entries:
            if split_by_layer and current and entry.layer != current_layer:
                buckets.append(current)
                current = []
                current_bytes = 0

            current.append(entry)
            current_bytes += entry.size_bytes
            current_layer = entry.layer
            if current_bytes >= cap_bytes:
                buckets.append(current)
                current = []
                current_bytes = 0
                current_layer = None

        if current:
            buckets.append(current)
        return buckets

    @staticmethod
    def _edge_or_node_bytes(edge: Edge, node: OpNode) -> int:
        if edge.tensor is not None and hasattr(edge.tensor, "mem_bytes"):
            return int(edge.tensor.mem_bytes)
        return DataParallelPass._node_output_bytes(node)

    @staticmethod
    def _node_output_bytes(node: OpNode) -> int:
        return sum(int(o.mem_bytes) for o in node.outputs if hasattr(o, "mem_bytes"))

    def _is_backward_node(self, node: OpNode) -> bool:
        """Check if a node belongs to the backward phase."""
        phase = node.annotations.get("phase", "")
        if phase in _BWD_PHASES:
            return True

        op_lower = node.op_type.lower()
        if "grad" in op_lower or "backward" in op_lower:
            return True

        return "grad" in node.scope.lower()

    def _add_side_branch(self, graph: OpGraph, src_node: OpNode, dst_node: OpNode) -> None:
        """Add dst_node as an extra dependency branch after src_node."""
        tensor = src_node.outputs[0] if src_node.outputs else None
        graph.nodes[dst_node.id] = dst_node
        graph.edges.append(Edge(
            src=src_node.id,
            src_idx=0,
            dst=dst_node.id,
            dst_idx=0,
            tensor=tensor,
        ))
        graph._rebuild_adjacency()

    def _insert_after(self, graph: OpGraph, src_node: OpNode, dst_node: OpNode) -> None:
        """Insert dst_node between src_node and all current successors.

        This preserves pure DP ordering by making later compute wait for the
        reduction and scaling nodes. DDP cap-bucket mode deliberately uses
        _add_side_branch instead.
        """
        src_id = src_node.id
        old_out = [e for e in graph.edges if e.src == src_id]

        graph.edges = [e for e in graph.edges if e.src != src_id]
        graph.nodes[dst_node.id] = dst_node

        if src_node.outputs:
            for i, out_tensor in enumerate(src_node.outputs):
                graph.edges.append(Edge(
                    src=src_id,
                    src_idx=i,
                    dst=dst_node.id,
                    dst_idx=i,
                    tensor=out_tensor,
                ))
        elif old_out:
            graph.edges.append(Edge(
                src=src_id,
                src_idx=0,
                dst=dst_node.id,
                dst_idx=0,
                tensor=old_out[0].tensor,
            ))
        else:
            graph.edges.append(Edge(
                src=src_id,
                src_idx=0,
                dst=dst_node.id,
                dst_idx=0,
                tensor=None,
            ))

        for e in old_out:
            graph.edges.append(Edge(
                src=dst_node.id,
                src_idx=e.src_idx,
                dst=e.dst,
                dst_idx=e.dst_idx,
                tensor=e.tensor,
            ))

        graph._rebuild_adjacency()

    @staticmethod
    def _layer_sort_key(layer: str) -> tuple[int, str]:
        return (int(layer), layer) if layer.isdigit() else (10**9, layer)
