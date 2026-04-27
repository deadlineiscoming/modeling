"""GraphHierarchy: scope-tree view of an OpGraph for multi-granularity analysis.

Builds a tree from the ``scope`` (module_path) strings of OpNodes.
Supports:
  - at_depth(d)   — list all HierNodes at a given depth
  - find(pattern) — glob-match scopes
  - aggregate()   — recursively sum a per-leaf metric over a subtree
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .graph import OpGraph


# ─────────────────────────────────────────────────────────────────────────────
# HierNode
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HierNode:
    """One node in the module hierarchy tree.

    ``scope``         — full dotted path, e.g. "model.layers.0.self_attn"
    ``name``          — last segment, e.g. "self_attn"
    ``children``      — child HierNodes (ordered by first-appearance)
    ``leaf_node_ids`` — OpNode IDs that belong directly to this scope
    ``depth``         — 0 = root (whole graph), 1 = top-level modules, ...
    """
    scope:         str
    name:          str
    depth:         int
    children:      list["HierNode"]     = field(default_factory=list)
    leaf_node_ids: list[str]            = field(default_factory=list)
    _metrics:      dict[str, float]     = field(default_factory=dict, repr=False)

    def all_leaf_ids(self) -> list[str]:
        """Recursively collect all OpNode IDs under this subtree."""
        ids = list(self.leaf_node_ids)
        for c in self.children:
            ids.extend(c.all_leaf_ids())
        return ids

    def __repr__(self) -> str:
        return (
            f"HierNode('{self.scope}', depth={self.depth}, "
            f"children={len(self.children)}, ops={len(self.leaf_node_ids)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# GraphHierarchy
# ─────────────────────────────────────────────────────────────────────────────

class GraphHierarchy:
    """Scope-tree built from an OpGraph's node scopes.

    Construction is O(N) where N = number of OpNodes.

    Depth convention
    ----------------
    depth 0  → root (entire graph, one node)
    depth 1  → top-level keys: "model", "embed_tokens", ...
    depth 2  → "model.layers", "model.norm", ...
    depth 3  → "model.layers.0", "model.layers.1", ...
    depth 4  → "model.layers.0.self_attn", "model.layers.0.mlp", ...
    """

    def __init__(self, graph: "OpGraph") -> None:
        self._graph = graph
        self.root = HierNode(scope="", name="<root>", depth=0)
        self._scope_map: dict[str, HierNode] = {"": self.root}
        self._build(graph)

    # ── construction ─────────────────────────────────────────────────────────

    def _build(self, graph: "OpGraph") -> None:
        for node in graph.nodes.values():
            scope = node.scope or ""
            hier_node = self._get_or_create(scope)
            hier_node.leaf_node_ids.append(node.id)

    def _get_or_create(self, scope: str) -> HierNode:
        if scope in self._scope_map:
            return self._scope_map[scope]
        # ensure parent exists first
        if "." in scope:
            parent_scope = scope.rsplit(".", 1)[0]
        else:
            parent_scope = ""
        parent = self._get_or_create(parent_scope)
        name = scope.rsplit(".", 1)[-1] if scope else scope
        node = HierNode(scope=scope, name=name, depth=parent.depth + 1)
        parent.children.append(node)
        self._scope_map[scope] = node
        return node

    # ── queries ───────────────────────────────────────────────────────────────

    def at_depth(self, depth: int) -> list[HierNode]:
        """Return all HierNodes at exactly ``depth``.

        depth=0 → [root]
        depth=1 → top-level module nodes
        """
        results: list[HierNode] = []
        self._collect_depth(self.root, depth, results)
        return results

    def _collect_depth(self, node: HierNode, target: int,
                       out: list[HierNode]) -> None:
        if node.depth == target:
            out.append(node)
            return
        for c in node.children:
            self._collect_depth(c, target, out)

    def find(self, scope_pattern: str) -> list[HierNode]:
        """Return HierNodes whose scope matches ``scope_pattern`` (glob syntax).

        Examples::

            hier.find("model.layers.*.self_attn")
            hier.find("model.layers.*")
        """
        return [
            node for scope, node in self._scope_map.items()
            if scope and fnmatch.fnmatch(scope, scope_pattern)
        ]

    def get(self, scope: str) -> HierNode | None:
        """Return the HierNode for an exact scope, or None."""
        return self._scope_map.get(scope)

    def aggregate(self, node: HierNode, values: dict[str, float]) -> float:
        """Recursively sum ``values[op_node_id]`` over all leaves in ``node``'s subtree.

        ``values`` is a dict mapping OpNode ID → scalar (e.g. latency_us, flops).
        Missing keys are treated as 0.
        """
        if node.leaf_node_ids and not node.children:
            # leaf HierNode: sum its own op IDs
            return sum(values.get(nid, 0.0) for nid in node.leaf_node_ids)

        # internal node: sum children + own direct ops (if any)
        total = sum(values.get(nid, 0.0) for nid in node.leaf_node_ids)
        for c in node.children:
            total += self.aggregate(c, values)
        return total

    def module_breakdown(self, values: dict[str, float],
                         depth: int = 4) -> dict[str, float]:
        """Return {scope: aggregated_value} for all HierNodes at ``depth``.

        Useful for generating module-level latency/flops tables in reports.
        """
        return {
            node.scope: self.aggregate(node, values)
            for node in self.at_depth(depth)
        }

    # ── stats ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"GraphHierarchy(graph='{self._graph.name}', "
            f"scopes={len(self._scope_map)})"
        )
