"""Content-hash cache for SimResult objects.

The cache key is derived from (op_type, input shapes+dtypes, attrs, hw.name)
so that semantically identical ops on the same hardware always hit the cache,
regardless of node ID or position in the graph.
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from .result import SimResult

if TYPE_CHECKING:
    from python.zrt.ir.node import OpNode
    from python.zrt.hardware.spec import HardwareSpec


def content_hash(node: "OpNode", hw: "HardwareSpec") -> str:
    """Stable MD5 hex-digest for (node semantics, hw)."""
    parts: list[str] = [node.op_type, hw.name]

    for t in node.inputs:
        parts.append(str(t.shape))
        parts.append(t.dtype.value)

    # Sort attrs for stability (dict order varies across Python versions)
    try:
        attrs_repr = str(sorted(node.attrs.items()))
    except TypeError:
        attrs_repr = str(node.attrs)
    parts.append(attrs_repr)

    # For fused nodes, include constituent op list for correctness
    if node.fused_from:
        parts.append(",".join(node.fused_from))

    digest = hashlib.md5("|".join(parts).encode(), usedforsecurity=False).hexdigest()
    return digest


class SimCache:
    """Simple dict-backed cache keyed by content hash."""

    def __init__(self) -> None:
        self._store: dict[str, SimResult] = {}
        self.hits = 0
        self.misses = 0

    def get(self, node: "OpNode", hw: "HardwareSpec") -> SimResult | None:
        key = content_hash(node, hw)
        result = self._store.get(key)
        if result is None:
            self.misses += 1
        else:
            self.hits += 1
        return result

    def put(self, node: "OpNode", hw: "HardwareSpec", result: SimResult) -> None:
        key = content_hash(node, hw)
        self._store[key] = result

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._store)
