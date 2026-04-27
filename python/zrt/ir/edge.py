"""Edge: directed data-flow or control-flow edge in the computation graph."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .types import TensorMeta


@dataclass
class Edge:
    """A directed edge between two OpNodes.

    For data edges, ``tensor`` carries the TensorMeta of the flowing value
    and ``src_idx`` / ``dst_idx`` identify which output/input slot the tensor
    connects.  Control edges (dependency without data flow) have ``tensor=None``.

    ``tensor_id`` preserves the original integer ID assigned by TensorTracker
    during dispatch so that the raw record ↔ IR mapping is traceable.
    """
    src:     str            # source node id
    src_idx: int            # output slot on the source node
    dst:     str            # destination node id
    dst_idx: int            # input slot on the destination node
    tensor:  Optional[TensorMeta] = None   # None for control edges
    tensor_id: Optional[int] = None        # original dispatch tensor ID (int)

    @property
    def is_control(self) -> bool:
        return self.tensor is None

    @property
    def is_data(self) -> bool:
        return self.tensor is not None

    def __repr__(self) -> str:
        t = repr(self.tensor) if self.tensor else "ctrl"
        return f"Edge({self.src}[{self.src_idx}] → {self.dst}[{self.dst_idx}], {t})"
