"""GraphPass: abstract base for all transform passes."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext


class GraphPass(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        """Pure transform: return a (possibly new) graph; do not mutate the input."""
        ...
