"""OpSimulator abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .result import SimResult

if TYPE_CHECKING:
    from python.zrt.ir.node import OpNode
    from python.zrt.hardware.spec import HardwareSpec


class OpSimulator(ABC):
    """Abstract interface for a single-operator latency estimator.

    Backends are registered with a ``priority`` value; ``SimulatorHub`` picks
    the highest-priority backend that ``can_simulate()`` returns True for.
    The Roofline backend has priority=0 and always returns True, acting as
    the universal fallback.
    """

    #: Unique backend name (set as class attribute in subclasses)
    name: str = ""

    #: Selection priority — higher wins.  Roofline=0, ProfileDB=30, etc.
    priority: int = 0

    @abstractmethod
    def can_simulate(self, node: "OpNode", hw: "HardwareSpec") -> bool:
        """Return True if this backend can handle *node* on *hw*."""
        ...

    @abstractmethod
    def simulate(self, node: "OpNode", hw: "HardwareSpec") -> SimResult:
        """Estimate latency and resource usage for *node* on *hw*."""
        ...
