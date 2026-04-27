"""zrt.simulator — Operator latency estimation.

Quickstart::

    from python.zrt.simulator import SimulatorHub, SimResult
    from python.zrt.hardware import load as load_hw

    hw  = load_hw("ascend_910b")
    hub = SimulatorHub.default()   # Roofline backend pre-registered

    # Single node
    result: SimResult = hub.simulate(some_op_node, hw)
    print(result.latency_us, result.bound, result.hw_utilization)

    # Whole graph
    results: dict[str, SimResult] = hub.simulate_graph(op_graph, hw)
    total_us = sum(r.latency_us for r in results.values())

Extend with a custom backend::

    from python.zrt.simulator import OpSimulator, SimResult

    class MyBackend(OpSimulator):
        name = "my_backend"
        priority = 50   # higher than roofline (0)

        def can_simulate(self, node, hw):
            return node.op_type == "aten.mm.default"

        def simulate(self, node, hw):
            ...
            return SimResult(...)

    hub.register(MyBackend())
"""

from .result import SimResult
from .base import OpSimulator
from .cache import SimCache, content_hash
from .hub import SimulatorHub
from .backends.roofline import RooflineSimulator

__all__ = [
    "SimResult",
    "OpSimulator",
    "SimCache",
    "content_hash",
    "SimulatorHub",
    "RooflineSimulator",
]
