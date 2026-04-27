"""SimResult: per-operator simulation output."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimResult:
    """Simulation result for a single operator node.

    Fields
    ------
    op_node_id          : matches OpNode.id in the originating graph
    latency_us          : total estimated execution time (μs)
    compute_us          : compute-bound time = flops / peak_flops (μs)
    memory_us           : memory-bound time  = bytes  / hbm_bw    (μs)
    flops               : floating-point (or integer) operations count
    read_bytes          : bytes read from HBM
    write_bytes         : bytes written to HBM
    arithmetic_intensity: flops / (read_bytes + write_bytes)  [ops/byte]
    bound               : "compute" | "memory" | "latency"
    hw_utilization      : fraction of peak compute actually used  [0-1]
    backend             : name of the simulator backend that produced this
    confidence          : estimate quality  [0-1]; roofline = 0.3, profDB = 0.9
    """

    op_node_id: str
    latency_us: float
    compute_us: float
    memory_us: float
    flops: int
    read_bytes: int
    write_bytes: int
    arithmetic_intensity: float
    bound: str
    hw_utilization: float
    backend: str
    confidence: float

    @property
    def total_bytes(self) -> int:
        return self.read_bytes + self.write_bytes

    @property
    def latency_ms(self) -> float:
        return self.latency_us / 1_000.0

    def __repr__(self) -> str:
        return (
            f"SimResult({self.op_node_id}, {self.latency_us:.2f}μs, "
            f"{self.bound}-bound, util={self.hw_utilization:.1%}, "
            f"backend={self.backend})"
        )
