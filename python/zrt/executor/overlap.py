"""Compute-Communication overlap analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.executor.scheduler import Timeline


@dataclass
class OverlapReport:
    """Detailed compute-communication overlap analysis."""
    compute_us: float      # Total compute latency (sum of all compute ops)
    comm_us: float         # Total communication latency (sum of all comm ops)
    overlap_us: float      # Communication time hidden behind compute
    exposed_comm_us: float  # Communication time NOT hidden (critical for latency)
    overlap_ratio: float   # overlap_us / comm_us (0.0 to 1.0)
    critical_path_us: float  # Total wall-clock latency


@dataclass
class PerStrategyOverlapReport:
    """Per-parallelism-strategy overlap breakdown."""
    tp_total_us:    float = 0.0
    tp_exposed_us:  float = 0.0
    tp_hidden_us:   float = 0.0
    ep_total_us:    float = 0.0
    ep_exposed_us:  float = 0.0
    ep_hidden_us:   float = 0.0
    pp_total_us:    float = 0.0
    pp_exposed_us:  float = 0.0
    pp_hidden_us:   float = 0.0
    cp_total_us:    float = 0.0
    cp_exposed_us:  float = 0.0
    cp_hidden_us:   float = 0.0
    total_compute_us: float = 0.0
    total_comm_us:    float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "tp_total_us": self.tp_total_us,
            "tp_exposed_us": self.tp_exposed_us,
            "tp_hidden_us": self.tp_hidden_us,
            "ep_total_us": self.ep_total_us,
            "ep_exposed_us": self.ep_exposed_us,
            "ep_hidden_us": self.ep_hidden_us,
            "pp_total_us": self.pp_total_us,
            "pp_exposed_us": self.pp_exposed_us,
            "pp_hidden_us": self.pp_hidden_us,
            "cp_total_us": self.cp_total_us,
            "cp_exposed_us": self.cp_exposed_us,
            "cp_hidden_us": self.cp_hidden_us,
            "total_compute_us": self.total_compute_us,
            "total_comm_us": self.total_comm_us,
        }


def per_strategy_overlap(timeline: "Timeline") -> PerStrategyOverlapReport:
    """Compute per-parallelism-strategy overlap from a DAGScheduler Timeline.

    Extracts compute intervals and per-strategy communication intervals,
    then uses sweep-line intersection to compute hidden vs exposed comm time
    for each parallelism strategy (tp, ep, pp, cp).
    """
    compute_intervals = OverlapAnalyzer._intervals_static(timeline, "compute")
    total_compute = OverlapAnalyzer._sum_duration_static(compute_intervals)

    report = PerStrategyOverlapReport(total_compute_us=total_compute)

    all_comm_intervals: list[tuple[float, float]] = []
    for tag in ("tp", "ep", "pp", "cp"):
        comm_intervals = [
            (op.start_us, op.end_us)
            for op in timeline.scheduled_ops
            if op.stream_type == "comm" and op.parallelism_tag == tag
        ]
        all_comm_intervals.extend(comm_intervals)
        if not comm_intervals:
            continue
        total_comm = OverlapAnalyzer._sum_duration_static(comm_intervals)
        overlap = OverlapAnalyzer._intersection_static(compute_intervals, comm_intervals)
        exposed = max(0.0, total_comm - overlap)

        setattr(report, f"{tag}_total_us", total_comm)
        setattr(report, f"{tag}_exposed_us", exposed)
        setattr(report, f"{tag}_hidden_us", overlap)

    # Include untagged comm ops (overlap_type but no inserted_by)
    untagged_intervals = [
        (op.start_us, op.end_us)
        for op in timeline.scheduled_ops
        if op.stream_type == "comm" and op.parallelism_tag == ""
    ]
    all_comm_intervals.extend(untagged_intervals)

    report.total_comm_us = OverlapAnalyzer._sum_duration_static(all_comm_intervals)
    return report


class OverlapAnalyzer:
    """Analyze compute-communication overlap using interval intersection.

    The key insight: overlap = intersection of compute and comm time intervals.
    This is more accurate than the approximation (compute + comm - total).
    """

    def analyze(self, timeline: "Timeline") -> OverlapReport:
        """Analyze overlap in a timeline.

        Uses a sweep-line algorithm to compute the exact intersection of compute
        and communication intervals.
        """
        compute = self._intervals(timeline, "compute")
        comm = self._intervals(timeline, "comm")

        total_compute = self._sum_duration(compute)
        total_comm = self._sum_duration(comm)
        overlap = self._intersection(compute, comm)

        critical_path = timeline.total_latency_us
        exposed = total_comm - overlap

        overlap_ratio = 0.0
        if total_comm > 0:
            overlap_ratio = min(1.0, overlap / total_comm)

        return OverlapReport(
            compute_us=total_compute,
            comm_us=total_comm,
            overlap_us=overlap,
            exposed_comm_us=exposed,
            overlap_ratio=overlap_ratio,
            critical_path_us=critical_path,
        )

    def _intervals(self, timeline: "Timeline",
                   stream_type: str) -> list[tuple[float, float]]:
        """Extract time intervals for ops of a given type.

        Returns list of (start_us, end_us) tuples, not necessarily sorted.
        """
        return [
            (op.start_us, op.end_us)
            for op in timeline.scheduled_ops
            if op.stream_type == stream_type
        ]

    @staticmethod
    def _intervals_static(timeline: "Timeline",
                          stream_type: str) -> list[tuple[float, float]]:
        return [
            (op.start_us, op.end_us)
            for op in timeline.scheduled_ops
            if op.stream_type == stream_type
        ]

    def _sum_duration(self, intervals: list[tuple[float, float]]) -> float:
        """Sum the duration of all intervals (they may or may not overlap)."""
        return sum(end - start for start, end in intervals)

    @staticmethod
    def _sum_duration_static(intervals: list[tuple[float, float]]) -> float:
        return sum(end - start for start, end in intervals)

    def _intersection(self, a: list[tuple[float, float]],
                      b: list[tuple[float, float]]) -> float:
        """Compute total intersection time of two sets of intervals.

        Uses sweep-line algorithm:
        - Create events for all interval boundaries
        - Sort by time
        - Sweep through, tracking how many intervals from each set are active
        - When both sets are active, add to overlap
        """
        return self._intersection_static(a, b)

    @staticmethod
    def _intersection_static(a: list[tuple[float, float]],
                             b: list[tuple[float, float]]) -> float:
        if not a or not b:
            return 0.0

        events = []
        for s, e in a:
            events.append((s, 1, "a"))     # interval starts
            events.append((e, -1, "a"))    # interval ends
        for s, e in b:
            events.append((s, 1, "b"))
            events.append((e, -1, "b"))

        events.sort()

        overlap = 0.0
        active_a = 0
        active_b = 0
        prev_t = 0.0

        for t, delta, src in events:
            if active_a > 0 and active_b > 0:
                overlap += t - prev_t
            if src == "a":
                active_a += delta
            else:
                active_b += delta
            prev_t = t

        return max(0.0, overlap)
