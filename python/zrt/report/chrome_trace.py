"""Chrome Trace builder — returns dict for chrome://tracing."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.executor.scheduler import Timeline


def build_chrome_trace(timeline: "Timeline", name: str = "") -> dict:
    """Convert a Timeline to Chrome Trace Event Format dict.

    Caller is responsible for json.dump to disk.
    """
    from python.zrt.training.trace.exporter import _CATEGORY_COLORS
    events = []
    for op in timeline.scheduled_ops:
        cat = op.category if op.category in _CATEGORY_COLORS else "compute"
        events.append({
            "name": op.node_id, "cat": cat, "ph": "X",
            "pid": 0, "tid": op.stream_id,
            "ts": op.start_us, "dur": op.latency_us,
            "args": {"op_type": op.op_type, "stream_type": op.stream_type},
        })
    return {
        "traceEvents": events,
        "displayTimeUnit": "us",
        "metadata": {"name": name or timeline.graph_name},
    }
