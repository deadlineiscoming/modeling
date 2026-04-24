"""Chrome Trace JSON exporter for Timeline visualization.

Generates files compatible with chrome://tracing viewer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zrt.executor.scheduler import Timeline

_CATEGORY_COLORS = {
    "compute": "yellow",
    "communication": "blue",
    "memory": "green",
}


def export_chrome_trace(
    timeline: "Timeline",
    path: str | Path,
    graph_name: str = "",
) -> Path:
    """Export a Timeline to Chrome Trace JSON format.

    Parameters
    ----------
    timeline
        Scheduled timeline from DAGScheduler.
    path
        Output file path (.json).
    graph_name
        Optional process name.

    Returns
    -------
    Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    events = []
    pid = 0

    for op in timeline.scheduled_ops:
        cat = op.category if op.category in _CATEGORY_COLORS else "compute"
        color = _CATEGORY_COLORS.get(cat, "yellow")

        events.append({
            "name": op.node_id,
            "cat": cat,
            "ph": "X",
            "pid": pid,
            "tid": op.stream_id,
            "ts": op.start_us,
            "dur": op.latency_us,
            "args": {
                "op_type": op.op_type,
                "stream_type": op.stream_type,
                "color": color,
            },
        })

    data = {
        "traceEvents": events,
        "displayTimeUnit": "us",
        "metadata": {
            "name": graph_name or timeline.graph_name,
        },
    }

    path.write_text(json.dumps(data, indent=2))
    return path
