"""Test Chrome Trace export."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from zrt.training.trace.exporter import export_chrome_trace


def _make_timeline():
    from zrt.executor.scheduler import ScheduledOp, Timeline
    ops = [
        ScheduledOp(
            node_id="mm_0", stream_id=0, stream_type="compute",
            start_us=0.0, end_us=100.0, latency_us=100.0,
            op_type="aten.mm.default", category="compute",
        ),
        ScheduledOp(
            node_id="ar_0", stream_id=1, stream_type="comm",
            start_us=50.0, end_us=200.0, latency_us=150.0,
            op_type="comm.all_reduce", category="communication",
        ),
    ]
    return Timeline(scheduled_ops=ops, graph_name="test_graph")


def test_export_chrome_trace_produces_valid_json(tmp_path):
    tl = _make_timeline()
    out = export_chrome_trace(tl, tmp_path / "trace.json", graph_name="test")

    assert out.exists()
    data = json.loads(out.read_text())
    assert "traceEvents" in data
    assert len(data["traceEvents"]) == 2
    assert data["displayTimeUnit"] == "us"

    evt0 = data["traceEvents"][0]
    assert evt0["name"] == "mm_0"
    assert evt0["ph"] == "X"
    assert evt0["tid"] == 0
    assert evt0["dur"] == 100.0


def test_export_chrome_trace_categories(tmp_path):
    tl = _make_timeline()
    out = export_chrome_trace(tl, tmp_path / "trace.json")
    data = json.loads(out.read_text())

    cats = {e["cat"] for e in data["traceEvents"]}
    assert "compute" in cats
    assert "communication" in cats
