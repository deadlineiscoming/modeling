"""Chrome Trace exporter — convert pipeline scheduling results to JSON.

Supports three input types:
  - ``PPStitchedTimeline``   : per-stage x microbatch pipeline grid tasks
  - ``Timeline`` (per-stage) : per-op trace from DAGScheduler
  - ``List[Timeline]``       : multi-stage per-op trace, one per GPU

Chrome Trace event format (JSON array of dicts):

  {
    "ph":  "X",            // complete event (has duration)
    "name": "op_name",     // display name
    "cat":  "compute",     // category for filtering
    "pid":  stage_id,      // process = pipeline stage
    "tid":  stream_id,     // thread  = compute/comm stream within stage
    "ts":   123456.0,      // timestamp in microseconds
    "dur":  789.0,         // duration in microseconds
    "args": {"desc": "..."}
  }

Usage
-----
>>> from python.zrt.executor.chrome_trace import ChromeTraceExporter
>>> exporter = ChromeTraceExporter()
>>> exporter.export_stitched(result, "pipeline.json")      # PP grid
>>> exporter.export_per_stage(timelines, "detail.json")    # per-op detail
>>> exporter.export_combined(result, timelines, "all.json") # combined
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.executor.pp_stitcher import PPStitchedTimeline
    from python.zrt.executor.scheduler import ScheduledOp, Timeline


# ── colour palette ────────────────────────────────────────────────────────────

_COLORS: dict[str, str] = {
    # FWD phases — warm tones
    "fwd_compute":        "good",     # green
    "fwd_comm":           "olive",    # olive
    "fwd_p2p":            "terracotta",
    # BWD phases — cool tones
    "bwd_compute":        "blue",     # blue
    "bwd_comm":           "purple",
    "bwd_p2p":            "magenta",
    "bwd_dx_compute":     "blue",
    "bwd_dw_compute":     "navy",
    # Memory / other
    "memory":             "yellow",
    "idle":               "grey",
    "bubble":             "light_grey",
    "warmup":             "orange",
    "cooldown":           "teal",
}

_NAMES: dict[str, str] = {
    "fwd_compute":        "▼ FWD [c]",
    "fwd_comm":           "▼ FWD [n]",
    "fwd_p2p":            "▼ FWD [p2p]",
    "bwd_compute":        "▲ BWD [c]",
    "bwd_comm":           "▲ BWD [n]",
    "bwd_p2p":            "▲ BWD [p2p]",
    "bwd_dx_compute":     "▲ BWD_dX [c]",
    "bwd_dw_compute":     "▲ BWD_dW [c]",
    "memory":             "◇ MEM",
    "idle":               "· idle",
    "bubble":             "∅ bubble",
}


@dataclass
class ChromeTraceEvent:
    """A single Chrome Trace complete event (ph="X")."""

    name: str
    cat: str
    pid: int
    tid: int
    ts: float
    dur: float
    args: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ph": "X",
            "name": self.name,
            "cat": self.cat,
            "pid": self.pid,
            "tid": self.tid,
            "ts": self.ts,
            "dur": self.dur,
            "args": self.args,
        }


class ChromeTraceExporter:
    """Export pipeline scheduling results as Chrome Trace JSON.

    Parameters
    ----------
    time_unit : str
        "us" (default) or "ns".  Chrome Trace ``ts`` / ``dur`` fields
        are always in microseconds; ``ns`` multiplies by 1000.
    """

    _MIN_VISIBLE_US = 1.0

    def __init__(self, time_unit: str = "us") -> None:
        self._mult = 1000.0 if time_unit == "ns" else 1.0
        self._time_unit = time_unit

    # ── metadata helpers ──────────────────────────────────────────────────

    @staticmethod
    def _process_name_meta(pid: int, name: str) -> dict:
        return {"ph": "M", "pid": pid, "ts": 0, "name": "process_name", "args": {"name": name}}

    @staticmethod
    def _thread_name_meta(pid: int, tid: int, name: str) -> dict:
        return {"ph": "M", "pid": pid, "tid": tid, "ts": 0, "name": "thread_name", "args": {"name": name}}

    def _grid_meta_events(self, pp: int) -> list[dict]:
        meta: list[dict] = []
        for s in range(pp):
            meta.append(self._process_name_meta(s, f"Stage {s}"))
            meta.append(self._thread_name_meta(s, 0, "Grid Schedule"))
        return meta

    def _per_stage_meta_events(self, pp: int) -> list[dict]:
        meta: list[dict] = []
        for s in range(pp):
            meta.append(self._process_name_meta(s, f"Stage {s}"))
            meta.append(self._thread_name_meta(s, 0, "Compute Ops"))
            meta.append(self._thread_name_meta(s, 1, "Comm Ops"))
        return meta

    def _combined_meta_events(self, pp: int) -> list[dict]:
        meta: list[dict] = []
        for s in range(pp):
            meta.append(self._process_name_meta(s, f"Stage {s}"))
            meta.append(self._thread_name_meta(s, 0, "Grid Schedule"))
            meta.append(self._thread_name_meta(s, 2, "Compute Ops"))
            meta.append(self._thread_name_meta(s, 3, "Comm Ops"))
        return meta

    # ── deduplication ─────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(events: list[dict]) -> list[dict]:
        """Merge X-events that share (pid, tid, ts, name, cat) into one with count."""
        meta = [e for e in events if e.get("ph") != "X"]
        x_events = [e for e in events if e.get("ph") == "X"]

        groups: dict[tuple, list[dict]] = {}
        for e in x_events:
            key = (e["pid"], e["tid"], e["ts"], e["name"], e.get("cat", ""))
            groups.setdefault(key, []).append(e)

        merged: list[dict] = []
        for key, group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                first = dict(group[0])
                first["args"] = dict(first.get("args", {}))
                first["args"]["count"] = len(group)
                merged.append(first)

        return meta + merged

    # ── public API ────────────────────────────────────────────────────────

    def export_stitched(
        self, result: "PPStitchedTimeline", path: str | None = None,
    ) -> str:
        """Export PPStitchedTimeline (stage x microbatch grid).

        Each GridTask becomes one trace event.  pid = stage_id,
        tid = 0 (grid schedule).  Same microbatch index gets the same
        colour across all stages via ``cat = mb_{mb}``.

        Metadata events name each process "Stage N" and thread
        "Grid Schedule".
        """
        events: list[dict] = []
        events.extend(self._grid_meta_events(result.pp))

        for task in result.tasks:
            cat = self._grid_cat(task)
            name = self._name_for_task(task)
            events.append(ChromeTraceEvent(
                name=name,
                cat=cat,
                pid=task.stage_id,
                tid=0,
                ts=task.start_us * self._mult,
                dur=max(task.latency_us, self._MIN_VISIBLE_US) * self._mult,
                args={
                    "phase": task.phase,
                    "mb": task.mb_id,
                    "stage": task.stage_id,
                    "dep_count": len(task.dependencies),
                },
            ).to_dict())

        if result.warmup_us > 0:
            for s in range(result.pp):
                events.append(self._instant(
                    name=_NAMES.get("warmup", "warmup"),
                    pid=s, tid=0,
                    ts=result.warmup_us * self._mult,
                    args={"phase": "warmup", "dur_us": result.warmup_us},
                ))
        if result.cooldown_us > 0:
            cooldown_ts = (result.step_time_us - result.cooldown_us) * self._mult
            for s in range(result.pp):
                events.append(self._instant(
                    name=_NAMES.get("cooldown", "cooldown"),
                    pid=s, tid=0,
                    ts=cooldown_ts,
                    args={"phase": "cooldown", "dur_us": result.cooldown_us},
                ))

        doc = self._build_doc(events)
        if path:
            self._write(path, doc)
        return doc

    def export_per_stage(
        self,
        timelines: list["Timeline"],
        path: str | None = None,
        *,
        M: int = 1,
        pp_stitched: "PPStitchedTimeline | None" = None,
        replicate: bool = True,
    ) -> str:
        """Export per-stage DAGScheduler Timelines (per-op detail).

        ``timelines[s]`` maps to ``pid=s``.  Within each stage, ops on
        different streams are rendered on separate ``tid`` values
        (0 = "Compute Ops", 1 = "Comm Ops").

        When ``replicate=True`` and ``M > 1``, each microbatch's ops are
        expanded as time-offset replicas aligned with the pipeline grid
        schedule.  When ``replicate=False``, only one clean reference copy
        of each per-stage timeline is exported.

        Zero-latency ops receive a minimum visible duration so they
        are not invisible in Chrome Trace.
        """
        pp = len(timelines)
        events: list[dict] = []
        events.extend(self._per_stage_meta_events(pp))

        grid_slot: dict[tuple[int, int, str], float] = {}
        if pp_stitched is not None and M > 1:
            for task in pp_stitched.tasks:
                if task.phase in ("fwd", "bwd", "bwd_dx", "bwd_dw"):
                    grid_slot[(task.stage_id, task.mb_id, task.phase)] = task.start_us

        for s, tl in enumerate(timelines):
            fwd_lat = tl.phase_latency("fwd")
            bwd_lat = tl.phase_latency("bwd")
            if fwd_lat == 0.0 and bwd_lat == 0.0:
                fwd_lat = tl.total_latency_us
            stage_total = fwd_lat + bwd_lat

            num_replicas = M if replicate else 1

            for m in range(num_replicas):
                fwd_base = grid_slot.get((s, m, "fwd"), m * stage_total)
                bwd_base = grid_slot.get(
                    (s, m, "bwd"),
                    grid_slot.get((s, m, "bwd_dx"), m * stage_total + fwd_lat),
                )

                for op in tl.scheduled_ops:
                    cat = "communication" if op.stream_type == "comm" else "compute"
                    if replicate:
                        name = f"m{m}:{op.phase}:{op.op_type}" if op.phase else f"m{m}:{op.op_type}"
                    else:
                        name = f"{op.phase}:{op.op_type}" if op.phase else op.op_type

                    if op.phase == "fwd" or not op.phase:
                        base = fwd_base
                        rel_start = op.start_us
                    else:
                        base = bwd_base
                        rel_start = op.start_us - fwd_lat

                    dur_us = max(op.latency_us, self._MIN_VISIBLE_US) if op.stream_type == "comm" else op.latency_us

                    events.append(ChromeTraceEvent(
                        name=name,
                        cat=cat,
                        pid=s,
                        tid=op.stream_id,
                        ts=(base + rel_start) * self._mult,
                        dur=dur_us * self._mult,
                        args={
                            "phase": op.phase,
                            "op_type": op.op_type,
                            "stream_type": op.stream_type,
                            "mb": m,
                        },
                    ).to_dict())

        events = self._deduplicate(events)
        doc = self._build_doc(events)
        if path:
            self._write(path, doc)
        return doc

    def export_combined(
        self,
        stitched: "PPStitchedTimeline",
        timelines: list["Timeline"],
        path: str | None = None,
    ) -> str:
        """Combined export: PP grid + per-stage op detail shared on same pids.

        pid = stage_id (0 .. pp-1).

        Within each stage:
          tid 0 = "Grid Schedule"   — grid-level FWD/BWD blocks, coloured by mb
          tid 2 = "Compute Ops"     — per-op compute detail (reference copy)
          tid 3 = "Comm Ops"        — per-op communication detail (reference copy)

        Grid task categories use ``mb_{mb}`` so the same
        microbatch index gets the same colour across all stages.
        """
        pp = stitched.pp
        events: list[dict] = []
        events.extend(self._combined_meta_events(pp))

        for task in stitched.tasks:
            events.append(ChromeTraceEvent(
                name=self._name_for_task(task),
                cat=self._grid_cat(task),
                pid=task.stage_id,
                tid=0,
                ts=task.start_us * self._mult,
                dur=max(task.latency_us, self._MIN_VISIBLE_US) * self._mult,
                args={
                    "phase": task.phase,
                    "mb": task.mb_id,
                    "stage": task.stage_id,
                    "view": "grid",
                    "dep_count": len(task.dependencies),
                },
            ).to_dict())

        for s, tl in enumerate(timelines):
            for op in tl.scheduled_ops:
                cat = "communication" if op.stream_type == "comm" else "compute"
                name = f"{op.phase}:{op.op_type}" if op.phase else op.op_type
                dur_us = max(op.latency_us, self._MIN_VISIBLE_US) if op.stream_type == "comm" else op.latency_us
                events.append(ChromeTraceEvent(
                    name=name,
                    cat=cat,
                    pid=s,
                    tid=2 + op.stream_id,
                    ts=op.start_us * self._mult,
                    dur=dur_us * self._mult,
                    args={
                        "phase": op.phase,
                        "op_type": op.op_type,
                        "stream_type": op.stream_type,
                        "view": "detail",
                    },
                ).to_dict())

        events = self._deduplicate(events)
        doc = self._build_doc(events)
        if path:
            self._write(path, doc)
        return doc

    def export_stitched_detailed(
        self,
        stitched: "PPStitchedTimeline",
        timelines: list["Timeline"],
        path: str | None = None,
    ) -> str:
        """Stitched grid with per-stage detail on separate tids (same pid).

        pid = stage_id.  Within each stage:
          tid 0              = grid-level fwd/bwd blocks, coloured by mb
          tid 2 + stream_id  = per-op detail from DAGScheduler Timeline

        All microbatches share the same detail rows — they are distinguished
        by time offsets from the pipeline schedule, naturally serialised on
        their physical stream.
        """
        pp = stitched.pp
        events: list[dict] = []
        events.extend(self._combined_meta_events(pp))

        grid_index: dict[tuple[int, int, str], float] = {}
        for task in stitched.tasks:
            key = (task.stage_id, task.mb_id, task.phase)
            grid_index[key] = task.start_us

        for task in stitched.tasks:
            events.append(ChromeTraceEvent(
                name=self._name_for_task(task),
                cat=self._grid_cat(task),
                pid=task.stage_id,
                tid=0,
                ts=task.start_us * self._mult,
                dur=max(task.latency_us, self._MIN_VISIBLE_US) * self._mult,
                args={"phase": task.phase, "mb": task.mb_id, "view": "grid"},
            ).to_dict())

        for s, tl in enumerate(timelines):
            for m in range(stitched.M):
                fwd_base = grid_index.get((s, m, "fwd"), 0.0)
                for op in tl.scheduled_ops:
                    if op.phase == "fwd":
                        dur_us = max(op.latency_us, self._MIN_VISIBLE_US) if op.stream_type == "comm" else op.latency_us
                        events.append(ChromeTraceEvent(
                            name=f"{op.op_type}",
                            cat="compute" if op.stream_type != "comm" else "communication",
                            pid=s,
                            tid=2 + op.stream_id,
                            ts=(fwd_base + op.start_us) * self._mult,
                            dur=dur_us * self._mult,
                            args={
                                "phase": "fwd",
                                "mb": m,
                                "op_type": op.op_type,
                                "view": "detail",
                            },
                        ).to_dict())

                bwd_base = grid_index.get((s, m, "bwd"), grid_index.get((s, m, "bwd_dx"), 0.0))
                for op in tl.scheduled_ops:
                    if "bwd" in op.phase:
                        dur_us = max(op.latency_us, self._MIN_VISIBLE_US) if op.stream_type == "comm" else op.latency_us
                        events.append(ChromeTraceEvent(
                            name=f"{op.op_type}",
                            cat="compute" if op.stream_type != "comm" else "communication",
                            pid=s,
                            tid=2 + op.stream_id,
                            ts=(bwd_base + op.start_us) * self._mult,
                            dur=dur_us * self._mult,
                            args={
                                "phase": "bwd",
                                "mb": m,
                                "op_type": op.op_type,
                                "view": "detail",
                            },
                        ).to_dict())

        doc = self._build_doc(events)
        if path:
            self._write(path, doc)
        return doc

    # ── internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _grid_cat(task) -> str:
        return f"mb_{task.mb_id}"

    @staticmethod
    def _name_for_task(task) -> str:
        arrow = "▼" if task.phase in ("fwd",) else "▲"
        kind = task.phase.upper() if task.phase else "?"
        return f"{arrow} {kind} s{task.stage_id} m{task.mb_id}"

    @staticmethod
    def _instant(name: str, pid: int, tid: int, ts: float, args: dict) -> dict:
        return {
            "ph": "i",
            "name": name,
            "pid": pid,
            "tid": tid,
            "ts": ts,
            "s": "g",   # scope = global
            "args": args,
        }

    def _build_doc(self, events: list[dict]) -> str:
        return json.dumps(
            {
                "traceEvents": events,
                "displayTimeUnit": "ns" if self._time_unit == "ns" else "ms",
            },
            indent=2,
            ensure_ascii=False,
        )

    @staticmethod
    def _write(path: str, content: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)