"""Demo: graph-stitched PP pipeline scheduling.

Shows how PPStitcher replaces formula-based PipelineComposer with
a topology-driven stage x microbatch grid approach.

The demo:
  1. Creates synthetic per-stage fwd/bwd latencies (simulating DAGScheduler output)
  2. Stitches PP pipeline via PPStitcher
  3. Compares different PP schedules
  4. Prints Chrome Trace style output
"""
from __future__ import annotations

from python.zrt.executor.pp_stitcher import (
    PPStitcher,
    PPStitchedTimeline,
    stitch_pp_pipeline,
    GridTask,
)


def demo_basic_1f1b():
    """Demonstrate basic 1F1B pipeline stitch."""
    print("=" * 72)
    print("DEMO 1: 1F1B Pipeline (pp=4, M=8)")
    print("=" * 72)

    # Simulated per-stage latencies from DAGScheduler (us)
    # stage 2 is the bottleneck with 120 fwd + 240 bwd
    stage_fwd_us = {0: 100, 1: 80, 2: 120, 3: 90}
    stage_bwd_us = {0: 200, 1: 160, 2: 240, 3: 180}

    result = stitch_pp_pipeline(
        stage_fwd_us=stage_fwd_us,
        stage_bwd_us=stage_bwd_us,
        pp=4, M=8,
        p2p_latency_us=5,   # P2P activation transfer
        schedule="1f1b",
    )

    print(result.summary())
    print()

    _print_stage_timeline(result)


def demo_schedule_comparison():
    """Compare different PP schedules on the same per-stage times."""
    print("\n" + "=" * 72)
    print("DEMO 2: Schedule Comparison (pp=4, M=16)")
    print("=" * 72)

    stage_fwd_us = {0: 50, 1: 60, 2: 55, 3: 48}
    stage_bwd_us = {0: 100, 1: 120, 2: 110, 3: 96}
    stage_bwd_dw_us = {0: 30, 1: 36, 2: 33, 3: 28}

    schedules = [
        ("1f1b", {}),
        ("dualpipe", {}),
        ("zb", {"stage_bwd_dw_us": stage_bwd_dw_us}),
    ]

    results: list[tuple[str, PPStitchedTimeline]] = []
    for sched_name, extra in schedules:
        dw = extra.get("stage_bwd_dw_us")
        result = stitch_pp_pipeline(
            stage_fwd_us=stage_fwd_us,
            stage_bwd_us=stage_bwd_us,
            pp=4, M=16,
            p2p_latency_us=2,
            schedule=sched_name,
            stage_bwd_dw_us=dw,
        )
        results.append((sched_name, result))

    # Comparison table
    header = f"{'Schedule':<16} {'Step(us)':>12} {'Step(ms)':>10} {'Bubble(us)':>12} {'Bubble%':>10}"
    print(header)
    print("-" * len(header))
    for name, r in results:
        print(
            f"{name:<16} {r.step_time_us:>12.1f} {r.step_time_us/1000:>10.3f} "
            f"{r.bubble_us:>12.1f} {r.bubble_fraction:>9.2%}"
        )

    # Formula-based reference for 1F1B
    print()
    print("Formula-based 1F1B reference (for comparison):")
    t_fwd_max = max(stage_fwd_us.values())
    t_bwd_max = max(stage_bwd_us.values())
    t_stage_max = max(
        stage_fwd_us[s] + stage_bwd_us[s] for s in range(4)
    )
    pp = 4
    M = 16
    formula_step = (pp - 1) * t_fwd_max + M * t_stage_max + (pp - 1) * t_bwd_max
    formula_bubble = (pp - 1) * (t_fwd_max + t_bwd_max)
    print(f"  step = {formula_step:.1f} us, bubble = {formula_bubble:.1f} us "
          f"({formula_bubble/formula_step:.2%})")


def demo_with_dagscheduler_mock():
    """Demonstrate integration with DAGScheduler-like input.

    This simulates what TrainingPipelinePass would provide:
    per-stage Timelines with TP/EP communication already embedded.
    """
    print("\n" + "=" * 72)
    print("DEMO 3: Integration with DAGScheduler (mock)")
    print("=" * 72)

    from python.zrt.executor.scheduler import Timeline, ScheduledOp

    # Build mock per-stage Timelines with TP all_reduce and EP all_to_all
    timelines: list[Timeline] = []
    for s in range(4):
        ops: list[ScheduledOp] = []
        t = 0.0

        # Forward: attention -> TP all_reduce -> FFN -> P2P send
        ops.append(ScheduledOp("attn_fwd", s, "compute", t, t + 30, 30, "matmul", "compute", "fwd"))
        t += 30
        ops.append(ScheduledOp("tp_ar_1", s + 10, "comm", t, t + 5, 5, "comm.all_reduce", "communication", "fwd"))
        t += 5
        ops.append(ScheduledOp("ffn_fwd", s, "compute", t, t + 25, 25, "matmul", "compute", "fwd"))
        t += 25
        # P2P send: modeled as comm node on a separate stream
        ops.append(ScheduledOp("p2p_send", s + 10, "comm", t, t + 3, 3, "comm.send_recv", "communication", "fwd"))
        t += 3

        # Backward: P2P recv -> FFN bwd -> TP all_reduce -> attn bwd
        ops.append(ScheduledOp("p2p_recv", s + 10, "comm", t, t + 3, 3, "comm.send_recv", "communication", "bwd"))
        t += 3
        ops.append(ScheduledOp("ffn_bwd", s, "compute", t, t + 50, 50, "matmul", "compute", "bwd"))
        t += 50
        ops.append(ScheduledOp("tp_ar_2", s + 10, "comm", t, t + 5, 5, "comm.all_reduce", "communication", "bwd"))
        t += 5
        ops.append(ScheduledOp("attn_bwd", s, "compute", t, t + 60, 60, "matmul", "compute", "bwd"))
        t += 60

        tl = Timeline(scheduled_ops=ops, graph_name=f"stage_{s}", phase="fwd+bwd")
        timelines.append(tl)

    # Extract per-stage latencies
    stage_fwd = {s: tl.phase_latency("fwd") for s, tl in enumerate(timelines)}
    stage_bwd = {s: tl.phase_latency("bwd") for s, tl in enumerate(timelines)}

    print("Per-stage latencies from DAGScheduler:")
    for s in range(4):
        print(f"  stage {s}: fwd={stage_fwd[s]:.0f} us, bwd={stage_bwd[s]:.0f} us")

    # Stitch PP pipeline
    stitcher = PPStitcher(
        stage_fwd_us=stage_fwd,
        stage_bwd_us=stage_bwd,
        pp=4, M=8,
        p2p_latency_us=3,
        schedule="1f1b",
    )
    result = stitcher.stitch_from_timelines(timelines)

    print()
    print(result.summary())
    print()
    _print_device_gantt(result, max_entries=16)


def demo_vpp_interleaved():
    """Demonstrate VPP/Interleaved scheduling."""
    print("\n" + "=" * 72)
    print("DEMO 4: VPP Interleaved (pp=2, vpp_chunks=2, M=8)")
    print("=" * 72)

    stage_fwd_us = {0: 100, 1: 100}
    stage_bwd_us = {0: 200, 1: 200}

    # VPP: each physical stage holds 2 virtual stages
    # The PPStitcher currently delegates VPP to the same edge pattern
    # as 1F1B; VPP-specific interleaving could be added as a refinement.
    result = stitch_pp_pipeline(
        stage_fwd_us=stage_fwd_us,
        stage_bwd_us=stage_bwd_us,
        pp=2, M=8,
        p2p_latency_us=3,
        schedule="interleaved",
        vpp_chunks=2,
    )

    print(result.summary())
    print()
    print("Note: VPP edge refinement (virtual stage interleaving) is a future extension.")
    print("      Current implementation uses 1F1B edge pattern as baseline.")


# --- visualization helpers -----------------------------------------------------

def _print_stage_timeline(result: PPStitchedTimeline):
    """Print per-stage, per-microbatch timeline table."""
    # Group tasks by stage
    by_stage: dict[int, list[GridTask]] = {}
    for t in result.tasks:
        by_stage.setdefault(t.stage_id, []).append(t)

    print("Per-stage timeline (first 4 microbatches):")
    for s in sorted(by_stage.keys()):
        stage_tasks = sorted(by_stage[s], key=lambda t: (t.mb_id, t.phase))
        entries = []
        for t in stage_tasks[:16]:  # limit output
            entries.append(
                f"  mb={t.mb_id} {t.phase:6s} [{t.start_us:>8.0f} -> {t.end_us:>8.0f}] "
                f"dur={t.latency_us:>6.0f} us"
            )
        print(f" Stage {s}:")
        for e in entries:
            print(e)


def _print_device_gantt(result: PPStitchedTimeline, max_entries: int = 20):
    """Print a text-based Gantt chart for each device."""
    by_device: dict[int, list[GridTask]] = {}
    for t in result.tasks:
        by_device.setdefault(t.stream_id, []).append(t)

    total = result.step_time_us
    width = 60

    print("Text Gantt Chart (F=fwd, B=bwd, X=bwd_dx/dw):")
    for dev in sorted(by_device.keys()):
        dev_tasks = sorted(by_device[dev], key=lambda t: t.start_us)
        timeline = [" "] * width
        for t in dev_tasks[:max_entries]:
            start_col = int(t.start_us / total * width)
            end_col = int(t.end_us / total * width)
            ch = "F" if t.phase == "fwd" else "B" if t.phase == "bwd" else "X"
            for col in range(start_col, min(end_col, width)):
                timeline[col] = ch
        print(f"  GPU {dev}: |{''.join(timeline)}|")


# --- main ----------------------------------------------------------------------

if __name__ == "__main__":
    demo_basic_1f1b()
    demo_schedule_comparison()
    demo_with_dagscheduler_mock()
    demo_vpp_interleaved()