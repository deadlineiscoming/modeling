"""Test graph-path schedule dispatch in TrainingPipelinePass."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.transform.analysis.training import TrainingPipelinePass
from python.zrt.transform.context import (
    ParallelConfig,
    TrainingConfig,
    TransformContext,
)


def _make_graph(metadata=None):
    return OpGraph(name="test", phase="prefill", metadata=metadata or {})


def _make_ctx(tp=1, pp=4, dp=1, pp_schedule="1f1b", vpp_chunks=1,
              micro_batch=1, global_batch=8):
    from python.zrt.hardware.spec import (
        ComputeSpec, HardwareSpec, InterconnectSpec, LinkSpec, MemorySpec,
    )
    hw = HardwareSpec(
        name="test_h100", vendor="nvidia", device_type="gpu",
        compute=ComputeSpec(bf16_tflops=989, fp8_tops=1979),
        memory=MemorySpec(capacity_gb=80, hbm_bandwidth_gbps=3350),
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=100, latency_us=10.0),
        ),
    )
    return TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=tp, pp=pp, dp=dp),
        training=TrainingConfig(
            pp_schedule=pp_schedule,
            vpp_chunks=vpp_chunks,
            micro_batch=micro_batch,
            global_batch=global_batch,
        ),
    )


def _run_pass(pp, pp_schedule, vpp_chunks=1, per_stage_us=1000.0):
    g = _make_graph(metadata={
        "num_layers": 4, "num_layers_traced": 4,
        "training_flops": 1e12,
    })
    ctx = _make_ctx(pp=pp, pp_schedule=pp_schedule, vpp_chunks=vpp_chunks)

    mock_timeline = MagicMock()
    mock_timeline.total_latency_us = per_stage_us * pp

    with patch("python.zrt.executor.scheduler.DAGScheduler") as MockSched:
        MockSched.return_value.schedule.return_value = mock_timeline
        result = TrainingPipelinePass().run(g, ctx)

    return result.metadata["pipeline_metrics"]


def test_vpp_reduces_step_time_vs_1f1b():
    f1b = _run_pass(pp=4, pp_schedule="1f1b")
    vpp = _run_pass(pp=4, pp_schedule="interleaved", vpp_chunks=2)
    assert vpp.step_time_ms < f1b.step_time_ms


def test_dualpipe_reduces_step_time_vs_1f1b():
    f1b = _run_pass(pp=4, pp_schedule="1f1b")
    dp = _run_pass(pp=4, pp_schedule="dualpipe")
    assert dp.step_time_ms < f1b.step_time_ms


def test_dualpipev_reduces_step_time_vs_dualpipe():
    dp = _run_pass(pp=4, pp_schedule="dualpipe")
    dpv = _run_pass(pp=4, pp_schedule="dualpipev", vpp_chunks=2)
    assert dpv.step_time_ms <= dp.step_time_ms


def test_1f1b_bubble_fraction():
    metrics = _run_pass(pp=4, pp_schedule="1f1b")
    pp = 4
    M = 8
    expected_bubble = (pp - 1) / (M + pp - 1)
    assert metrics.bubble_fraction == pytest.approx(expected_bubble, abs=0.01)


def test_pp1_no_pipeline():
    metrics = _run_pass(pp=1, pp_schedule="1f1b")
    assert metrics.step_time_ms > 0
