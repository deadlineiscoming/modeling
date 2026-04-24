"""Test DualPipe and DualPipeV composers."""
from __future__ import annotations

import pytest

from zrt.training.compose.pipeline import (
    DualPipeComposer,
    DualPipeVComposer,
    OneF1BComposer,
    StepResult,
)
from zrt.training.compose.stage import StageTime
from zrt.training.spec.strategy import PPSched, Strategy


def _make_stage_times(pp, fwd=0.01, bwd=0.02):
    return [StageTime(fwd=fwd, bwd=bwd) for _ in range(pp)]


def _make_strategy(
    pp=4, vpp_chunks=1, dp=1, micro_batch=1, global_batch=32,
    schedule=PPSched.DUALPIPE,
):
    return Strategy(
        pp=pp, vpp_chunks=vpp_chunks, dp=dp,
        micro_batch=micro_batch, global_batch=global_batch,
        pp_schedule=schedule,
    )


def test_dualpipe_bubble_less_than_1f1b():
    st = [
        StageTime(fwd=0.02, bwd=0.01),
        StageTime(fwd=0.01, bwd=0.02),
        StageTime(fwd=0.02, bwd=0.01),
        StageTime(fwd=0.01, bwd=0.02),
    ]
    s = _make_strategy(pp=4)
    M = s.num_microbatches()

    dp_result = DualPipeComposer().compose(st, M, 4, 0.0, s)
    s1f1b = Strategy(pp=4, dp=1, micro_batch=1, global_batch=32)
    f1b_result = OneF1BComposer().compose(st, M, 4, 0.0, s1f1b)

    assert dp_result.bubble_fraction < f1b_result.bubble_fraction


def test_dualpipe_pp1_no_pipeline():
    st = _make_stage_times(1)
    s = _make_strategy(pp=1)
    M = s.num_microbatches()

    dp_result = DualPipeComposer().compose(st, M, 1, 0.0, s)
    f1b_result = OneF1BComposer().compose(st, M, 1, 0.0, s)

    assert dp_result.step_time == pytest.approx(f1b_result.step_time, rel=1e-9)


def test_dualpipev_bubble_less_than_dualpipe():
    st = _make_stage_times(4)
    s_v = _make_strategy(pp=4, vpp_chunks=2, schedule=PPSched.DUALPIPE_V)
    s_dp = _make_strategy(pp=4, vpp_chunks=1, schedule=PPSched.DUALPIPE)
    M = s_v.num_microbatches()

    dpv_result = DualPipeVComposer().compose(st, M, 4, 0.0, s_v)
    dp_result = DualPipeComposer().compose(st, M, 4, 0.0, s_dp)

    assert dpv_result.bubble_fraction <= dp_result.bubble_fraction + 1e-9


def test_dualpipev_vpp1_equals_dualpipe():
    st = _make_stage_times(4)
    s = _make_strategy(pp=4, vpp_chunks=1, schedule=PPSched.DUALPIPE_V)
    M = s.num_microbatches()

    dpv_result = DualPipeVComposer().compose(st, M, 4, 0.0, s)
    dp_result = DualPipeComposer().compose(st, M, 4, 0.0, s)

    assert dpv_result.step_time == pytest.approx(dp_result.step_time, rel=1e-9)
