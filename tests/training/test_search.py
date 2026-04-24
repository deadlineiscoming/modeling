"""Test grid search and Pareto frontier."""
from __future__ import annotations

import pytest

from zrt.training.search.space import SearchSpace
from zrt.training.search.estimator import pareto_frontier, Report
from zrt.training.spec.strategy import PPSched
from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.system import SystemSpec, GPU, NetTier


def _make_model():
    return ModelSpec(
        hidden=4096,
        ffn=11008,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        vocab=32000,
        seq_len=2048,
        layers=[LayerKind.DENSE] * 8,
        num_experts=0,
        top_k=0,
    )


def _make_system():
    return SystemSpec(
        gpu=GPU(
            name="h100",
            flops_bf16=989,
            flops_fp8=1979,
            hbm_gb=80,
            hbm_bw_gbps=3350,
        ),
        host_mem_gb=512,
        nets=[
            NetTier(scope="intra_node", bw_gbps=900, latency_us=0.5, topology="nvswitch"),
            NetTier(scope="inter_node", bw_gbps=100, latency_us=10, topology="fattree"),
        ],
        nodes=1,
        gpus_per_node=8,
    )


def test_search_space_generates_valid_strategies():
    space = SearchSpace(
        tp_values=[1, 2, 4, 8],
        cp_values=[1],
        pp_values=[1, 2],
        ep_values=[1],
        dp_values=[1, 2, 4, 8],
        zero_stages=[0, 1],
        pp_schedules=[PPSched.ONE_F_ONE_B],
        recompute_policies=["none"],
    )
    strategies = space.strategies(world_size=8)
    assert len(strategies) > 0

    for s in strategies:
        total = s.tp * s.cp * s.pp * s.ep * s.dp
        assert total == 8


def test_search_space_skips_invalid_zero():
    space = SearchSpace(
        tp_values=[8],
        cp_values=[1],
        pp_values=[1],
        ep_values=[1],
        dp_values=[1],
        zero_stages=[1],
        pp_schedules=[PPSched.ONE_F_ONE_B],
        recompute_policies=["none"],
    )
    strategies = space.strategies(world_size=8)
    assert len(strategies) == 0


def test_pareto_frontier_basic():
    reports = [
        Report(step_time_ms=100.0, mfu=0.5, total_flops=1e12),
        Report(step_time_ms=80.0, mfu=0.6, total_flops=1e12),
        Report(step_time_ms=120.0, mfu=0.4, total_flops=1e12),
    ]
    frontier = pareto_frontier(reports)
    assert len(frontier) >= 1
    assert frontier[0].step_time_ms == 80.0


def test_pareto_frontier_empty():
    assert pareto_frontier([]) == []
