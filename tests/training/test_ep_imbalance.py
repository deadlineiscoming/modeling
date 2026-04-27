"""Test EP load imbalance factor."""
from __future__ import annotations

from zrt.training.compose.stage import ep_imbalance_factor


def test_no_imbalance_when_ep1():
    assert ep_imbalance_factor(256, 1) == 1.0


def test_no_imbalance_when_no_experts():
    assert ep_imbalance_factor(0, 8) == 1.0


def test_imbalance_when_ep8():
    factor = ep_imbalance_factor(256, 8, topk=6)
    assert factor > 1.0


def test_imbalance_increases_with_fewer_experts_per_gpu():
    f8 = ep_imbalance_factor(256, 8, topk=6)
    f64 = ep_imbalance_factor(256, 64, topk=6)
    assert f64 > f8


def test_imbalance_increases_with_topk():
    f1 = ep_imbalance_factor(256, 8, topk=1)
    f6 = ep_imbalance_factor(256, 8, topk=6)
    assert f6 > f1
