"""Test anchor validation."""
from __future__ import annotations

import pytest

from zrt.training.anchor.validate import Anchor, validate_anchor
from zrt.training.search.estimator import Report


def test_validate_anchor_pass_within_tolerance():
    report = Report(step_time_ms=100.0, mfu=0.50, total_flops=1e12)
    anchor = Anchor(name="test", step_time_ms=105.0, mfu=0.52, total_flops=1.05e12, tolerance=0.15)

    warnings = validate_anchor(report, anchor)
    assert len(warnings) == 0


def test_validate_anchor_warns_on_step_time_deviation():
    report = Report(step_time_ms=150.0, mfu=0.50, total_flops=1e12)
    anchor = Anchor(name="test", step_time_ms=100.0, tolerance=0.15)

    warnings = validate_anchor(report, anchor)
    assert len(warnings) == 1
    assert "step_time_ms" in warnings[0]
    assert "50.0%" in warnings[0]


def test_validate_anchor_warns_on_mfu_deviation():
    report = Report(step_time_ms=100.0, mfu=0.30, total_flops=1e12)
    anchor = Anchor(name="test", mfu=0.50, tolerance=0.15)

    warnings = validate_anchor(report, anchor)
    assert len(warnings) == 1
    assert "mfu" in warnings[0]


def test_validate_anchor_skips_none_fields():
    report = Report(step_time_ms=100.0, mfu=0.50, total_flops=1e12)
    anchor = Anchor(name="test", step_time_ms=100.0)  # mfu and total_flops are None

    warnings = validate_anchor(report, anchor)
    assert len(warnings) == 0


def test_validate_anchor_custom_tolerance():
    report = Report(step_time_ms=110.0)
    anchor = Anchor(name="test", step_time_ms=100.0, tolerance=0.05)  # 5% tolerance

    warnings = validate_anchor(report, anchor)
    assert len(warnings) == 1  # 10% deviation > 5% tolerance
