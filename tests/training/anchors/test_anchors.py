"""Test anchor YAML data files."""
from __future__ import annotations

import yaml
import pytest
from pathlib import Path

from zrt.training.anchor.validate import Anchor, validate_anchor
from zrt.training.search.estimator import Report


ANCHOR_DIR = Path(__file__).parent


def _load_anchor(yaml_path: Path) -> dict:
    return yaml.safe_load(yaml_path.read_text())


@pytest.mark.parametrize("yaml_file", sorted(ANCHOR_DIR.glob("*.yaml")))
def test_anchor_yaml_is_valid(yaml_file):
    data = _load_anchor(yaml_file)
    assert "name" in data
    assert "targets" in data
    anchor = Anchor(name=data["name"], **data["targets"])
    assert anchor.name


@pytest.mark.parametrize("yaml_file", sorted(ANCHOR_DIR.glob("*.yaml")))
def test_anchor_yaml_has_config(yaml_file):
    data = _load_anchor(yaml_file)
    assert "config" in data
    config = data["config"]
    assert "tp" in config
    assert "dp" in config


def test_anchor_validate_with_report():
    report = Report(step_time_ms=100.0, mfu=0.50, total_flops=1e12)
    anchor = Anchor(name="test", mfu=0.50, tolerance=0.15)
    warnings = validate_anchor(report, anchor)
    assert len(warnings) == 0


def test_anchor_validate_fails_with_bad_report():
    report = Report(step_time_ms=200.0, mfu=0.20, total_flops=1e12)
    anchor = Anchor(name="test", mfu=0.50, tolerance=0.15)
    warnings = validate_anchor(report, anchor)
    assert len(warnings) > 0
