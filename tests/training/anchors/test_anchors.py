"""Test anchor YAML data files.

Phase 4.5: Structural validation + MFU calibration output
Phase 4.5 (after phase 3): Strict MFU tolerance gating

TODO Phase 3: Convert anchors to runnable ModelSpec/SystemSpec/Strategy
and run actual estimates. Current implementation validates YAML shape.
"""
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


@pytest.mark.parametrize("yaml_file", sorted(ANCHOR_DIR.glob("*.yaml")))
def test_anchor_config_is_internally_consistent(yaml_file):
    """Verify anchor configs are internally consistent (no conflicting products)."""
    data = _load_anchor(yaml_file)
    config = data["config"]
    name = data["name"]

    tp = config.get("tp", 1)
    pp = config.get("pp", 1)
    dp = config.get("dp", 1)

    # Basic sanity: world_size should be consistent with tp * pp * dp
    # (EP excluded per current policy — see test_ep_rank_product.py)
    world_size = config.get("world_size", tp * pp * dp)
    rank_product = tp * pp * dp

    assert rank_product == world_size, (
        f"Anchor '{name}': TP*PP*DP={rank_product} != world_size={world_size}. "
        f"Internal consistency check failed."
    )


def test_anchor_validate_with_report():
    report = Report(step_time_ms=100.0, mfu=0.50, total_flops=1e12)
    anchor = Anchor(name="test", mfu=0.50, tolerance=0.15, strict_mfu_check=True)
    warnings = validate_anchor(report, anchor)
    assert len(warnings) == 0


def test_anchor_validate_fails_with_bad_report_strict():
    """Strict MFU check should fail when deviation exceeds tolerance."""
    report = Report(step_time_ms=200.0, mfu=0.20, total_flops=1e12)
    anchor = Anchor(name="test", mfu=0.50, tolerance=0.15, strict_mfu_check=True)
    warnings = validate_anchor(report, anchor)
    assert len(warnings) > 0
    assert "[STRICT]" in warnings[0]


def test_anchor_validate_calibration_mode_no_failure():
    """Calibration mode records MFU deviation but doesn't fail."""
    report = Report(step_time_ms=200.0, mfu=0.20, total_flops=1e12)
    anchor = Anchor(name="test", mfu=0.50, tolerance=0.15, strict_mfu_check=False)
    warnings = validate_anchor(report, anchor)
    # Should have warning but marked as [CALIBRATION], not [STRICT]
    assert len(warnings) > 0
    assert "[CALIBRATION]" in warnings[0]
    assert "[STRICT]" not in warnings[0]


def test_anchor_estimate_integration_placeholder():
    """TODO Phase 3: Run actual estimates for each anchor YAML.

    Phase 3 integration steps:
      1. Load anchor YAML → ModelSpec, SystemSpec, Strategy
      2. Run estimate(model, system, strategy) → Report
      3. Validate against anchor targets (with strict_mfu_check for calibrated anchors)
      4. Record estimated MFU as calibration output

    For now, this test is a placeholder documenting the intended workflow.
    """
    # Example intended workflow (to be implemented in phase 3):
    #
    # from zrt.training.io.config_loader import load_anchor_config
    # from zrt.training.search.estimator import estimate
    #
    # for yaml_file in ANCHOR_DIR.glob("*.yaml"):
    #     model, system, strategy = load_anchor_config(yaml_file)
    #     anchor_data = _load_anchor(yaml_file)
    #     anchor = Anchor(name=anchor_data["name"], **anchor_data["targets"])
    #
    #     report = estimate(model, system, strategy)
    #     warnings = validate_anchor(report, anchor)
    #
    #     # Calibration output: record estimated vs reference MFU
    #     print(f"{anchor.name}: estimated MFU={report.mfu:.4f}, "
    #           f"reference MFU={anchor.mfu:.4f}")
    #
    #     # Only fail if strict_mfu_check=True (phase 3)
    #     if anchor.strict_mfu_check:
    #         assert len(warnings) == 0, f"Anchor {anchor.name} failed strict validation"
    pass
