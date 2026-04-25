"""Anchor validation — compare estimates against known reference values.

Phase 4.5 (current): Structural validation + MFU calibration output
Phase 4.5 (after phase 3): Strict MFU tolerance gating for calibrated anchors

TODO Phase 3: MFU tolerance should only gate anchors whose dependencies are
implemented and calibrated. For now, record MFU as calibration output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zrt.training.search.estimator import Report


@dataclass
class Anchor:
    """Reference values for a known training configuration.

    Attributes
    ----------
    name
        Anchor identifier (e.g., "llama3_70b_meta")
    step_time_ms
        Reference step time in milliseconds (optional)
    mfu
        Reference Model FLOPs Utilization (optional)
    total_flops
        Reference total FLOPs (optional)
    tolerance
        Relative tolerance for MFU gating (default 15%)
    strict_mfu_check
        If True, enforce MFU tolerance; if False, record as calibration only
        (default: False until phase 3 calibration is complete)
    """

    name: str
    step_time_ms: float | None = None
    mfu: float | None = None
    total_flops: float | None = None
    tolerance: float = 0.15  # 15% default tolerance
    strict_mfu_check: bool = False  # Phase-3 calibration flag


def validate_anchor(report: "Report", anchor: Anchor) -> list[str]:
    """Compare a Report against an Anchor and return warnings for deviations.

    Phase 4.5 behavior:
      - Always return structural warnings (step_time, mfu deviations)
      - Only fail on MFU tolerance if strict_mfu_check=True
      - Record estimated MFU as calibration output for all anchors

    Phase 4.5 (after phase 3): Enable strict_mfu_check for calibrated anchors.

    Returns
    -------
    warnings
        List of warning strings. Empty list means all checks pass.
        For non-strict mode, warnings include calibration notes.
    """
    warnings = []

    if anchor.step_time_ms is not None and anchor.step_time_ms > 0:
        deviation = abs(report.step_time_ms - anchor.step_time_ms) / anchor.step_time_ms
        if deviation > anchor.tolerance:
            warnings.append(
                f"step_time_ms: estimated={report.step_time_ms:.1f}, "
                f"anchor={anchor.step_time_ms:.1f}, "
                f"deviation={deviation:.1%} (tolerance={anchor.tolerance:.0%})"
            )

    if anchor.mfu is not None and anchor.mfu > 0:
        deviation = abs(report.mfu - anchor.mfu) / anchor.mfu
        if deviation > anchor.tolerance:
            msg = (
                f"mfu: estimated={report.mfu:.4f}, "
                f"anchor={anchor.mfu:.4f}, "
                f"deviation={deviation:.1%} (tolerance={anchor.tolerance:.0%})"
            )
            if anchor.strict_mfu_check:
                warnings.append(f"[STRICT] {msg}")
            else:
                warnings.append(f"[CALIBRATION] {msg}")

    if anchor.total_flops is not None and anchor.total_flops > 0:
        deviation = abs(report.total_flops - anchor.total_flops) / anchor.total_flops
        if deviation > anchor.tolerance:
            warnings.append(
                f"total_flops: estimated={report.total_flops:.2e}, "
                f"anchor={anchor.total_flops:.2e}, "
                f"deviation={deviation:.1%} (tolerance={anchor.tolerance:.0%})"
            )

    return warnings
