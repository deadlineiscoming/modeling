"""Anchor validation — compare estimates against known reference values."""
from __future__ import annotations

from dataclasses import dataclass

from zrt.training.search.estimator import Report


@dataclass
class Anchor:
    """Reference values for a known training configuration."""

    name: str
    step_time_ms: float | None = None
    mfu: float | None = None
    total_flops: float | None = None
    tolerance: float = 0.15  # 15% default tolerance


def validate_anchor(report: Report, anchor: Anchor) -> list[str]:
    """Compare a Report against an Anchor and return warnings for deviations.

    Returns a list of warning strings. Empty list means all checks pass.
    A warning is emitted when the relative deviation exceeds anchor.tolerance.
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
            warnings.append(
                f"mfu: estimated={report.mfu:.4f}, "
                f"anchor={anchor.mfu:.4f}, "
                f"deviation={deviation:.1%} (tolerance={anchor.tolerance:.0%})"
            )

    if anchor.total_flops is not None and anchor.total_flops > 0:
        deviation = abs(report.total_flops - anchor.total_flops) / anchor.total_flops
        if deviation > anchor.tolerance:
            warnings.append(
                f"total_flops: estimated={report.total_flops:.2e}, "
                f"anchor={anchor.total_flops:.2e}, "
                f"deviation={deviation:.1%} (tolerance={anchor.tolerance:.0%})"
            )

    return warnings
