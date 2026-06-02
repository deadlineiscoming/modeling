"""Metric-based filtering and ranking for parallel-strategy search.

These helpers are intentionally dependency-light (stdlib only) so both the
heavyweight ``training_search_util`` script and the FastAPI ``/search`` endpoint
can share them — the server environment does not ship pandas/tqdm.

A *filter* is a constraint ``{"metric": <name>, "op": <comparator>, "value": <float>}``.
A config is kept only when it satisfies **every** constraint. *Ranking* sorts
the survivors by a single named metric in either direction.

The metric names below are the public surface exposed by the API/UI. ``memory_gb``
uses the OOM-relevant *peak* footprint, so ``memory_gb <= 80`` reads as
"fits in 80 GB".
"""

from __future__ import annotations

import operator
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# User-facing metric name → accessor on a TrainingReport.
SEARCH_METRIC_ACCESSORS: Dict[str, Callable[[Any], Optional[float]]] = {
    "step_time_ms":    lambda r: r.step_time_ms,
    "tokens_per_sec":  lambda r: r.tokens_per_sec,
    "mfu":             lambda r: r.mfu,
    "hfu":             lambda r: r.hfu,
    "bubble_fraction": lambda r: r.bubble_fraction,
    "memory_gb":       lambda r: (r.memory.peak_overall / 1e9) if getattr(r, "memory", None) else None,
}

# Public list of selectable metrics (stable order for UI rendering).
SEARCH_METRICS: Tuple[str, ...] = tuple(SEARCH_METRIC_ACCESSORS.keys())

FILTER_OPS: Dict[str, Callable[[float, float], bool]] = {
    "<":  operator.lt,
    "<=": operator.le,
    ">":  operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def report_metric(report: Any, metric: str) -> Optional[float]:
    """Read a named search metric off a TrainingReport.

    Returns ``None`` for an unknown metric or when the value is unavailable
    (e.g. ``memory_gb`` with no memory breakdown).
    """
    accessor = SEARCH_METRIC_ACCESSORS.get(metric)
    if accessor is None:
        return None
    try:
        return accessor(report)
    except Exception:
        return None


def report_passes_filters(report: Any, filters: Optional[Iterable[Dict[str, Any]]]) -> bool:
    """True when ``report`` satisfies every ``{metric, op, value}`` constraint.

    Fails closed: an unknown metric/operator, a missing metric value, or a
    non-numeric threshold excludes the config rather than silently passing.
    Empty/None ``filters`` accepts everything.
    """
    for f in filters or []:
        op = FILTER_OPS.get(str(f.get("op")))
        val = report_metric(report, str(f.get("metric")))
        if op is None or val is None:
            return False
        try:
            threshold = float(f.get("value"))
        except (TypeError, ValueError):
            return False
        if not op(val, threshold):
            return False
    return True


def _sort_key(metric: str, ascending: bool) -> Callable[[Any], float]:
    """Sort key extractor; configs with a missing metric always sort last."""
    inf = float("inf")

    def key(report: Any) -> float:
        v = report_metric(report, metric)
        if v is None:
            return inf
        return v if ascending else -v

    return key


def sort_reports_with_strategies(
    pairs: List[Tuple[Any, Any]],
    sort_by: str = "tokens_per_sec",
    ascending: bool = False,
) -> List[Tuple[Any, Any]]:
    """Sort ``(strategy, report)`` pairs by a named metric (missing values last)."""
    key = _sort_key(sort_by, ascending)
    return sorted(pairs, key=lambda pair: key(pair[1]))
