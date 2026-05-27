from __future__ import annotations

from typing import Any


def _to_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _to_int(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _pct(time_ms: float, step_time_ms: float) -> float:
    if step_time_ms <= 0:
        return 0.0
    return time_ms / step_time_ms


def _is_attention_op(op: dict) -> bool:
    component = str(op.get("component", "") or "").lower()
    component_group = str(op.get("component_group", "") or "").lower()
    if component == "attention" or component_group == "attention":
        return True

    name = str(op.get("name", "") or "").lower()
    kind = str(op.get("kind", "") or "").lower()
    return (
        "attn" in name
        or "attention" in name
        or kind in {"attn_core", "sparse_attn", "hca_attn", "swa_attn"}
        or kind in {"compressor_pool", "indexer_topk", "rope"}
    )


def _is_dsv4(model: Any) -> bool:
    model_type = str(getattr(model, "model_type", "") or "").lower()
    return model_type == "deepseek_v4" or bool(getattr(model, "use_v4_attn", False))


def _is_dsv32(model: Any) -> bool:
    model_type = str(getattr(model, "model_type", "") or "").lower()
    if model_type in {"deepseek_v3_2", "deepseek-v3-2", "dsv3.2", "dsv32"}:
        return True
    return (
        bool(getattr(model, "use_mla", False))
        and _to_int(getattr(model, "index_topk", 0), 0) > 0
        and not bool(getattr(model, "use_v4_attn", False))
    )


def _row(
    label: str,
    time_ms: float,
    op_count: int,
    step_time_ms: float,
    useful_compute_ms: float,
) -> dict:
    return {
        "label": label,
        "time_ms": time_ms,
        "pct_of_step": _pct(time_ms, step_time_ms),
        "pct_of_useful_compute": _pct(time_ms, useful_compute_ms),
        "op_count": op_count,
    }


def _append_if_present(
    rows: list[dict],
    label: str,
    ops: list[dict],
    step_time_ms: float,
    useful_compute_ms: float,
) -> None:
    if not ops:
        return
    rows.append(
        _row(
            label,
            sum(_to_float(op.get("total_ms", 0.0)) for op in ops),
            len(ops),
            step_time_ms,
            useful_compute_ms,
        )
    )


def build_operator_time_stats(*, model: Any, report: Any, op_dicts: list[dict]) -> list[dict]:
    """Build estimate-report operator time-share rows.

    Percentages are relative to ``report.step_time_ms`` and
    ``report.compute_time_ms``. The input ``op_dicts`` use the same shape
    produced by ``html_exporter._op_to_dict``.
    """
    step_time_ms = _to_float(getattr(report, "step_time_ms", 0.0))
    useful_compute_ms = _to_float(getattr(report, "compute_time_ms", 0.0))
    rows: list[dict] = []

    matmul_ops = [
        op for op in op_dicts
        if str(op.get("kind", "") or "").lower() in {"matmul", "lm_head"}
    ]
    _append_if_present(
        rows,
        "Matmul family total",
        matmul_ops,
        step_time_ms,
        useful_compute_ms,
    )

    if _is_dsv4(model):
        csa_ops: list[dict] = []
        hca_ops: list[dict] = []
        for op in op_dicts:
            if not _is_attention_op(op):
                continue
            layer_id = _to_int(op.get("layer_id"), -1)
            if layer_id < 0 or not hasattr(model, "get_layer_cp_type"):
                continue
            cp_type = str(model.get_layer_cp_type(layer_id)).lower()
            if cp_type == "csa":
                csa_ops.append(op)
            elif cp_type == "hca":
                hca_ops.append(op)

        swa_ops = [
            op for op in op_dicts
            if str(op.get("kind", "") or "").lower() == "swa_attn"
        ]

        _append_if_present(rows, "CSA attention block", csa_ops, step_time_ms, useful_compute_ms)
        _append_if_present(rows, "HCA attention block", hca_ops, step_time_ms, useful_compute_ms)
        _append_if_present(rows, "SWA operator", swa_ops, step_time_ms, useful_compute_ms)

    if _is_dsv32(model):
        flash_ops = [
            op for op in op_dicts
            if str(op.get("kind", "") or "").lower() == "attn_core"
        ]
        mla_ops = [op for op in op_dicts if _is_attention_op(op)]

        _append_if_present(rows, "FlashAttention", flash_ops, step_time_ms, useful_compute_ms)
        _append_if_present(rows, "MLA attention block", mla_ops, step_time_ms, useful_compute_ms)

    return rows
