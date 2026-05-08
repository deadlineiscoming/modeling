"""Per-op analytical FLOPs model.

Returns raw cost per op. Recompute multiplier applied by the stage composer.
Reference: Calculon (Isaev et al. SC'23), Korthikanti et al. 2022.
"""

from __future__ import annotations

from dataclasses import dataclass

from zrt.training.ir.training_graph import Graph, Op
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.strategy import Strategy


@dataclass
class OpCost:
    fwd_flops: float = 0.0
    dx_flops: float = 0.0
    dw_flops: float = 0.0
    fwd_bytes: float = 0.0   # memory-bound ops: byte traffic
    dx_bytes: float = 0.0
    dw_bytes: float = 0.0
    bound: str = "compute"   # "compute" | "memory"


def op_cost(op: Op, model: ModelSpec) -> OpCost:
    """Compute raw cost per op. Bound determines the cost model used."""
    if op.kind == "matmul":
        return _matmul_cost(op)
    if op.kind == "attn_core":
        return _attn_cost(op, model)
    if op.kind == "mhc_pre":
        return _mhc_pre_cost(op)
    if op.kind == "mhc_post":
        return _mhc_post_cost(op)
    if op.kind == "mhc_head":
        return _mhc_head_cost(op)
    if op.kind == "hc_expand":
        return _memory_bound_cost(op)
    if op.kind in ("ln", "softmax", "rope", "swiglu", "add"):
        return _memory_bound_cost(op)
    if op.kind in ("embed", "lm_head"):
        return _matmul_cost(op)
    if op.kind == "compressor_pool":
        return _compressor_pool_cost(op)
    if op.kind == "indexer_topk":
        return _indexer_topk_cost(op)
    if op.kind == "hash_route":
        return OpCost()  # table lookup, negligible FLOPs
    # Unknown ops: zero cost
    return OpCost()


def _matmul_cost(op: Op) -> OpCost:
    m = op.meta.get("m", 0)
    n = op.meta.get("n_local", op.meta.get("n", 0))
    k = op.meta.get("k_local", op.meta.get("k", 0))
    fwd = 2.0 * m * n * k

    # Apply fwd_multiplier if present (e.g., for MoE routed expert FFNs)
    fwd_multiplier = op.meta.get("fwd_multiplier", 1.0)
    fwd = fwd * fwd_multiplier

    return OpCost(
        fwd_flops=fwd,
        dx_flops=fwd,     # dX: 2*m*n*k
        dw_flops=fwd,     # dW: 2*m*n*k
    )


def _attn_cost(op: Op, model: ModelSpec) -> OpCost:
    b = op.meta.get("b", 1)
    s = op.meta.get("s", 0)
    h = op.meta.get("heads", 0)
    d = op.meta.get("head_dim", 0)
    causal = op.meta.get("causal", True)

    # V4 attention variants — identified by metadata:
    sparse_topk = op.meta.get("sparse_topk", 0)
    compress_ratio = op.meta.get("compress_ratio", 0)
    swa_window = op.meta.get("swa_window", 0)

    if sparse_topk > 0:
        # CSA: sparse attention over topk compressed KV + sliding window
        effective_len = sparse_topk + swa_window
        fwd = 2.0 * b * s * effective_len * h * d
    elif compress_ratio > 0:
        # HCA: dense attention on compressed KV (seq/ratio) + sliding window
        compressed_len = max(1, s // compress_ratio)
        effective_len = compressed_len + swa_window
        fwd = 2.0 * b * s * effective_len * h * d
    elif swa_window > 0:
        # SWA-only: pure sliding window attention
        fwd = 2.0 * b * s * swa_window * h * d
    else:
        # Standard / MLA: full causal attention (possibly with compression ratio)
        compression_ratio = _attn_compression_ratio(
            op.meta.get("attn_compression_ratio", model.attn_compression_ratio)
        )
        mult = 2.0 if causal else 4.0
        fwd = mult * b * s * s * h * d * compression_ratio

    # Ring-CP: multiply by cp_tiles to account for multiple rounds
    if op.meta.get("cp_tiles", 0) > 1:
        fwd *= op.meta.get("cp_tiles", 1)

    dx = 2.5 * fwd

    return OpCost(
        fwd_flops=fwd,
        dx_flops=dx,
        dw_flops=0.0,
    )


def _attn_compression_ratio(value: float) -> float:
    ratio = float(value)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"attn_compression_ratio must be in (0, 1], got {value}")
    return ratio


def _mhc_pre_cost(op: Op) -> OpCost:
    """Hyper-Connections pre-mix: mixes-Linear + sinkhorn iters + weighted sum.

    Math:
      mixes  = x[b,s,hc*h] @ hc_fn[hc*h, mix_hc]      → 2·b·s·(hc·h)·mix_hc  (compute-bound)
      sink   = sinkhorn_iters · O(b·s·mix_hc·hc)     elementwise              (memory-ish)
      sum    = pre[b,s,hc] · x[b,s,hc,h]              → 2·b·s·hc·h            (weighted sum)
    """
    b = op.meta.get("b", 1)
    s = op.meta.get("s", 0)
    h = op.meta.get("h", 0)
    hc = op.meta.get("hc", 1)
    mix = op.meta.get("mix_hc", (2 + hc) * hc)
    it = op.meta.get("sinkhorn_iters", 20)

    fwd_lin = 2.0 * b * s * (hc * h) * mix
    fwd_sink = float(it * b * s * mix * hc) * 4.0
    fwd_sum = float(b * s * hc * h) * 2.0
    fwd = fwd_lin + fwd_sink + fwd_sum

    return OpCost(
        fwd_flops=fwd,
        dx_flops=2.5 * fwd,
        dw_flops=fwd_lin,  # only the mixes Linear has trainable params
    )


def _mhc_post_cost(op: Op) -> OpCost:
    """Hyper-Connections post-mix: post·x + Σ comb·residual.

    Math:
      post · x:           b·s·hc·h            (broadcast multiply)
      comb · residual:    b·s·hc·hc·h         (full hc×hc combination)
      sum over hc:        b·s·hc·h
    No trainable parameters in this op (post / comb come from mhc_pre).
    """
    b = op.meta.get("b", 1)
    s = op.meta.get("s", 0)
    h = op.meta.get("h", 0)
    hc = op.meta.get("hc", 1)

    fwd = float(b * s * hc * h) * 2.0 + float(b * s * hc * hc * h) * 2.0

    return OpCost(
        fwd_flops=fwd,
        dx_flops=2.5 * fwd,
        dw_flops=0.0,
    )


def _mhc_head_cost(op: Op) -> OpCost:
    """Final HC mix-down before final_ln (no sinkhorn, no comb).

    Math:
      mixes  = x[b,s,hc*h] @ hc_head_fn[hc*h, hc]   → 2·b·s·hc·h·hc
      sum    = pre[b,s,hc] · x[b,s,hc,h]            → 2·b·s·hc·h
    """
    b = op.meta.get("b", 1)
    s = op.meta.get("s", 0)
    h = op.meta.get("h", 0)
    hc = op.meta.get("hc", 1)
    mix = op.meta.get("mix_hc", hc)

    fwd_lin = 2.0 * b * s * (hc * h) * mix
    fwd_sum = float(b * s * hc * h) * 2.0
    fwd = fwd_lin + fwd_sum

    return OpCost(
        fwd_flops=fwd,
        dx_flops=2.5 * fwd,
        dw_flops=fwd_lin,
    )


def _memory_bound_cost(op: Op) -> OpCost:
    bytes_fwd = op.meta.get("bytes_fwd", 0.0)
    # Bwd byte traffic ≈ fwd (read activations + write gradients)
    bytes_bwd = bytes_fwd * 1.5  # conservative: read input + write grad

    return OpCost(
        fwd_bytes=bytes_fwd,
        dx_bytes=bytes_bwd,
        dw_bytes=0.0,
        bound="memory",
    )


def _compressor_pool_cost(op: Op) -> OpCost:
    """KV compressor gated pooling: softmax + weighted sum over compression windows."""
    s = op.meta.get("s", 0)
    m = op.meta.get("m", 4)
    coff = op.meta.get("coff", 1)
    d = op.meta.get("d_local", op.meta.get("d", 0))
    # softmax over coff*m elements × (s/m) groups, then weighted sum
    fwd_flops = 4.0 * (s // m) * coff * m * d
    bytes_fwd = op.meta.get("bytes_fwd", s * d * 4)  # read kv + write compressed
    return OpCost(fwd_flops=fwd_flops, dx_flops=fwd_flops,
                  fwd_bytes=bytes_fwd, dx_bytes=bytes_fwd * 1.5, bound="compute")


def _indexer_topk_cost(op: Op) -> OpCost:
    """Indexer scoring: einsum(q, kv) + ReLU + weighted sum + top-k."""
    s = op.meta.get("s", 0)
    ih = op.meta.get("ih_local", op.meta.get("ih", 0))
    id_ = op.meta.get("id", 0)
    topk = op.meta.get("topk", 0)
    # kv_len: full seq for V3.2, compressed seq//m for V4-CSA
    kv_len = op.meta.get("kv_len", s)
    einsum_flops = 2.0 * s * kv_len * ih * id_
    # ReLU + weighted sum + topk: memory-bound
    bytes_fwd = op.meta.get("bytes_fwd", s * ih * id_ * 4)
    return OpCost(
        fwd_flops=einsum_flops,
        dx_flops=2.0 * einsum_flops,
        dw_flops=0.0,
        fwd_bytes=bytes_fwd,
        bound="compute",
    )


def total_training_flops(
    graph: Graph, model: ModelSpec, strategy: Strategy,
) -> float:
    """Total FLOPs per training step (forward + backward).

    Standard transformer: 6 * total_params * tokens (6P rule).
    With recompute: adds extra forward for recomputed ops.
    """
    total = 0.0
    for op in graph.ops:
        cost = op_cost(op, model)
        if cost.bound == "compute":
            # Forward + dx + dw = 3× fwd_flops (2mnk × 3 = 6mnk for matmul)
            total += cost.fwd_flops + cost.dx_flops + cost.dw_flops
        # Memory-bound ops contribute negligible FLOPs

    # Scale by microbatch count
    M = strategy.num_microbatches()
    total *= M

    return total


def recompute_overhead_flops(
    graph: Graph, model: ModelSpec, strategy: Strategy,
) -> float:
    """Extra FLOPs from recomputing forward activations during backward pass.

    Selective recompute re-runs the forward for specific ops (typically attention)
    during backward. Full recompute re-runs the entire forward pass.

    Respects per-layer policies: only ops belonging to a layer whose kind
    appears in ``RecomputePolicy.per_layer`` are counted.

    Returns the additional FLOPs (not the total).
    """
    rc = strategy.recompute
    if not rc.per_layer:
        return 0.0

    extra = 0.0
    for op in graph.ops:
        # Look up the layer kind for this op
        if op.layer_id < 0 or op.layer_id >= len(model.layers):
            continue
        lk = model.layers[op.layer_id].value
        cats = rc.per_layer.get(lk)
        if not cats:
            continue

        op_cats = _op_recompute_categories(op)
        if "full" in cats or (op_cats & cats):
            cost = op_cost(op, model)
            if cost.bound == "compute":
                extra += cost.fwd_flops

    M = strategy.num_microbatches()
    return extra * M


def _op_recompute_categories(op: Op) -> set[str]:
    """Map an op to its recompute category set."""
    if op.kind == "attn_core":
        return {"attn"}
    if op.kind == "matmul":
        name = op.name.lower()
        if any(k in name for k in ("qkv", "q_a_proj", "q_b_proj", "kv_a_proj",
                                    "kv_b_proj", "o_proj", "wq_a", "wq_b",
                                    "wkv", "wo_a", "wo_b")):
            return {"attn"}
        if "up_proj" in name or "gate_proj" in name or "down_proj" in name:
            return {"ffn_swiglu"}
        return set()
    if op.kind in ("compressor_pool", "indexer_topk"):
        return {"attn"}
    if op.kind == "swiglu":
        return {"ffn_swiglu"}
    if op.kind == "ln":
        return {"ln"}
    if op.kind in ("mhc_pre", "mhc_post", "mhc_head"):
        return {"hc"}
    return set()
