from __future__ import annotations

from zrt.training.io.html_exporter import _op_detail
from zrt.training.ir.training_graph import Op, Tensor
from zrt.training.models.flops import OpCost, op_cost
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec


def _model() -> ModelSpec:
    return ModelSpec(
        hidden=4096,
        ffn=16384,
        num_heads=32,
        num_kv_heads=32,
        head_dim=128,
        vocab=32000,
        seq_len=4096,
        layers=[LayerKind.DENSE],
    )


def test_matmul_formula_uses_local_sharded_dimensions():
    op = Op(
        name="L0.wq_a",
        kind="matmul",
        inputs=[
            Tensor(
                name="x_ln1",
                shape_logical=(4096, 4096),
                shape_local=(2048, 4096),
                dtype=Dtype.BF16,
                is_activation=True,
            )
        ],
        outputs=[
            Tensor(
                name="qr",
                shape_logical=(4096, 1024),
                shape_local=(2048, 128),
                dtype=Dtype.BF16,
                is_activation=True,
            )
        ],
        meta={"m": 4096, "n": 1024, "k": 4096},
    )

    detail = _op_detail(op, op_cost(op, _model()))

    assert "2048" in detail["fwd_formula"]
    assert "128" in detail["fwd_formula"]
    assert "1024" not in detail["fwd_formula"]
    assert "1024" not in detail["fwd_bytes_formula"]


def test_meta_authoritative_matmul_formula_keeps_local_meta_dimensions():
    op = Op(
        name="L0.routed_expert_ffn",
        kind="matmul",
        inputs=[
            Tensor(
                name="x_ln2",
                shape_logical=(4096, 7168),
                shape_local=(4096, 7168),
                dtype=Dtype.BF16,
                is_activation=True,
            )
        ],
        outputs=[
            Tensor(
                name="routed_ffn_out",
                shape_logical=(4096, 7168),
                shape_local=(4096, 224),
                dtype=Dtype.BF16,
                is_activation=True,
            )
        ],
        meta={
            "m": 4096,
            "n": 7168,
            "k": 7168,
            "k_local": 224,
            "fwd_multiplier": 18,
            "fused_weight_dims": True,
        },
    )

    detail = _op_detail(op, op_cost(op, _model()))

    assert "4096" in detail["fwd_formula"]
    assert "224" in detail["fwd_formula"]
    assert "7168" in detail["fwd_formula"]
    assert "1024" not in detail["fwd_formula"]


def test_formula_cells_do_not_look_like_excel_formulas():
    ops_and_costs = [
        (
            Op(name="L0.mhc_pre_attn", kind="mhc_pre"),
            OpCost(fwd_bytes=26_460_000, dx_bytes=39_690_000),
        ),
        (
            Op(name="L0.comp_pool", kind="compressor_pool", meta={"s": 4096, "m": 4, "coff": 1, "d": 512}),
            OpCost(fwd_cube_flops=2_097_152, dx_cube_flops=2_097_152, fwd_bytes=1_000_000, dx_bytes=1_500_000),
        ),
    ]

    for op, cost in ops_and_costs:
        detail = _op_detail(op, cost)
        formulas = (
            detail["fwd_formula"],
            detail["bwd_formula"],
            detail["fwd_bytes_formula"],
            detail["bwd_bytes_formula"],
        )
        assert all(not formula.startswith("=") for formula in formulas)
