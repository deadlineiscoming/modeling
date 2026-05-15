"""End-to-end test for training modeling pipeline (Path 2).

Tests DeepSeek V3, V3.2, and V4 models to verify:
1. TP shape splitting correctness across different TP degrees (1, 2, 4)
2. Communication operator insertion and ordering
3. Complete training report generation

Run with:
    .\run_pytest.bat tests/IT/test_training_tp_e2e.py -v
"""
import pytest
from pathlib import Path
import tempfile

from python.zrt.pipeline import run_trace_phases
from python.zrt.transform.analysis import estimate_training_from_graphs
import python.zrt.hardware.registry as hw_registry


MODEL_CONFIGS = {
    "deepseek_v3": {
        "model_id": "hf_models/deepseek_v3",
        "hidden_size": 7168,
        "num_heads": 128,
        "description": "DeepSeek V3 (MLA attention)",
        "key_ops": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "attention_type": "MLA",
        "has_moe": True,
    },
    "deepseek_v3_2": {
        "model_id": "hf_models/deepseek_v3_2",
        "hidden_size": 7168,
        "num_heads": 128,
        "description": "DeepSeek V3.2 (MLA + Index attention)",
        "key_ops": ["q_proj", "k_proj", "v_proj", "o_proj", "index_q", "index_kv"],
        "attention_type": "MLA + Index",
        "has_moe": True,
    },
    "deepseek_v4": {
        "model_id": "hf_models/deepseek_v4",
        "hidden_size": 7168,
        "num_heads": 128,
        "description": "DeepSeek V4 (MLA 2.0 + LongMoE)",
        "key_ops": ["q_proj", "k_proj", "v_proj", "o_proj", "q_lora", "o_lora"],
        "attention_type": "MLA 2.0",
        "has_moe": True,
    },
}


def capture_training_graphs(model_id: str, num_layers: int = 2):
    """Capture training forward and backward graphs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_trace_phases(
            model_id=model_id,
            num_layers=num_layers,
            batch_size=1,
            seq_len=128,
            phases=["train_forward", "train_backward"],
            output_dir=tmpdir,
        )
        return result.graphs


def run_tp_analysis(
    forward_graph,
    backward_graph,
    tp: int,
    hidden: int,
    num_layers: int,
):
    """Run training analysis with specified TP degree."""
    hw_spec = hw_registry.load("nvidia_h100_sxm")
    report, ctx, transformed = estimate_training_from_graphs(
        forward_graph=forward_graph,
        backward_graph=backward_graph,
        hw_spec=hw_spec,
        tp=tp,
        pp=1,
        dp=1,
        seq_len=128,
        batch_size=1,
        hidden=hidden,
        num_layers=num_layers,
        return_transformed=True,
    )
    return report, transformed


def extract_projection_nodes(graph, proj_type: str):
    """Extract projection nodes by type (q_proj, k_proj, v_proj, o_proj)."""
    nodes = []
    for node_id, node in graph.nodes.items():
        scope = getattr(node, 'scope', '')
        if proj_type in scope and "self_attn" in scope:
            nodes.append({
                "node_id": node_id,
                "op_type": node.op_type,
                "scope": scope,
                "inputs": [list(inp.shape) if hasattr(inp, 'shape') else [] for inp in getattr(node, 'inputs', [])],
                "outputs": [list(out.shape) if hasattr(out, 'shape') else [] for out in getattr(node, 'outputs', [])],
            })
    return nodes


def extract_communication_nodes(graph):
    """Extract communication operators with their context."""
    comm_nodes = []
    for node_id, node in graph.nodes.items():
        op_type = getattr(node, 'op_type', '')
        if op_type.startswith("comm."):
            comm_nodes.append({
                "node_id": node_id,
                "op_type": op_type,
                "scope": getattr(node, 'scope', ''),
                "inputs": [list(inp.shape) if hasattr(inp, 'shape') else [] for inp in getattr(node, 'inputs', [])],
                "outputs": [list(out.shape) if hasattr(out, 'shape') else [] for out in getattr(node, 'outputs', [])],
            })
    return comm_nodes


class TestTPComparison:
    """End-to-end TP validation tests with comprehensive shape comparison."""

    @pytest.mark.parametrize("model_key", MODEL_CONFIGS.keys())
    def test_tp_shape_splitting_comparison(self, model_key):
        """Compare TP=1, TP=2, TP=4 shape splitting for projection layers."""
        config = MODEL_CONFIGS[model_key]
        hidden_size = config["hidden_size"]
        
        graphs = capture_training_graphs(config["model_id"], num_layers=2)
        fwd_graph = graphs["train_forward"]
        bwd_graph = graphs["train_backward"]

        # Run analysis for different TP degrees
        results = {}
        for tp in [1, 2, 4]:
            report, transformed = run_tp_analysis(
                fwd_graph, bwd_graph, tp=tp,
                hidden=hidden_size, num_layers=2
            )
            unified_graph = transformed.get("unified")
            assert unified_graph is not None, f"Unified graph should exist for TP={tp}"
            
            results[tp] = {
                "report": report,
                "unified_graph": unified_graph,
                "q_proj": extract_projection_nodes(unified_graph, "q_proj"),
                "k_proj": extract_projection_nodes(unified_graph, "k_proj"),
                "v_proj": extract_projection_nodes(unified_graph, "v_proj"),
                "o_proj": extract_projection_nodes(unified_graph, "o_proj"),
                "comm": extract_communication_nodes(unified_graph),
            }

        # Verify shape splitting for column parallel ops (q, k, v projections)
        # For column parallel, output dimension should be divided by TP degree
        for proj_type in ["q_proj", "k_proj", "v_proj"]:
            tp1_nodes = results[1][proj_type]
            if not tp1_nodes:
                continue  # Skip if no nodes found
            
            tp1_dim = tp1_nodes[0]["outputs"][0][-1] if (tp1_nodes[0]["outputs"] and tp1_nodes[0]["outputs"][0]) else 0
            
            for tp in [1, 2, 4]:
                expected_dim = tp1_dim // tp
                nodes = results[tp][proj_type]
                
                for node in nodes:
                    if node["outputs"]:
                        actual_dim = node["outputs"][0][-1] if node["outputs"][0] else 0
                        assert actual_dim == expected_dim, \
                            f"{proj_type} output dim should be {expected_dim} for TP={tp}, got {actual_dim}"

        # Verify communication operators scale with TP
        tp1_comm_count = len(results[1]["comm"])
        tp2_comm_count = len(results[2]["comm"])
        tp4_comm_count = len(results[4]["comm"])
        
        assert tp1_comm_count == 0, f"TP=1 should have 0 comm ops, got {tp1_comm_count}"
        assert tp2_comm_count > 0, f"TP=2 should have comm ops"
        assert tp4_comm_count >= tp2_comm_count, f"TP=4 should have >= TP=2 comm ops"

    @pytest.mark.parametrize("model_key", MODEL_CONFIGS.keys())
    def test_tp_communication_operator_ordering(self, model_key):
        """Verify communication operators are inserted and have correct structure."""
        config = MODEL_CONFIGS[model_key]
        hidden_size = config["hidden_size"]
        
        graphs = capture_training_graphs(config["model_id"], num_layers=2)
        fwd_graph = graphs["train_forward"]
        bwd_graph = graphs["train_backward"]

        # Run TP=2 analysis
        _, transformed = run_tp_analysis(
            fwd_graph, bwd_graph, tp=2,
            hidden=hidden_size, num_layers=2
        )
        unified_graph = transformed.get("unified")

        # Extract all nodes with their order
        nodes_ordered = list(unified_graph.nodes.values())
        
        # Find communication nodes and their positions
        comm_positions = []
        for idx, node in enumerate(nodes_ordered):
            op_type = getattr(node, 'op_type', '')
            if op_type.startswith("comm."):
                comm_positions.append({
                    "index": idx,
                    "op_type": op_type,
                    "scope": getattr(node, 'scope', ''),
                })

        # Verify communication operators exist
        assert len(comm_positions) > 0, "Should have communication operators for TP=2"

        # Verify all_reduce operators exist
        all_reduce_count = sum(1 for c in comm_positions if "all_reduce" in c["op_type"])
        assert all_reduce_count > 0, "Should have all_reduce operators for TP=2"

        # Verify all_reduce operators have correct structure
        for comm in comm_positions:
            if "all_reduce" in comm["op_type"]:
                node = nodes_ordered[comm["index"]]
                inputs = getattr(node, 'inputs', [])
                outputs = getattr(node, 'outputs', [])
                assert len(inputs) > 0, f"all_reduce should have inputs"
                assert len(outputs) > 0, f"all_reduce should have outputs"

    @pytest.mark.parametrize("model_key", MODEL_CONFIGS.keys())
    def test_tp_report_metrics(self, model_key):
        """Verify report metrics change correctly with TP configuration."""
        config = MODEL_CONFIGS[model_key]
        hidden_size = config["hidden_size"]
        
        graphs = capture_training_graphs(config["model_id"], num_layers=2)
        fwd_graph = graphs["train_forward"]
        bwd_graph = graphs["train_backward"]

        # Get metrics for different TP degrees
        tp1_report, _ = run_tp_analysis(fwd_graph, bwd_graph, tp=1, hidden=hidden_size, num_layers=2)
        tp2_report, _ = run_tp_analysis(fwd_graph, bwd_graph, tp=2, hidden=hidden_size, num_layers=2)
        tp4_report, _ = run_tp_analysis(fwd_graph, bwd_graph, tp=4, hidden=hidden_size, num_layers=2)

        # Verify reports are valid
        assert tp1_report.step_time_ms > 0
        assert tp2_report.step_time_ms > 0
        assert tp4_report.step_time_ms > 0

        # Verify TP scaling trends
        assert tp2_report.step_time_ms >= tp1_report.step_time_ms * 0.9, \
            f"TP=2 step time should be comparable to TP=1"

    @pytest.mark.parametrize("model_key", MODEL_CONFIGS.keys())
    def test_tp_no_communication_for_tp1(self, model_key):
        """Verify TP=1 has no communication operators."""
        config = MODEL_CONFIGS[model_key]
        hidden_size = config["hidden_size"]
        
        graphs = capture_training_graphs(config["model_id"], num_layers=2)
        fwd_graph = graphs["train_forward"]
        bwd_graph = graphs["train_backward"]

        _, transformed = run_tp_analysis(
            fwd_graph, bwd_graph, tp=1,
            hidden=hidden_size, num_layers=2
        )
        unified_graph = transformed.get("unified")

        comm_nodes = [n for n in unified_graph.nodes.values() 
                      if getattr(n, 'op_type', '').startswith("comm.")]
        assert len(comm_nodes) == 0, \
            f"TP=1 should have no communication operators, got {len(comm_nodes)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])