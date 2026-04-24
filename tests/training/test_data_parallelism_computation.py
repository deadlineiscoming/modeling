import pytest
from unittest.mock import MagicMock, patch
from python.zrt.training.models.flops import total_training_flops


class TestTrainingFlopsDP:

    @pytest.fixture
    def mock_env(self):
        model = MagicMock()
        graph = MagicMock()
        # 构造两个算子：一个计算密集型，一个访存密集型
        op_compute = MagicMock()
        op_memory = MagicMock()
        graph.ops = [op_compute, op_memory]

        strategy = MagicMock()
        strategy.global_batch = 64
        strategy.micro_batch = 2
        strategy.dp = 4
        # num_microbatches = 64 // (2 * 4) = 8

        return graph, model, strategy, op_compute, op_memory

    def test_flops_scaling_with_dp(self, mock_env):
        """验证 DP 增加时，单卡的总 FLOPs 按比例减少（因为 microbatches 变少）"""
        graph, model, strategy, op_c, op_m = mock_env

        # 1. 模拟算子的开销
        def side_effect_op_cost(op, m):
            if op == op_c:
                # 假设 forward = 100, dx = 100, dw = 100, 总计 300
                return MagicMock(bound="compute", fwd_flops=100.0, dx_flops=100.0, dw_flops=100.0)
            return MagicMock(bound="memory", fwd_flops=0.0)  # 访存算子 FLOPs 为 0

        with patch("python.zrt.training.models.flops.op_cost", side_effect=side_effect_op_cost):
            # --- 场景 A: DP = 4 ---
            strategy.dp = 4
            strategy.num_microbatches.return_value = 8  # 64 / (2*4)
            flops_dp4 = total_training_flops(graph, model, strategy)

            # 预期: (300) * 8 = 2400
            assert flops_dp4 == 2400.0

            # --- 场景 B: DP 翻倍到 8 ---
            strategy.dp = 8
            strategy.num_microbatches.return_value = 4  # 64 / (2*8)
            flops_dp8 = total_training_flops(graph, model, strategy)

            # 预期: (300) * 4 = 1200
            # 验证 DP 翻倍，单卡 FLOPs 减半
            assert flops_dp8 == flops_dp4 / 2

    def test_flops_zero_global_batch(self, mock_env):
        graph, model, strategy, op_c, op_m = mock_env  # 拿到两个算子

        strategy.num_microbatches.return_value = 1

        # 使用 side_effect 根据不同的算子返回不同的 cost
        def side_effect(op, m):
            if op == op_c:
                return MagicMock(bound="compute", fwd_flops=100.0, dx_flops=100.0, dw_flops=100.0)
            return MagicMock(bound="memory", fwd_flops=0.0)  # 另一个算子不贡献 FLOPs

        with patch("python.zrt.training.models.flops.op_cost", side_effect=side_effect):
            flops = total_training_flops(graph, model, strategy)
            # 此时：300 (op_c) + 0 (op_m) = 300
            assert flops == 300.0

    def test_only_compute_ops_contribute(self, mock_env):
        """确保只有 bound='compute' 的算子被计入 FLOPs"""
        graph, model, strategy, op_c, op_m = mock_env

        strategy.num_microbatches.return_value = 1

        def side_effect_op_cost(op, m):
            if op == op_c:
                return MagicMock(bound="compute", fwd_flops=100.0, dx_flops=100.0, dw_flops=100.0)
            else:
                # 即使内存受限算子有 fwd_flops，也不应计入
                return MagicMock(bound="memory", fwd_flops=9999.0)

        with patch("python.zrt.training.models.flops.op_cost", side_effect=side_effect_op_cost):
            flops = total_training_flops(graph, model, strategy)
            assert flops == 300.0  # 证明 9999 没有被加上去