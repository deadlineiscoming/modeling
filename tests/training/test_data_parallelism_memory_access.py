import pytest
from unittest.mock import MagicMock, patch
from python.zrt.training.models.memory import memory_breakdown, MemBreakdown


class TestMemoryBreakdownDP:

    @pytest.fixture
    def mock_env(self):
        """构造基础 Mock 环境"""
        model = MagicMock()
        model.param_dtype.bytes = 2
        model.grad_dtype.bytes = 4
        model.layers = [MagicMock()] * 10
        model.total_params.return_value = 1000000

        strategy = MagicMock()
        strategy.dp = 4
        strategy.tp = 1
        strategy.pp = 1
        strategy.zero_stage = 0
        strategy.offload.pct = 0.0

        graph = MagicMock()
        return graph, model, strategy

    def test_zero_stage_3_full_sharding(self, mock_env):
        """
        验证 ZeRO-3 下 DP 对所有静态内存的分片影响。
        逻辑：weights, grads, opt_state 全部除以 strategy.dp
        """
        graph, model, strategy = mock_env
        strategy.dp = 8
        strategy.zero_stage = 3

        P = 16000  # 假设单卡参数量
        initial_opt_bytes = 32000.0

        # 拦截底层计算函数
        with patch("python.zrt.training.models.memory._params_on_rank", return_value=P), \
                patch("python.zrt.training.models.memory._optimizer_state_bytes", return_value=initial_opt_bytes), \
                patch("python.zrt.training.models.memory._activation_memory", return_value=500.0), \
                patch("python.zrt.training.models.memory._comm_buffer_memory", return_value=100.0):

            res = memory_breakdown(graph, model, None, strategy)

            # 权重校验: (16000 * 2) // 8 = 4000
            assert res.weights == pytest.approx(4000.0)
            # 梯度校验: (16000 * 4) // 8 = 8000
            assert res.grads == pytest.approx(8000.0)
            # 优化器校验: 32000 // 8 = 4000
            assert res.opt_state == pytest.approx(4000.0)
            # 激活值和 Buffer 不受 ZeRO 影响
            assert res.activations == 500.0
            assert res.comm_buffers == 100.0

    def test_zero_stage_1_vs_2(self, mock_env):
        """
        对比验证 ZeRO-1 和 ZeRO-2 的差异。
        ZeRO-1 只分片 OptState；ZeRO-2 分片 OptState 和 Grads。
        """
        graph, model, strategy = mock_env
        strategy.dp = 2
        P = 1000
        opt_total = 4000.0

        with patch("python.zrt.training.models.memory._params_on_rank", return_value=P), \
                patch("python.zrt.training.models.memory._optimizer_state_bytes", return_value=opt_total), \
                patch("python.zrt.training.models.memory._activation_memory", return_value=0), \
                patch("python.zrt.training.models.memory._comm_buffer_memory", return_value=0):
            # 测试 ZeRO-1
            strategy.zero_stage = 1
            res_z1 = memory_breakdown(graph, model, None, strategy)
            assert res_z1.opt_state == 2000.0
            assert res_z1.grads == 4000.0  # 不分片 (1000 * 4)

            # 测试 ZeRO-2
            strategy.zero_stage = 2
            res_z2 = memory_breakdown(graph, model, None, strategy)
            assert res_z2.opt_state == 2000.0
            assert res_z2.grads == 2000.0  # 分片 (4000 // 2)
