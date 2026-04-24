import pytest
from unittest.mock import MagicMock, patch


from zrt.training.ir.graph import Collective, Graph
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import NetTier, SystemSpec
from zrt.training.spec.dtype import Dtype
from python.zrt.training.models.comm import total_comm_time


class TestDataParallelCommAnalysis:

    @pytest.fixture
    def mock_env(self):
        """构造符合定义的入参对象"""
        # 1. ModelSpec: 定义模型参数和数据类型
        model = ModelSpec(
            hidden=4096, ffn=11008, num_heads=32, num_kv_heads=32, head_dim=128,
            vocab=32000, seq_len=2048, layers=[],
            grad_dtype=Dtype.FP32  # 4 bytes
        )

        # 2. SystemSpec: 定义硬件环境
        system = SystemSpec(
            gpu=MagicMock(), host_mem_gb=512.0, nets=[],
            nodes=2, gpus_per_node=8
        )

        # 3. Graph: 初始只有 1 个中间算子的 Collective
        graph = Graph(collectives=[
            MagicMock(name="tp_comm", group="TP", bytes_=1024)
        ])

        # 4. Strategy: 默认策略
        strategy = Strategy(dp=2, tp=1, pp=1, zero_stage=0)

        return graph, model, system, strategy

    def test_dp_grad_reduce_insertion(self, mock_env):
        """验证 DP > 1 时，结果字典中包含 dp_grad_reduce"""
        graph, model, system, strategy = mock_env

        # 使用 patch 模拟外部依赖函数
        with patch("python.zrt.training.models.comm.collective_time", return_value=0.05), \
                patch("python.zrt.training.models.comm._params_on_rank_for_dp", return_value=10 ** 6), \
                patch("python.zrt.training.models.comm.tier_for_group", return_value="inter_node"):

            results = total_comm_time(graph, model, system, strategy)

            assert "dp_grad_reduce" in results
            assert results["dp_grad_reduce"] == 0.05

    def test_zero_stage_algorithm_selection(self, mock_env):
        """验证 ZeRO-0 使用 All-Reduce (AR)，ZeRO-1+ 使用 Reduce-Scatter (RS)"""
        graph, model, system, strategy = mock_env



        # 情况 A: ZeRO Stage 0
        strategy.zero_stage = 0
        with patch("python.zrt.training.models.comm.collective_time") as mock_time:
            total_comm_time(graph, model, system, strategy)
            # 检查传给 collective_time 的第一个参数 (Collective 对象)
            call_args = mock_time.call_args_list[-1]  # 最后一个调用应该是 DP
            inserted_collective = call_args[0][0]
            assert inserted_collective.kind == "AR"

        # 情况 B: ZeRO Stage 1
        strategy.zero_stage = 1
        with patch("python.zrt.training.models.comm.collective_time") as mock_time:
            total_comm_time(graph, model, system, strategy)
            call_args = mock_time.call_args_list[-1]
            inserted_collective = call_args[0][0]
            assert inserted_collective.kind == "RS"

    def test_grad_sync_byte_calculation(self, mock_env):
        """验证梯度同步的数据量 = 单卡参数量 * 梯度数据类型字节数"""
        graph, model, system, strategy = mock_env
        model.grad_dtype = Dtype.FP32  # 4 bytes
        num_params = 500_000

        with patch("python.zrt.training.models.comm._params_on_rank_for_dp", return_value=num_params), \
                patch("python.zrt.training.models.comm.collective_time") as mock_time:
            total_comm_time(graph, model, system, strategy)

            # 获取注入的 Collective 对象
            dp_comm = mock_time.call_args[0][0]
            # 500,000 params * 4 bytes = 2,000,000 bytes
            assert dp_comm.bytes_ == num_params * 4

    def test_no_dp_no_comm(self, mock_env):
        """当 DP=1 时，不应产生 dp_grad_reduce 开销"""
        graph, model, system, strategy = mock_env
        strategy.dp = 1


        with patch("python.zrt.training.models.comm.collective_time", return_value=0.1):
            results = total_comm_time(graph, model, system, strategy)
            assert "dp_grad_reduce" not in results

    def test_group_size_logic(self, mock_env):
        """验证传递给通信时间计算函数的 group_size 是否等于 strategy.dp"""
        graph, model, system, strategy = mock_env
        strategy.dp = 16


        with patch("python.zrt.training.models.comm.collective_time") as mock_time:
            total_comm_time(graph, model, system, strategy)
            # 第二个参数应该是 group_size
            group_size_passed = mock_time.call_args[0][1]
            assert group_size_passed == 16