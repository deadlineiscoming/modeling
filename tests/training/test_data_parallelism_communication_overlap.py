import pytest
from unittest.mock import MagicMock, patch
from python.zrt.training.compose.pipeline import pipeline_step_time
from python.zrt.training.compose.stage import StageTime


class TestPipelineStepTimeDP:

    @pytest.fixture
    def mock_env(self):
        graph = MagicMock()
        model = MagicMock()
        system = MagicMock()
        strategy = MagicMock()

        # 默认设置
        strategy.pp = 2
        strategy.num_microbatches.return_value = 8
        strategy.dp_overlap_in_bubble = True

        return graph, model, system, strategy

    def test_dp_time_added_to_step_time(self, mock_env):
        """验证 DP 通信时间确实被累加到了最终的 step_time 中"""
        graph, model, system, strategy = mock_env

        # 模拟 stage_time: 每个 stage fwd=10, bwd=10
        st = StageTime(fwd=10.0, bwd=10.0)

        # 模拟 total_comm_time 返回 DP 耗时 50
        comm_times = {"dp_grad_reduce": 50.0}

        with patch("python.zrt.training.compose.pipeline._assign_stages", return_value={0: [0], 1: [1]}), \
                patch("python.zrt.training.compose.pipeline.stage_time", return_value=st), \
                patch("python.zrt.training.compose.pipeline.total_comm_time", return_value=comm_times), \
                patch("python.zrt.training.compose.pipeline.memory_breakdown"), \
                patch("python.zrt.training.compose.pipeline.compute_mfu"):

            result = pipeline_step_time(graph, model, system, strategy)

            # 计算预期:
            # pp=2, M=8, t_fwd=10, t_bwd=10
            # warmup = (2-1)*10 = 10
            # steady = 8*(10+10) = 160
            # cooldown = (2-1)*10 = 10
            # bubble = 10 + 10 = 20
            # dp_ar = 50. 由于 50 > bubble(20)，暴露出来的 dp_exposed = 50 - 20 = 30
            # step_time = 10 + 160 + 10 + 30 = 210
            assert result.step_time == 210.0
            assert result.dp_ar_exposed == 30.0

    def test_dp_overlap_in_bubble_full_hide(self, mock_env):
        """验证当 DP 耗时小于气泡时，暴露出的 DP 时间为 0 (完全掩盖)"""
        graph, model, system, strategy = mock_env
        strategy.pp = 4
        strategy.num_microbatches.return_value = 10
        strategy.dp_overlap_in_bubble = True

        st = StageTime(fwd=10.0, bwd=10.0)
        # 气泡大小 = (4-1)*10 + (4-1)*10 = 30 + 30 = 60
        # 设置 DP 耗时为 40 (小于 60)
        comm_times = {"dp_grad_reduce": 40.0}

        with patch("python.zrt.training.compose.pipeline._assign_stages", return_value={i: [i] for i in range(4)}), \
                patch("python.zrt.training.compose.pipeline.stage_time", return_value=st), \
                patch("python.zrt.training.compose.pipeline.total_comm_time", return_value=comm_times), \
                patch("python.zrt.training.compose.pipeline.memory_breakdown"), \
                patch("python.zrt.training.compose.pipeline.compute_mfu"):

            result = pipeline_step_time(graph, model, system, strategy)

            # dp_exposed 应该为 0
            assert result.dp_ar_exposed == 0.0
            # 总时间 = 30 (warmup) + 200 (steady) + 30 (cooldown) + 0 (exposed) = 260
            assert result.step_time == 260.0

    def test_dp_no_overlap_flag(self, mock_env):
        """验证当 strategy.dp_overlap_in_bubble 为 False 时，DP 时间全量暴露"""
        graph, model, system, strategy = mock_env
        strategy.dp_overlap_in_bubble = False

        st = StageTime(fwd=10.0, bwd=10.0)
        comm_times = {"dp_grad_reduce": 40.0}

        with patch("python.zrt.training.compose.pipeline._assign_stages", return_value={0: [0], 1: [1]}), \
                patch("python.zrt.training.compose.pipeline.stage_time", return_value=st), \
                patch("python.zrt.training.compose.pipeline.total_comm_time", return_value=comm_times), \
                patch("python.zrt.training.compose.pipeline.memory_breakdown"), \
                patch("python.zrt.training.compose.pipeline.compute_mfu"):

            result = pipeline_step_time(graph, model, system, strategy)

            # 虽然有气泡，但因为 flag 为 False，DP 时间不被掩盖
            assert result.dp_ar_exposed == 40.0

    def test_pp1_dp_logic(self, mock_env):
        """验证 PP=1 时，DP 时间不进行气泡掩盖（直接累加）"""
        graph, model, system, strategy = mock_env
        strategy.pp = 1
        strategy.num_microbatches.return_value = 8

        st = StageTime(fwd=10.0, bwd=10.0)
        comm_times = {"dp_grad_reduce": 50.0}

        with patch("python.zrt.training.compose.pipeline._assign_stages", return_value={0: [0]}), \
                patch("python.zrt.training.compose.pipeline.stage_time", return_value=st), \
                patch("python.zrt.training.compose.pipeline.total_comm_time", return_value=comm_times), \
                patch("python.zrt.training.compose.pipeline.memory_breakdown"), \
                patch("python.zrt.training.compose.pipeline.compute_mfu"):

            result = pipeline_step_time(graph, model, system, strategy)

            # PP=1 逻辑: M * (fwd + bwd) + dp_ar_time
            # 8 * (10 + 10) + 50 = 160 + 50 = 210
            assert result.step_time == 210.0
            assert result.dp_ar_exposed == 50.0