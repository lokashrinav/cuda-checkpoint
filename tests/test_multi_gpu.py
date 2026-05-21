"""Tests for gpu_checkpoint_orchestrator.multi_gpu — parallel orchestration."""

from unittest.mock import MagicMock, patch
import pytest


class TestMultiGPUCheckpointer:

    def _make_mgpu(self, pids=None):
        with patch("ctypes.CDLL") as mock_cdll:
            mock_lib = MagicMock()
            mock_cdll.return_value = mock_lib
            for name in ["Lock", "Checkpoint", "Restore", "Unlock"]:
                fn = MagicMock()
                fn.return_value = 0
                setattr(mock_lib, f"cuCheckpointProcess{name}", fn)

            from gpu_checkpoint_orchestrator.multi_gpu import MultiGPUCheckpointer
            mgpu = MultiGPUCheckpointer(pids or [100, 200, 300])
            return mgpu, mock_lib

    def test_checkpoint_returns_timing(self):
        mgpu, _ = self._make_mgpu()
        result = mgpu.checkpoint()
        assert "checkpoint_time" in result
        assert result["pids"] == 3

    def test_restore_returns_timing(self):
        mgpu, _ = self._make_mgpu()
        mgpu.checkpoint()
        result = mgpu.restore()
        assert "restore_time" in result
        assert result["pids"] == 3

    def test_cycle_returns_both(self):
        mgpu, _ = self._make_mgpu()
        result = mgpu.cycle()
        assert "checkpoint_time" in result
        assert "restore_time" in result

    def test_empty_pids_raises(self):
        mgpu, _ = self._make_mgpu(pids=[])
        with pytest.raises(RuntimeError, match="No PIDs"):
            mgpu.checkpoint()

    def test_required_env(self):
        from gpu_checkpoint_orchestrator.multi_gpu import MultiGPUCheckpointer
        env = MultiGPUCheckpointer.required_env()
        assert env["CUDA_MODULE_LOADING"] == "EAGER"
        assert env["NCCL_NVLS_ENABLE"] == "0"
        assert env["NCCL_P2P_DISABLE"] == "1"

    def test_sequential_mode(self):
        mgpu, _ = self._make_mgpu()
        mgpu.parallel = False
        result = mgpu.checkpoint()
        assert result["pids"] == 3
