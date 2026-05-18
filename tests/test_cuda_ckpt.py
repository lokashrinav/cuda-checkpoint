"""Unit tests for vllm_cuda_ckpt package.

These tests mock the CUDA driver and subprocess calls so they run
without GPUs. Integration tests run on Modal (see cuda_serializer/).
"""

import subprocess
from unittest.mock import MagicMock, patch, call
import pytest


class TestCudaCheckpointAPI:
    """Tests for CudaCheckpointAPI ctypes bindings."""

    def _make_api(self):
        with patch("ctypes.CDLL") as mock_cdll:
            mock_lib = MagicMock()
            mock_cdll.return_value = mock_lib
            for name in ["Lock", "Checkpoint", "Restore", "Unlock"]:
                fn = MagicMock()
                fn.return_value = 0
                setattr(mock_lib, f"cuCheckpointProcess{name}", fn)

            from vllm_cuda_ckpt.api import CudaCheckpointAPI
            api = CudaCheckpointAPI()
            return api, mock_lib

    def test_lock_success(self):
        api, lib = self._make_api()
        api.lock(1234)
        lib.cuCheckpointProcessLock.assert_called_once()

    def test_lock_failure_raises(self):
        api, lib = self._make_api()
        lib.cuCheckpointProcessLock.return_value = 304
        with pytest.raises(RuntimeError, match="rc=304"):
            api.lock(1234)

    def test_checkpoint_success(self):
        api, lib = self._make_api()
        api.checkpoint(1234)
        lib.cuCheckpointProcessCheckpoint.assert_called_once()

    def test_checkpoint_failure_raises(self):
        api, lib = self._make_api()
        lib.cuCheckpointProcessCheckpoint.return_value = 1
        with pytest.raises(RuntimeError, match="rc=1"):
            api.checkpoint(1234)

    def test_restore_success(self):
        api, lib = self._make_api()
        api.restore(1234)
        lib.cuCheckpointProcessRestore.assert_called_once()

    def test_unlock_success(self):
        api, lib = self._make_api()
        api.unlock(1234)
        lib.cuCheckpointProcessUnlock.assert_called_once()

    def test_safe_lock_returns_true_on_success(self):
        api, lib = self._make_api()
        assert api.safe_lock(1234) is True

    def test_safe_lock_returns_false_on_failure(self):
        api, lib = self._make_api()
        lib.cuCheckpointProcessLock.return_value = 304
        assert api.safe_lock(1234) is False

    def test_safe_checkpoint_returns_true(self):
        api, lib = self._make_api()
        assert api.safe_checkpoint(1234) is True

    def test_safe_checkpoint_returns_false(self):
        api, lib = self._make_api()
        lib.cuCheckpointProcessCheckpoint.return_value = 1
        assert api.safe_checkpoint(1234) is False

    def test_safe_restore_returns_true(self):
        api, lib = self._make_api()
        assert api.safe_restore(1234) is True

    def test_safe_restore_returns_false(self):
        api, lib = self._make_api()
        lib.cuCheckpointProcessRestore.return_value = 1
        assert api.safe_restore(1234) is False

    def test_safe_unlock_returns_true(self):
        api, lib = self._make_api()
        assert api.safe_unlock(1234) is True

    def test_safe_unlock_returns_false(self):
        api, lib = self._make_api()
        lib.cuCheckpointProcessUnlock.return_value = 1
        assert api.safe_unlock(1234) is False

    def test_full_cycle(self):
        api, lib = self._make_api()
        api.lock(100)
        api.checkpoint(100)
        api.restore(100)
        api.unlock(100)
        lib.cuCheckpointProcessLock.assert_called_once()
        lib.cuCheckpointProcessCheckpoint.assert_called_once()
        lib.cuCheckpointProcessRestore.assert_called_once()
        lib.cuCheckpointProcessUnlock.assert_called_once()


class TestDiscover:
    """Tests for PID discovery functions."""

    @patch("subprocess.run")
    def test_find_vllm_server_single_pid(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="1234\n", stderr=""
        )
        from vllm_cuda_ckpt.discover import find_vllm_server
        assert find_vllm_server() == 1234

    @patch("subprocess.run")
    def test_find_vllm_server_multiple_pids_uses_oldest(self, mock_run):
        mock_run.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="1234\n5678\n", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="1234\n", stderr=""),
        ]
        from vllm_cuda_ckpt.discover import find_vllm_server
        assert find_vllm_server() == 1234

    @patch("subprocess.run")
    def test_find_vllm_server_not_found_raises(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr=""
        )
        from vllm_cuda_ckpt.discover import find_vllm_server
        with pytest.raises(RuntimeError, match="No vllm serve process found"):
            find_vllm_server()

    @patch("subprocess.run")
    def test_discover_cuda_pids_filters_non_cuda(self, mock_run):
        def side_effect(args, **kwargs):
            if args[0] == "pgrep":
                if "-P" in args:
                    parent = args[args.index("-P") + 1]
                    if parent == "100":
                        return subprocess.CompletedProcess(args=[], returncode=0, stdout="200\n300\n", stderr="")
                    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
            if args[0] == "cuda-checkpoint":
                pid = args[args.index("--pid") + 1]
                if args[args.index("--action") + 1] == "lock":
                    if pid in ("100", "200"):
                        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
                    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
                return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")

        mock_run.side_effect = side_effect
        from vllm_cuda_ckpt.discover import discover_cuda_pids
        pids = discover_cuda_pids(100)
        assert pids == [100, 200]
        assert 300 not in pids


class TestCLI:
    """Tests for CLI argument parsing and command dispatch."""

    def test_main_requires_command(self):
        from vllm_cuda_ckpt.cli import main
        with pytest.raises(SystemExit):
            import sys
            with patch.object(sys, "argv", ["vllm-ckpt"]):
                main()

    def test_discover_subcommand_exists(self):
        import argparse
        from vllm_cuda_ckpt.cli import main
        import sys
        with patch.object(sys, "argv", ["vllm-ckpt", "discover", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_watch_subcommand_has_interval(self):
        import sys
        from vllm_cuda_ckpt.cli import main
        with patch.object(sys, "argv", ["vllm-ckpt", "watch", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_benchmark_subcommand_has_cycles(self):
        import sys
        from vllm_cuda_ckpt.cli import main
        with patch.object(sys, "argv", ["vllm-ckpt", "benchmark", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_recommend_subcommand_exists(self):
        import sys
        from vllm_cuda_ckpt.cli import main
        with patch.object(sys, "argv", ["vllm-ckpt", "recommend", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0


class TestVLLMCheckpointer:
    """Tests for VLLMCheckpointer orchestrator."""

    def test_required_env(self):
        from vllm_cuda_ckpt.api import VLLMCheckpointer
        env = VLLMCheckpointer.required_env()
        assert env["CUDA_MODULE_LOADING"] == "EAGER"
        assert env["NCCL_NVLS_ENABLE"] == "0"
        assert env["NCCL_P2P_DISABLE"] == "1"

    def test_required_llm_kwargs(self):
        from vllm_cuda_ckpt.api import VLLMCheckpointer
        kwargs = VLLMCheckpointer.required_llm_kwargs()
        assert kwargs["disable_custom_all_reduce"] is True

    def test_detect_v1_engine(self):
        with patch("ctypes.CDLL"):
            from vllm_cuda_ckpt.api import VLLMCheckpointer
            mock_llm = MagicMock()
            mock_engine = MagicMock()
            mock_engine.__class__.__module__ = "vllm.v1.engine.core"
            mock_llm.llm_engine = mock_engine
            type(mock_engine).__module__ = "vllm.v1.engine.core"

            ckpt = VLLMCheckpointer(mock_llm)
            assert ckpt.engine_version == "V1"

    def test_detect_v0_engine(self):
        with patch("ctypes.CDLL"):
            from vllm_cuda_ckpt.api import VLLMCheckpointer
            mock_llm = MagicMock()
            mock_engine = MagicMock()
            type(mock_engine).__module__ = "vllm.engine.llm_engine"
            mock_llm.llm_engine = mock_engine

            ckpt = VLLMCheckpointer(mock_llm)
            assert ckpt.engine_version == "V0"

    def test_restore_without_checkpoint_raises(self):
        with patch("ctypes.CDLL"):
            from vllm_cuda_ckpt.api import VLLMCheckpointer
            mock_llm = MagicMock()
            type(mock_llm.llm_engine).__module__ = "vllm.v1.engine.core"

            ckpt = VLLMCheckpointer(mock_llm)
            with pytest.raises(RuntimeError, match="No checkpoint to restore"):
                ckpt.restore()
