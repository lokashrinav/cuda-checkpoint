"""Tests for gpu_checkpoint_orchestrator_vllm — vLLM integration."""

from unittest.mock import MagicMock, patch
import pytest


class TestVLLMCheckpointer:

    def test_required_env(self):
        from gpu_checkpoint_orchestrator.multi_gpu import MultiGPUCheckpointer
        env = MultiGPUCheckpointer.required_env()
        assert env["CUDA_MODULE_LOADING"] == "EAGER"

    def test_required_llm_kwargs(self):
        with patch("ctypes.CDLL"):
            from gpu_checkpoint_orchestrator_vllm.orchestrator import VLLMCheckpointer
            kwargs = VLLMCheckpointer.required_llm_kwargs()
            assert kwargs["disable_custom_all_reduce"] is True

    def test_detect_v1_engine(self):
        with patch("ctypes.CDLL"):
            from gpu_checkpoint_orchestrator_vllm.orchestrator import VLLMCheckpointer
            mock_llm = MagicMock()
            mock_engine = MagicMock()
            type(mock_engine).__module__ = "vllm.v1.engine.core"
            mock_llm.llm_engine = mock_engine

            ckpt = VLLMCheckpointer(mock_llm)
            assert ckpt.engine_version == "V1"

    def test_detect_v0_engine(self):
        with patch("ctypes.CDLL"):
            from gpu_checkpoint_orchestrator_vllm.orchestrator import VLLMCheckpointer
            mock_llm = MagicMock()
            mock_engine = MagicMock()
            type(mock_engine).__module__ = "vllm.engine.llm_engine"
            mock_llm.llm_engine = mock_engine

            ckpt = VLLMCheckpointer(mock_llm)
            assert ckpt.engine_version == "V0"


class TestVLLMCLI:

    def test_main_requires_command(self):
        from gpu_checkpoint_orchestrator_vllm.cli import main
        import sys
        with pytest.raises(SystemExit):
            with patch.object(sys, "argv", ["vllm-ckpt"]):
                main()

    def test_discover_subcommand(self):
        from gpu_checkpoint_orchestrator_vllm.cli import main
        import sys
        with patch.object(sys, "argv", ["vllm-ckpt", "discover", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_watch_subcommand(self):
        from gpu_checkpoint_orchestrator_vllm.cli import main
        import sys
        with patch.object(sys, "argv", ["vllm-ckpt", "watch", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_benchmark_subcommand(self):
        from gpu_checkpoint_orchestrator_vllm.cli import main
        import sys
        with patch.object(sys, "argv", ["vllm-ckpt", "benchmark", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_recommend_subcommand(self):
        from gpu_checkpoint_orchestrator_vllm.cli import main
        import sys
        with patch.object(sys, "argv", ["vllm-ckpt", "recommend", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0


class TestVLLMDiscovery:

    @patch("gpu_checkpoint_orchestrator.discover.find_process_by_name")
    def test_find_vllm_server(self, mock_find):
        mock_find.return_value = 42
        from gpu_checkpoint_orchestrator_vllm.discovery import find_vllm_server
        assert find_vllm_server() == 42
        mock_find.assert_called_once_with("vllm.entrypoints.openai.api_server")
