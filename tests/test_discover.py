"""Tests for gpu_checkpoint_orchestrator.discover — PID discovery."""

import subprocess
from unittest.mock import patch
import pytest


class TestDiscover:

    @patch("subprocess.run")
    def test_find_process_by_name_single(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="1234\n", stderr=""
        )
        from gpu_checkpoint_orchestrator.discover import find_process_by_name
        assert find_process_by_name("some.server") == 1234

    @patch("subprocess.run")
    def test_find_process_by_name_multiple_uses_oldest(self, mock_run):
        mock_run.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="1234\n5678\n", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="1234\n", stderr=""),
        ]
        from gpu_checkpoint_orchestrator.discover import find_process_by_name
        assert find_process_by_name("some.server") == 1234

    @patch("subprocess.run")
    def test_find_process_not_found_raises(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr=""
        )
        from gpu_checkpoint_orchestrator.discover import find_process_by_name
        with pytest.raises(RuntimeError, match="No process found"):
            find_process_by_name("nonexistent")

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
        from gpu_checkpoint_orchestrator.discover import discover_cuda_pids
        pids = discover_cuda_pids(100)
        assert pids == [100, 200]
        assert 300 not in pids
