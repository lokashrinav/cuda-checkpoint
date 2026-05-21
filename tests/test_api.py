"""Tests for gpu_checkpoint_orchestrator.api — generic CUDA checkpoint bindings."""

from unittest.mock import MagicMock, patch
import pytest


class TestCudaCheckpointAPI:

    def _make_api(self):
        with patch("ctypes.CDLL") as mock_cdll:
            mock_lib = MagicMock()
            mock_cdll.return_value = mock_lib
            for name in ["Lock", "Checkpoint", "Restore", "Unlock"]:
                fn = MagicMock()
                fn.return_value = 0
                setattr(mock_lib, f"cuCheckpointProcess{name}", fn)

            from gpu_checkpoint_orchestrator.api import CudaCheckpointAPI
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

    def test_safe_lock_true(self):
        api, _ = self._make_api()
        assert api.safe_lock(1234) is True

    def test_safe_lock_false(self):
        api, lib = self._make_api()
        lib.cuCheckpointProcessLock.return_value = 304
        assert api.safe_lock(1234) is False

    def test_safe_checkpoint_true(self):
        api, _ = self._make_api()
        assert api.safe_checkpoint(1234) is True

    def test_safe_checkpoint_false(self):
        api, lib = self._make_api()
        lib.cuCheckpointProcessCheckpoint.return_value = 1
        assert api.safe_checkpoint(1234) is False

    def test_safe_restore_true(self):
        api, _ = self._make_api()
        assert api.safe_restore(1234) is True

    def test_safe_restore_false(self):
        api, lib = self._make_api()
        lib.cuCheckpointProcessRestore.return_value = 1
        assert api.safe_restore(1234) is False

    def test_safe_unlock_true(self):
        api, _ = self._make_api()
        assert api.safe_unlock(1234) is True

    def test_safe_unlock_false(self):
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
