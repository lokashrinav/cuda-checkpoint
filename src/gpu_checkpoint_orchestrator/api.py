"""Direct ctypes bindings to NVIDIA's cuCheckpointProcess* 4-step API.

Requires Linux with NVIDIA driver 570+ and libcuda.so.1.
Works with any CUDA process — not framework-specific.

Usage:
    api = CudaCheckpointAPI()
    api.lock(pid)
    api.checkpoint(pid)
    # ... GPU memory is now freed ...
    api.restore(pid)
    api.unlock(pid)
"""

import ctypes


class CudaCheckpointAPI:
    """Direct ctypes bindings to cuCheckpointProcess* 4-step API.

    The 4-step sequence: lock → checkpoint → restore → unlock.
    Each step operates on a single PID. For multi-GPU, call on each PID
    (or use MultiGPUCheckpointer for parallel orchestration).
    """

    def __init__(self):
        self._lib = ctypes.CDLL("libcuda.so.1")
        for name in ["Lock", "Checkpoint", "Restore", "Unlock"]:
            fn = getattr(self._lib, f"cuCheckpointProcess{name}")
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_int, ctypes.c_void_p]
            setattr(self, f"_fn_{name.lower()}", fn)

    def _make_args(self):
        return (ctypes.c_byte * 64)()

    def lock(self, pid: int):
        args = self._make_args()
        rc = self._fn_lock(pid, ctypes.byref(args))
        if rc != 0:
            raise RuntimeError(f"cuCheckpointProcessLock failed for PID {pid}: rc={rc}")

    def checkpoint(self, pid: int):
        args = self._make_args()
        rc = self._fn_checkpoint(pid, ctypes.byref(args))
        if rc != 0:
            raise RuntimeError(f"cuCheckpointProcessCheckpoint failed for PID {pid}: rc={rc}")

    def restore(self, pid: int):
        args = self._make_args()
        rc = self._fn_restore(pid, ctypes.byref(args))
        if rc != 0:
            raise RuntimeError(f"cuCheckpointProcessRestore failed for PID {pid}: rc={rc}")

    def unlock(self, pid: int):
        args = self._make_args()
        rc = self._fn_unlock(pid, ctypes.byref(args))
        if rc != 0:
            raise RuntimeError(f"cuCheckpointProcessUnlock failed for PID {pid}: rc={rc}")

    def safe_lock(self, pid: int) -> bool:
        try:
            self.lock(pid)
            return True
        except RuntimeError:
            return False

    def safe_checkpoint(self, pid: int) -> bool:
        try:
            self.checkpoint(pid)
            return True
        except RuntimeError:
            return False

    def safe_restore(self, pid: int) -> bool:
        try:
            self.restore(pid)
            return True
        except RuntimeError:
            return False

    def safe_unlock(self, pid: int) -> bool:
        try:
            self.unlock(pid)
            return True
        except RuntimeError:
            return False
