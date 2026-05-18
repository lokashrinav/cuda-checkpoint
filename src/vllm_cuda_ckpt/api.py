"""CUDA checkpoint/restore API bindings and vLLM orchestrator."""

import ctypes
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional


class CudaCheckpointAPI:
    """Direct ctypes bindings to cuCheckpointProcess* 4-step API.

    Requires Linux with NVIDIA driver 570+ and libcuda.so.1.
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


class VLLMCheckpointer:
    """Orchestrates cuda-checkpoint for a running vLLM LLM instance.

    Works with the Python LLM class (needs engine reference).
    For external process management, use the CLI or discover_cuda_pids().

    Optimizations (V1 engine, enabled by default):
    - sleep(): frees model weights before checkpoint (~6 GiB per worker)
    - Parallel PID processing: checkpoint/restore all PIDs concurrently
    - Combined: 3.1s multi-GPU restore (89% reduction from 28.5s load)

    Required environment:
        CUDA_MODULE_LOADING=EAGER
        NCCL_NVLS_ENABLE=0
        NCCL_P2P_DISABLE=1

    Required LLM kwargs:
        disable_custom_all_reduce=True
    """

    def __init__(self, llm, use_sleep: bool = True, parallel: bool = True):
        self.llm = llm
        self.use_sleep = use_sleep
        self.parallel = parallel
        self._cuda_pids: Optional[list[int]] = None
        self._engine_version: Optional[str] = None
        self._is_sleeping: bool = False
        self._api: Optional[CudaCheckpointAPI] = None
        self._detect_engine()
        self._init_api()

    def _detect_engine(self):
        module = type(self.llm.llm_engine).__module__
        self._engine_version = "V1" if "v1" in module else "V0"

    def _init_api(self):
        try:
            self._api = CudaCheckpointAPI()
        except (OSError, AttributeError):
            self._api = None

    def _discover_cuda_pids(self) -> list[int]:
        if self._cuda_pids is not None:
            return self._cuda_pids

        from vllm_cuda_ckpt.discover import discover_cuda_pids
        self._cuda_pids = discover_cuda_pids(os.getpid())
        return self._cuda_pids

    def _v1_sleep(self) -> float:
        t0 = time.perf_counter()
        self.llm.llm_engine.sleep()
        self._is_sleeping = True
        return time.perf_counter() - t0

    def _v1_wake_up(self) -> float:
        t0 = time.perf_counter()
        self.llm.llm_engine.wake_up()
        self._is_sleeping = False
        return time.perf_counter() - t0

    def _do_checkpoint_pids(self, pids: list[int]):
        if self._api and self.parallel and len(pids) > 1:
            with ThreadPoolExecutor(max_workers=len(pids)) as ex:
                futures = [ex.submit(self._api.checkpoint, pid) for pid in pids]
                for f in futures:
                    f.result()
        elif self._api:
            for pid in pids:
                self._api.checkpoint(pid)
        else:
            for pid in pids:
                r = subprocess.run(
                    ["cuda-checkpoint", "--action", "checkpoint", "--pid", str(pid)],
                    capture_output=True, text=True, timeout=300,
                )
                if r.returncode != 0:
                    raise RuntimeError(f"Checkpoint failed for PID {pid}: {r.stderr.strip()}")

    def _do_restore_pids(self, pids: list[int]):
        if self._api and self.parallel and len(pids) > 1:
            with ThreadPoolExecutor(max_workers=len(pids)) as ex:
                futures = [ex.submit(self._api.restore, pid) for pid in pids]
                for f in futures:
                    f.result()
        elif self._api:
            for pid in pids:
                self._api.restore(pid)
        else:
            for pid in pids:
                r = subprocess.run(
                    ["cuda-checkpoint", "--action", "restore", "--pid", str(pid)],
                    capture_output=True, text=True, timeout=300,
                )
                if r.returncode != 0:
                    raise RuntimeError(f"Restore failed for PID {pid}: {r.stderr.strip()}")

    def checkpoint(self) -> dict:
        """Checkpoint GPU state to host memory."""
        cuda_pids = self._discover_cuda_pids()
        if not cuda_pids:
            raise RuntimeError("No CUDA-active processes found")

        sleep_time = 0.0
        if self._engine_version == "V1" and self.use_sleep:
            sleep_time = self._v1_sleep()

        try:
            if self._api:
                for pid in cuda_pids:
                    self._api.lock(pid)
            else:
                for pid in cuda_pids:
                    r = subprocess.run(
                        ["cuda-checkpoint", "--action", "lock", "--pid", str(pid)],
                        capture_output=True, text=True, timeout=30,
                    )
                    if r.returncode != 0:
                        raise RuntimeError(f"Lock failed for PID {pid}")

            t0 = time.perf_counter()
            self._do_checkpoint_pids(cuda_pids)
            ckpt_time = time.perf_counter() - t0

        except Exception:
            for pid in cuda_pids:
                try:
                    if self._api:
                        self._api.unlock(pid)
                    else:
                        subprocess.run(
                            ["cuda-checkpoint", "--action", "unlock", "--pid", str(pid)],
                            capture_output=True, text=True, timeout=30,
                        )
                except Exception:
                    pass
            if self._is_sleeping:
                try:
                    self._v1_wake_up()
                except Exception:
                    pass
            raise

        result = {"ckpt_time": ckpt_time, "pids": cuda_pids}
        if sleep_time > 0:
            result["sleep_time"] = sleep_time
        return result

    def restore(self) -> dict:
        """Restore GPU state from host memory."""
        cuda_pids = self._cuda_pids
        if not cuda_pids:
            raise RuntimeError("No checkpoint to restore -- call checkpoint() first")

        t0 = time.perf_counter()
        try:
            self._do_restore_pids(cuda_pids)
            rest_time = time.perf_counter() - t0
        finally:
            for pid in cuda_pids:
                try:
                    if self._api:
                        self._api.unlock(pid)
                    else:
                        subprocess.run(
                            ["cuda-checkpoint", "--action", "unlock", "--pid", str(pid)],
                            capture_output=True, text=True, timeout=30,
                        )
                except Exception:
                    pass

        wake_time = 0.0
        if self._is_sleeping:
            wake_time = self._v1_wake_up()

        return {"rest_time": rest_time, "wake_time": wake_time,
                "total_restore": rest_time + wake_time}

    def cycle(self) -> dict:
        """Full checkpoint + restore cycle."""
        ckpt_info = self.checkpoint()
        rest_info = self.restore()
        return {**ckpt_info, **rest_info}

    @property
    def engine_version(self) -> str:
        return self._engine_version

    @staticmethod
    def required_env() -> dict[str, str]:
        return {
            "CUDA_MODULE_LOADING": "EAGER",
            "NCCL_NVLS_ENABLE": "0",
            "NCCL_P2P_DISABLE": "1",
        }

    @staticmethod
    def required_llm_kwargs() -> dict:
        return {
            "disable_custom_all_reduce": True,
        }
