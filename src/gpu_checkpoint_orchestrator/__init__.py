"""Multi-GPU checkpoint/restore orchestration for any CUDA process.

Works with vLLM, TensorRT-LLM, SGLang, PyTorch training, etc.
Wraps NVIDIA's cuCheckpointProcess* driver API with multi-GPU coordination.

Two layers:
  1. CudaCheckpointAPI — direct ctypes bindings to cuCheckpointProcess* 4-step API
  2. MultiGPUCheckpointer — parallel checkpoint/restore across multiple CUDA PIDs
"""

from gpu_checkpoint_orchestrator.api import CudaCheckpointAPI
from gpu_checkpoint_orchestrator.multi_gpu import MultiGPUCheckpointer
from gpu_checkpoint_orchestrator.discover import discover_cuda_pids, find_cuda_pids_for_process

__all__ = [
    "CudaCheckpointAPI",
    "MultiGPUCheckpointer",
    "discover_cuda_pids",
    "find_cuda_pids_for_process",
]
