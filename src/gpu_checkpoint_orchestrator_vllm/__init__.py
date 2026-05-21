"""vLLM-specific integration for gpu-checkpoint-orchestrator.

Builds on the generic gpu_checkpoint_orchestrator package with vLLM-aware orchestration:
  - VLLMCheckpointer: sleep/wake optimization, V0 NCCL reinit, V1 passthrough
  - CLI: vllm-ckpt command for sidecar deployment
  - Auto-discovery of vllm serve processes
"""

from gpu_checkpoint_orchestrator_vllm.orchestrator import VLLMCheckpointer
from gpu_checkpoint_orchestrator_vllm.discovery import find_vllm_server

__all__ = ["VLLMCheckpointer", "find_vllm_server"]
