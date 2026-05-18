# cuda-checkpoint

Python library for multi-GPU CUDA checkpoint/restore. Works with any CUDA process — vLLM, TensorRT-LLM, SGLang, PyTorch training, etc.

Wraps NVIDIA's `cuCheckpointProcess*` driver API with parallel multi-PID orchestration that achieves **92-98% cold start reduction**.

## What it does

Suspends all GPU state (memory, CUDA graphs, contexts) to host memory and restores it later. The process stays alive — no serialization to disk, no model reloading.

```
Before checkpoint:  GPU memory = 25 GB (model + KV cache + graphs)
After checkpoint:   GPU memory = 0 bytes (freed for other workloads)
After restore:      GPU memory = 25 GB (everything back, ready to serve)
```

## Quick start

```python
from cuda_checkpoint import CudaCheckpointAPI, MultiGPUCheckpointer, discover_cuda_pids

# Single process
api = CudaCheckpointAPI()
api.lock(pid)
api.checkpoint(pid)    # GPU memory → host memory
api.restore(pid)       # host memory → GPU memory
api.unlock(pid)

# Multi-GPU (parallel)
pids = discover_cuda_pids(server_pid)  # finds all CUDA-active PIDs in process tree
mgpu = MultiGPUCheckpointer(pids)     # parallel by default
mgpu.checkpoint()                      # all GPUs checkpointed concurrently
mgpu.restore()                         # all GPUs restored concurrently
```

## Install

```bash
pip install cuda-checkpoint

# With vLLM CLI integration
pip install cuda-checkpoint[vllm]
```

## Requirements

- Linux with NVIDIA driver 570+
- `cuda-checkpoint` binary from [NVIDIA/cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint)
- `CUDA_MODULE_LOADING=EAGER` environment variable (must be set before process starts)

For multi-GPU (NCCL):
```bash
export CUDA_MODULE_LOADING=EAGER
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=1
```

## Architecture

Two layers:

### `cuda_checkpoint` — generic core

| Module | What it does |
|--------|-------------|
| `api.py` | `CudaCheckpointAPI` — ctypes bindings to `cuCheckpointProcessLock/Checkpoint/Restore/Unlock` |
| `multi_gpu.py` | `MultiGPUCheckpointer` — parallel checkpoint/restore across N PIDs via ThreadPoolExecutor |
| `discover.py` | `discover_cuda_pids()` — walks process tree, probes each PID for CUDA activity |

### `cuda_checkpoint_vllm` — vLLM integration (optional)

| Module | What it does |
|--------|-------------|
| `orchestrator.py` | `VLLMCheckpointer` — sleep/wake optimization, V0/V1 engine detection |
| `discovery.py` | `find_vllm_server()` — auto-discovers running `vllm serve` |
| `cli.py` | `vllm-ckpt` CLI — discover, cycle, benchmark, watch (sidecar daemon), recommend |

## Validated results

Tested on Modal across 3 GPU architectures and 9 configurations:

| GPU | Config | Cold start | Reduction | Restore rate |
|-----|--------|-----------|-----------|-------------|
| H100 x2 | CUDA graphs, TP=2 | 4.0s | 98.2% | — |
| H100 x4 | Eager, TP=4 | 6.5s | 93.4% | — |
| A100 x2 | Eager, TP=2 | 3.5s | 96.8% | — |
| L4 | Eager, TP=1 | 4.0s | 93.9% | — |
| T4 | Raw PyTorch (no vLLM) | — | — | 3.7 GB/s |

Generic layer validated against raw PyTorch on T4:
- Tensor values survive checkpoint/restore
- `nn.Module` forward + backward pass work post-restore
- CUDA graphs replay correctly after restore
- 10.7 GB allocation, 5-cycle stable, zero memory leaks

## vLLM CLI

```bash
# Auto-discover and checkpoint/restore running vllm serve
vllm-ckpt cycle --port 8000 --model Qwen/Qwen2-7B

# Sidecar daemon (periodic checkpoint, SIGTERM-safe)
vllm-ckpt watch --port 8000 --interval 300 --json

# GPU-specific recommendations
vllm-ckpt recommend

# Benchmark
vllm-ckpt benchmark --port 8000 --model Qwen/Qwen2-7B --cycles 5
```

## Deploy

Kubernetes and Docker Compose manifests in `deploy/`:

```bash
# Kubernetes — vLLM + checkpoint sidecar
kubectl apply -f deploy/kubernetes/vllm-checkpoint-sidecar.yaml

# Docker Compose — local dev
docker compose -f deploy/docker-compose/docker-compose.yaml up
```

## Multi-GPU gap

NVIDIA's `cuda-checkpoint` CLI works on single processes. Most GPU workloads (tensor parallelism, distributed training) spawn multiple CUDA processes. This library bridges that gap:

1. **Multi-GPU support** — parallel checkpoint/restore across all CUDA PIDs in a process tree
2. **43-73% faster restore** — ThreadPoolExecutor parallelism + optional sleep optimization
3. **Framework-agnostic** — works with any CUDA process, not tied to a specific inference server

## License

Apache 2.0
