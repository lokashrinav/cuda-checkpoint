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

36 experiments across 3 GPU architectures, 9 configurations, 3 models. All tests on Modal with real hardware. Full details in [WRITEUP.md](WRITEUP.md).

### vLLM multi-GPU (the main result)

| Config | Model | GPU | TP | Startup | Cold Start | Reduction | Inference |
|--------|-------|-----|-----|---------|-----------|-----------|-----------|
| Eager | Qwen2-7B | H100 | 1 | 45.2s | 3.56s | 92.1% | 0.40s |
| Eager | Qwen2-7B | H100 | 2 | 102.4s | 4.87s | 95.3% | 0.63s |
| Eager | Qwen2-7B | H100 | 4 | 97.4s | 6.45s | 93.4% | 0.39s |
| Eager | Qwen2-7B | A100 | 2 | 108.4s | 3.46s | 96.8% | 0.529s |
| Eager | Qwen2-7B | L4 | 1 | 66.3s | 4.04s | 93.9% | 1.894s |
| Eager | TinyLlama-1.1B | H100 | 2 | 57.2s | 4.33s | 92.4% | 0.221s |
| CUDA graphs | Qwen2-7B | H100 | 2 | 224.5s | 4.0s | 98.2% | 0.199s |
| CUDA graphs | Qwen2-7B | A100 | 2 | 338.1s | 14.99s | 95.6% | 0.353s |
| AWQ + CUDA graphs | Qwen2-7B-AWQ | H100 | 2 | 310.7s | 4.97s | 98.4% | 0.733s |

### Stability and correctness

- **10-cycle stability**: zero memory leaks, 3.22s avg restore, all outputs identical ([benchmark](benchmarks/vllm_10cycle_stability.py))
- **Error recovery**: invalid PID, double checkpoint, double restore, rapid cycling — all handled safely ([benchmark](benchmarks/vllm_error_recovery.py))
- **Model-agnostic**: validated on Qwen2-7B, Qwen2-7B-AWQ, TinyLlama-1.1B
- **CUDA graphs**: survive checkpoint/restore on H100, 3x faster post-restore inference

### Generic layer (no vLLM)

Validated against raw PyTorch on T4 ([test](tests/modal_test_generic.py)):
- Tensor values survive checkpoint/restore
- `nn.Module` forward + backward pass work post-restore
- CUDA graphs replay correctly after restore
- 10.7 GB allocation, 5-cycle stable, 3.7 GB/s restore rate

### GPU recommendations

| GPU | CUDA graphs? | Expected cold start | Notes |
|-----|-------------|-------------------|-------|
| H100 | Yes (recommended) | 4.0-5.0s | 3x faster post-restore inference |
| A100 | No (enforce-eager) | 3.5s | CUDA graphs add 4x restore overhead on A100 |
| L4 | No (enforce-eager) | 4.0s | Single GPU, inference 4.7x slower than H100 |

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

## Benchmarks

Reproducible Modal scripts in `benchmarks/`:

| Script | What it tests |
|--------|--------------|
| `vllm_single_gpu.py` | Single GPU vllm serve, 92.1% reduction |
| `vllm_multi_gpu_tp2.py` | TP=2 on H100x2, 95.3% reduction |
| `vllm_multi_gpu_tp4.py` | TP=4 on H100x4, 93.4% reduction |
| `vllm_cuda_graphs_tp2.py` | CUDA graphs + TP=2, 98.2% reduction |
| `vllm_10cycle_stability.py` | 10 cycles, zero memory leaks |
| `vllm_error_recovery.py` | 6 failure scenarios, all handled |
| `vllm_a100_portability.py` | A100 hardware validation, 96.8% |
| `vllm_l4_single_gpu.py` | L4 single GPU, 93.9% |

```bash
modal run benchmarks/vllm_multi_gpu_tp2.py
```

## Multi-GPU gap

NVIDIA's `cuda-checkpoint` CLI works on single processes. Most GPU workloads (tensor parallelism, distributed training) spawn multiple CUDA processes. This library bridges that gap:

1. **Multi-GPU support** — parallel checkpoint/restore across all CUDA PIDs in a process tree
2. **43-73% faster restore** — ThreadPoolExecutor parallelism + optional sleep optimization
3. **Framework-agnostic** — works with any CUDA process, not tied to a specific inference server

## License

Apache 2.0
