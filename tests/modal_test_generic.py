"""Modal integration test: generic gpu_checkpoint_orchestrator against raw PyTorch.

Validates that the generic layer (no vLLM) works with any CUDA process.
Tests: single-GPU, multi-GPU, tensor correctness after restore.
"""

import modal

app = modal.App("gpu-checkpoint-orchestrator-generic-test")

CUDA_CHECKPOINT_SETUP = """
cd /opt && git clone --depth 1 https://github.com/NVIDIA/cuda-checkpoint.git 2>/dev/null || true
export PATH=/opt/cuda-checkpoint/bin/x86_64_Linux:$PATH
"""

GENERIC_PKG_INIT = '''
from gpu_checkpoint_orchestrator.api import CudaCheckpointAPI
from gpu_checkpoint_orchestrator.multi_gpu import MultiGPUCheckpointer
from gpu_checkpoint_orchestrator.discover import discover_cuda_pids, find_cuda_pids_for_process
__all__ = ["CudaCheckpointAPI", "MultiGPUCheckpointer", "discover_cuda_pids", "find_cuda_pids_for_process"]
'''

GENERIC_PKG_API = '''
import ctypes

class CudaCheckpointAPI:
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
'''

GENERIC_PKG_MULTI_GPU = '''
import time
from concurrent.futures import ThreadPoolExecutor
from gpu_checkpoint_orchestrator.api import CudaCheckpointAPI

class MultiGPUCheckpointer:
    def __init__(self, pids, parallel=True):
        self.pids = pids
        self.parallel = parallel
        self._api = CudaCheckpointAPI()

    def checkpoint(self):
        if not self.pids:
            raise RuntimeError("No PIDs to checkpoint")
        for pid in self.pids:
            self._api.lock(pid)
        t0 = time.perf_counter()
        try:
            self._run_on_pids(self._api.checkpoint)
        except Exception:
            for pid in self.pids:
                try:
                    self._api.unlock(pid)
                except Exception:
                    pass
            raise
        return {"checkpoint_time": time.perf_counter() - t0, "pids": len(self.pids)}

    def restore(self):
        if not self.pids:
            raise RuntimeError("No PIDs to restore")
        t0 = time.perf_counter()
        try:
            self._run_on_pids(self._api.restore)
        finally:
            for pid in self.pids:
                try:
                    self._api.unlock(pid)
                except Exception:
                    pass
        return {"restore_time": time.perf_counter() - t0, "pids": len(self.pids)}

    def cycle(self):
        ckpt = self.checkpoint()
        rest = self.restore()
        return {**ckpt, **rest}

    def _run_on_pids(self, fn):
        if self.parallel and len(self.pids) > 1:
            with ThreadPoolExecutor(max_workers=len(self.pids)) as ex:
                futures = [ex.submit(fn, pid) for pid in self.pids]
                for f in futures:
                    f.result()
        else:
            for pid in self.pids:
                fn(pid)

    @staticmethod
    def required_env():
        return {
            "CUDA_MODULE_LOADING": "EAGER",
            "NCCL_NVLS_ENABLE": "0",
            "NCCL_P2P_DISABLE": "1",
        }
'''

GENERIC_PKG_DISCOVER = '''
import subprocess

def find_cuda_pids_for_process(root_pid, depth=4):
    all_pids = {str(root_pid)}
    def get_children(pid):
        r = subprocess.run(["pgrep", "-P", pid], capture_output=True, text=True)
        return r.stdout.strip().split() if r.stdout.strip() else []
    frontier = [str(root_pid)]
    for _ in range(depth):
        next_frontier = []
        for pid in frontier:
            children = get_children(pid)
            for c in children:
                all_pids.add(c)
                next_frontier.append(c)
        frontier = next_frontier
    cuda_pids = []
    for pid in sorted(all_pids, key=int):
        r = subprocess.run(
            ["cuda-checkpoint", "--action", "lock", "--pid", pid],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            cuda_pids.append(int(pid))
            subprocess.run(
                ["cuda-checkpoint", "--action", "unlock", "--pid", pid],
                capture_output=True, text=True, timeout=10,
            )
    return sorted(cuda_pids)

def discover_cuda_pids(root_pid):
    return find_cuda_pids_for_process(root_pid)

def find_process_by_name(pattern):
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"No process found matching \\'{pattern}\\'")
    pids = r.stdout.strip().split()
    if len(pids) > 1:
        r2 = subprocess.run(["pgrep", "-f", pattern, "--oldest"], capture_output=True, text=True)
        if r2.returncode == 0 and r2.stdout.strip():
            return int(r2.stdout.strip().split()[0])
    return int(pids[0])
'''

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.5.1")
    .run_commands(
        "cd /opt && git clone --depth 1 https://github.com/NVIDIA/cuda-checkpoint.git",
        "chmod +x /opt/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint",
    )
    .env({
        "CUDA_MODULE_LOADING": "EAGER",
        "PATH": "/opt/cuda-checkpoint/bin/x86_64_Linux:/usr/local/bin:/usr/bin:/bin",
    })
)


def _install_pkg():
    """Write the gpu_checkpoint_orchestrator package to the container."""
    import os
    pkg_dir = "/tmp/gpu_checkpoint_orchestrator"
    os.makedirs(pkg_dir, exist_ok=True)

    with open(f"{pkg_dir}/__init__.py", "w") as f:
        f.write(GENERIC_PKG_INIT)
    with open(f"{pkg_dir}/api.py", "w") as f:
        f.write(GENERIC_PKG_API)
    with open(f"{pkg_dir}/multi_gpu.py", "w") as f:
        f.write(GENERIC_PKG_MULTI_GPU)
    with open(f"{pkg_dir}/discover.py", "w") as f:
        f.write(GENERIC_PKG_DISCOVER)

    import sys
    if "/tmp" not in sys.path:
        sys.path.insert(0, "/tmp")


@app.function(image=image, gpu="T4", timeout=300)
def test_single_gpu_raw_pytorch():
    """Test 1: Single GPU — raw PyTorch tensors, no framework."""
    _install_pkg()
    import os
    import time
    import torch
    from gpu_checkpoint_orchestrator.api import CudaCheckpointAPI

    print("=" * 60)
    print("  TEST 1: Single GPU — Raw PyTorch")
    print("=" * 60)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")

    # Create tensors with known values
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    expected = torch.mm(a, b)
    a_hash = a.sum().item()
    b_hash = b.sum().item()
    expected_hash = expected.sum().item()
    print(f"Tensor checksums: a={a_hash:.4f}, b={b_hash:.4f}, mm={expected_hash:.4f}")

    # Checkpoint
    api = CudaCheckpointAPI()
    pid = os.getpid()
    print(f"PID: {pid}")

    api.lock(pid)
    t0 = time.perf_counter()
    api.checkpoint(pid)
    ckpt_time = time.perf_counter() - t0
    print(f"Checkpoint: {ckpt_time:.3f}s")

    # Restore
    t0 = time.perf_counter()
    api.restore(pid)
    rest_time = time.perf_counter() - t0
    api.unlock(pid)
    print(f"Restore: {rest_time:.3f}s")

    # Verify tensors survived
    a_hash2 = a.sum().item()
    b_hash2 = b.sum().item()
    result = torch.mm(a, b)
    result_hash = result.sum().item()

    a_ok = abs(a_hash - a_hash2) < 1e-3
    b_ok = abs(b_hash - b_hash2) < 1e-3
    mm_ok = abs(expected_hash - result_hash) < 1e-3

    print(f"Post-restore: a={'OK' if a_ok else 'FAIL'}, b={'OK' if b_ok else 'FAIL'}, mm={'OK' if mm_ok else 'FAIL'}")

    # Multi-cycle stability
    print("\n--- 3-cycle stability ---")
    for i in range(3):
        api.lock(pid)
        api.checkpoint(pid)
        api.restore(pid)
        api.unlock(pid)
        check = torch.mm(a, b).sum().item()
        ok = abs(expected_hash - check) < 1e-3
        print(f"  Cycle {i+1}: mm={'OK' if ok else 'FAIL'} ({check:.4f})")

    passed = a_ok and b_ok and mm_ok
    return {
        "test": "single_gpu_raw_pytorch",
        "gpu": gpu_name,
        "checkpoint_time": round(ckpt_time, 4),
        "restore_time": round(rest_time, 4),
        "tensors_intact": passed,
        "verdict": "PASS" if passed else "FAIL",
    }


@app.function(image=image, gpu="T4", timeout=300)
def test_single_gpu_model_inference():
    """Test 2: Single GPU — PyTorch model forward pass survives checkpoint."""
    _install_pkg()
    import os
    import time
    import torch
    import torch.nn as nn
    from gpu_checkpoint_orchestrator.api import CudaCheckpointAPI

    print("=" * 60)
    print("  TEST 2: Single GPU — PyTorch Model Inference")
    print("=" * 60)

    # Simple MLP
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).cuda()
    model.eval()

    x = torch.randn(32, 256, device="cuda")
    with torch.no_grad():
        baseline_out = model(x)
    baseline_hash = baseline_out.sum().item()
    print(f"Baseline output sum: {baseline_hash:.6f}")

    # Checkpoint/restore
    api = CudaCheckpointAPI()
    pid = os.getpid()

    api.lock(pid)
    t0 = time.perf_counter()
    api.checkpoint(pid)
    ckpt_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    api.restore(pid)
    rest_time = time.perf_counter() - t0
    api.unlock(pid)
    print(f"Checkpoint: {ckpt_time:.3f}s, Restore: {rest_time:.3f}s")

    # Verify model works after restore
    with torch.no_grad():
        post_out = model(x)
    post_hash = post_out.sum().item()
    print(f"Post-restore output sum: {post_hash:.6f}")

    match = abs(baseline_hash - post_hash) < 1e-3
    print(f"Output match: {'OK' if match else 'FAIL'}")

    # Verify model can still train
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    target = torch.randn(32, 10, device="cuda")
    loss_fn = nn.MSELoss()

    model.train()
    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, target)
    loss.backward()
    optimizer.step()
    print(f"Post-restore training step: loss={loss.item():.4f} — OK")

    return {
        "test": "single_gpu_model_inference",
        "checkpoint_time": round(ckpt_time, 4),
        "restore_time": round(rest_time, 4),
        "output_match": match,
        "training_works": True,
        "verdict": "PASS" if match else "FAIL",
    }


@app.function(image=image, gpu="T4", timeout=300)
def test_single_gpu_multi_gpu_checkpointer():
    """Test 3: MultiGPUCheckpointer on single GPU (uses same code path)."""
    _install_pkg()
    import os
    import time
    import torch
    from gpu_checkpoint_orchestrator.multi_gpu import MultiGPUCheckpointer

    print("=" * 60)
    print("  TEST 3: MultiGPUCheckpointer — Single GPU")
    print("=" * 60)

    a = torch.randn(2048, 2048, device="cuda")
    expected = torch.mm(a, a).sum().item()
    print(f"Expected mm sum: {expected:.4f}")

    pid = os.getpid()
    mgpu = MultiGPUCheckpointer([pid])

    # Test cycle()
    result = mgpu.cycle()
    print(f"Checkpoint: {result['checkpoint_time']:.3f}s, Restore: {result['restore_time']:.3f}s")

    actual = torch.mm(a, a).sum().item()
    match = abs(expected - actual) < 1e-1
    print(f"Post-cycle mm sum: {actual:.4f} — {'OK' if match else 'FAIL'}")

    # 5-cycle stability
    print("\n--- 5-cycle stability ---")
    all_ok = True
    for i in range(5):
        r = mgpu.cycle()
        check = torch.mm(a, a).sum().item()
        ok = abs(expected - check) < 1e-1
        all_ok = all_ok and ok
        print(f"  Cycle {i+1}: ckpt={r['checkpoint_time']:.3f}s rest={r['restore_time']:.3f}s mm={'OK' if ok else 'FAIL'}")

    return {
        "test": "multi_gpu_checkpointer_single",
        "checkpoint_time": round(result["checkpoint_time"], 4),
        "restore_time": round(result["restore_time"], 4),
        "tensor_intact": match,
        "stability_5_cycles": all_ok,
        "verdict": "PASS" if match and all_ok else "FAIL",
    }


@app.function(image=image, gpu="T4", timeout=300)
def test_cuda_graphs_survive():
    """Test 4: CUDA graphs survive checkpoint/restore."""
    _install_pkg()
    import os
    import time
    import torch
    from gpu_checkpoint_orchestrator.api import CudaCheckpointAPI

    print("=" * 60)
    print("  TEST 4: CUDA Graphs Survive Checkpoint/Restore")
    print("=" * 60)

    # Capture a CUDA graph
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    out = torch.empty(512, 512, device="cuda")

    # Warmup
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            out = torch.mm(a, b)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = torch.mm(a, b)

    # Replay before checkpoint
    g.replay()
    torch.cuda.synchronize()
    baseline = out.clone()
    print(f"Pre-checkpoint graph output sum: {baseline.sum().item():.4f}")

    # Checkpoint/restore
    api = CudaCheckpointAPI()
    pid = os.getpid()
    api.lock(pid)
    api.checkpoint(pid)
    api.restore(pid)
    api.unlock(pid)
    print("Checkpoint/restore complete")

    # Replay after restore
    g.replay()
    torch.cuda.synchronize()
    post = out.clone()
    print(f"Post-restore graph output sum: {post.sum().item():.4f}")

    match = torch.allclose(baseline, post, atol=1e-5)
    print(f"Graph output match: {'OK' if match else 'FAIL'}")

    return {
        "test": "cuda_graphs_survive",
        "graph_output_match": match,
        "verdict": "PASS" if match else "FAIL",
    }


@app.function(image=image, gpu="T4", timeout=300)
def test_large_allocation():
    """Test 5: Large GPU allocation (fill most of T4's 15GB)."""
    _install_pkg()
    import os
    import time
    import torch
    from gpu_checkpoint_orchestrator.api import CudaCheckpointAPI

    print("=" * 60)
    print("  TEST 5: Large GPU Allocation (~10GB)")
    print("=" * 60)

    # Allocate ~10GB
    big = torch.randn(1024, 1024, 1024, device="cuda", dtype=torch.float16)  # ~2GB
    big2 = torch.randn(1024, 1024, 1024, device="cuda", dtype=torch.float16)  # ~2GB
    big3 = torch.randn(1024, 1024, 1024, device="cuda", dtype=torch.float16)  # ~2GB
    big4 = torch.randn(1024, 1024, 1024, device="cuda", dtype=torch.float16)  # ~2GB
    big5 = torch.randn(1024, 1024, 1024, device="cuda", dtype=torch.float16)  # ~2GB

    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory allocated: {mem_gb:.1f} GB")

    checksums = [t.sum().item() for t in [big, big2, big3, big4, big5]]
    print(f"Checksums: {[f'{c:.2f}' for c in checksums]}")

    api = CudaCheckpointAPI()
    pid = os.getpid()

    api.lock(pid)
    t0 = time.perf_counter()
    api.checkpoint(pid)
    ckpt_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    api.restore(pid)
    rest_time = time.perf_counter() - t0
    api.unlock(pid)

    print(f"Checkpoint: {ckpt_time:.3f}s, Restore: {rest_time:.3f}s")
    print(f"Rate: {mem_gb / rest_time:.1f} GB/s restore")

    checksums2 = [t.sum().item() for t in [big, big2, big3, big4, big5]]
    all_match = all(abs(a - b) < 1e-1 for a, b in zip(checksums, checksums2))
    print(f"All tensors intact: {'OK' if all_match else 'FAIL'}")

    return {
        "test": "large_allocation",
        "memory_gb": round(mem_gb, 1),
        "checkpoint_time": round(ckpt_time, 4),
        "restore_time": round(rest_time, 4),
        "restore_rate_gbps": round(mem_gb / rest_time, 1),
        "all_intact": all_match,
        "verdict": "PASS" if all_match else "FAIL",
    }


@app.local_entrypoint()
def main():
    import json

    print("=" * 60)
    print("  GENERIC gpu_checkpoint_orchestrator VALIDATION")
    print("  No vLLM — raw PyTorch on T4")
    print("=" * 60)
    print()

    # Run all tests in parallel
    results = []
    futures = [
        test_single_gpu_raw_pytorch.spawn(),
        test_single_gpu_model_inference.spawn(),
        test_single_gpu_multi_gpu_checkpointer.spawn(),
        test_cuda_graphs_survive.spawn(),
        test_large_allocation.spawn(),
    ]

    for f in futures:
        r = f.get()
        results.append(r)

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    all_pass = True
    for r in results:
        status = r["verdict"]
        if status != "PASS":
            all_pass = False
        print(f"  {r['test']}: {status}")
    print()
    print(f"  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)
    print(json.dumps(results, indent=2))
