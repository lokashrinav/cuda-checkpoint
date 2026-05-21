"""Benchmark: CUDA graphs vs enforce_eager checkpoint/restore on A100.

Measures how much CUDA graphs add to checkpoint/restore overhead.
The difference = maximum improvement available from freeing graphs before
checkpoint and loading them from Foundry serialization after restore.

Config A: CUDA graphs (vLLM default) — checkpoint/restore includes graph state
Config B: enforce_eager — checkpoint/restore is just weights + KV cache

If Config A restore = 15s and Config B restore = 3.5s, then freeing graphs
saves ~11.5s of restore time. If Foundry graph loading takes ~4s, the combined
approach (free graphs + checkpoint + restore + Foundry load) = ~7.5s vs 15s.

Usage:
    modal run benchmarks/cuda_graphs_vs_eager_a100.py
"""

import modal, os
app = modal.App("cuda-graphs-vs-eager-a100")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .run_commands(
        "apt-get update && apt-get install -y wget gnupg2 git build-essential curl",
        "cd /opt && git clone --depth 1 https://github.com/NVIDIA/cuda-checkpoint.git || true",
    )
    .pip_install("transformers==4.52.4", "vllm==0.8.5.post1", "httpx")
    .run_commands(
        "python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-7B')\"",
    )
)

SERVE_SCRIPT = '''
import os, sys, time, json, subprocess, ctypes, traceback
from concurrent.futures import ThreadPoolExecutor

os.environ["CUDA_MODULE_LOADING"] = "EAGER"
os.environ["NCCL_NVLS_ENABLE"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["VLLM_USE_V1"] = "1"
os.environ["PATH"] = "/opt/cuda-checkpoint/bin/x86_64_Linux:" + os.environ["PATH"]

import httpx

MODEL = "Qwen/Qwen2-7B"
PORT = 8000
BASE_URL = f"http://localhost:{PORT}"


class CudaCheckpointAPI:
    def __init__(self):
        self._lib = ctypes.CDLL("libcuda.so.1")
        for name in ["Lock", "Checkpoint", "Restore", "Unlock"]:
            fn = getattr(self._lib, f"cuCheckpointProcess{name}")
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_int, ctypes.c_void_p]
            setattr(self, f"_fn_{name.lower()}", fn)

    def _args(self):
        return (ctypes.c_byte * 64)()

    def lock(self, pid):
        rc = self._fn_lock(pid, ctypes.byref(self._args()))
        if rc != 0: raise RuntimeError(f"Lock {pid}: rc={rc}")

    def checkpoint(self, pid):
        rc = self._fn_checkpoint(pid, ctypes.byref(self._args()))
        if rc != 0: raise RuntimeError(f"Checkpoint {pid}: rc={rc}")

    def restore(self, pid):
        rc = self._fn_restore(pid, ctypes.byref(self._args()))
        if rc != 0: raise RuntimeError(f"Restore {pid}: rc={rc}")

    def unlock(self, pid):
        rc = self._fn_unlock(pid, ctypes.byref(self._args()))
        if rc != 0: raise RuntimeError(f"Unlock {pid}: rc={rc}")


def wait_for_server(timeout=600):
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                return time.perf_counter() - start
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(1)
    raise TimeoutError(f"Server didn't start within {timeout}s")


def query_server(prompt, max_tokens=32):
    r = httpx.post(
        f"{BASE_URL}/v1/completions",
        json={"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()


def get_gpu_memory():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    return [int(x.strip()) for x in r.stdout.strip().split("\\n")]


def find_cuda_pids(server_pid):
    all_pids = {str(server_pid)}
    def get_children(pid):
        r = subprocess.run(["pgrep", "-P", pid], capture_output=True, text=True)
        return r.stdout.strip().split() if r.stdout.strip() else []
    for cpid in get_children(str(server_pid)):
        all_pids.add(cpid)
        for gc in get_children(cpid):
            all_pids.add(gc)
            for ggc in get_children(gc):
                all_pids.add(ggc)
    cuda_pids = []
    for pid in sorted(all_pids, key=int):
        r = subprocess.run(["cuda-checkpoint", "--action", "lock", "--pid", pid],
                          capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            cuda_pids.append(int(pid))
            subprocess.run(["cuda-checkpoint", "--action", "unlock", "--pid", pid],
                          capture_output=True, text=True, timeout=10)
    return sorted(cuda_pids)


def run_checkpoint_cycle(api, cuda_pids, label):
    """Run one checkpoint/restore cycle, return timing dict."""
    mem_before = get_gpu_memory()

    for pid in cuda_pids:
        api.lock(pid)

    with ThreadPoolExecutor(max_workers=len(cuda_pids)) as ex:
        t0 = time.perf_counter()
        list(ex.map(api.checkpoint, cuda_pids))
        ckpt_time = time.perf_counter() - t0

    mem_checkpointed = get_gpu_memory()

    with ThreadPoolExecutor(max_workers=len(cuda_pids)) as ex:
        t0 = time.perf_counter()
        list(ex.map(api.restore, cuda_pids))
        restore_time = time.perf_counter() - t0

    for pid in cuda_pids:
        api.unlock(pid)

    mem_restored = get_gpu_memory()

    t0 = time.perf_counter()
    text = query_server("The capital of France is")
    infer_time = time.perf_counter() - t0

    cold_start = restore_time + infer_time

    print(f"  [{label}] GPU mem: before={mem_before} MB, ckpt={mem_checkpointed} MB, restored={mem_restored} MB", flush=True)
    print(f"  [{label}] Checkpoint: {ckpt_time:.2f}s, Restore: {restore_time:.2f}s, Infer: {infer_time:.3f}s", flush=True)
    print(f"  [{label}] Cold start: {cold_start:.2f}s, Output: {text[:60]}", flush=True)

    return {
        "checkpoint": ckpt_time,
        "restore": restore_time,
        "inference": infer_time,
        "cold_start": cold_start,
        "mem_before": mem_before,
        "mem_checkpointed": mem_checkpointed,
        "mem_restored": mem_restored,
        "output": text,
    }


def run_config(api, label, enforce_eager):
    """Start vLLM, run 3 checkpoint/restore cycles, return results."""
    print(f"\\n{'=' * 60}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'=' * 60}", flush=True)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--tensor-parallel-size", "2",
        "--gpu-memory-utilization", "0.30",
        "--max-model-len", "512",
        "--port", str(PORT),
        "--disable-custom-all-reduce",
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")

    server_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    startup_time = wait_for_server(timeout=600)
    print(f"  Server ready in {startup_time:.1f}s (PID {server_proc.pid})", flush=True)

    ref_text = query_server("The capital of France is")
    print(f"  Baseline output: {ref_text[:80]}", flush=True)

    mem_initial = get_gpu_memory()
    print(f"  Initial GPU memory: {mem_initial} MB", flush=True)

    cuda_pids = find_cuda_pids(server_proc.pid)
    print(f"  CUDA PIDs: {cuda_pids} ({len(cuda_pids)} total)", flush=True)

    cycles = []
    for i in range(3):
        result = run_checkpoint_cycle(api, cuda_pids, f"Cycle {i+1}")
        result["match"] = result["output"].strip() == ref_text.strip()
        cycles.append(result)

    server_proc.terminate()
    server_proc.wait(timeout=15)

    return {
        "startup_time": startup_time,
        "num_cuda_pids": len(cuda_pids),
        "mem_initial": mem_initial,
        "ref_text": ref_text,
        "cycles": cycles,
    }


try:
    print("=" * 60, flush=True)
    print("  CUDA GRAPHS vs ENFORCE_EAGER: A100 CHECKPOINT/RESTORE", flush=True)
    print("=" * 60, flush=True)

    api = CudaCheckpointAPI()

    # Config A: CUDA graphs (default)
    results_a = run_config(api, "CONFIG A: CUDA GRAPHS (default)", enforce_eager=False)

    time.sleep(10)

    # Config B: enforce_eager (no CUDA graphs)
    results_b = run_config(api, "CONFIG B: ENFORCE_EAGER (no graphs)", enforce_eager=True)

    # --- Analysis ---
    avg_a = {
        "checkpoint": sum(c["checkpoint"] for c in results_a["cycles"]) / 3,
        "restore": sum(c["restore"] for c in results_a["cycles"]) / 3,
        "inference": sum(c["inference"] for c in results_a["cycles"]) / 3,
        "cold_start": sum(c["cold_start"] for c in results_a["cycles"]) / 3,
    }
    avg_b = {
        "checkpoint": sum(c["checkpoint"] for c in results_b["cycles"]) / 3,
        "restore": sum(c["restore"] for c in results_b["cycles"]) / 3,
        "inference": sum(c["inference"] for c in results_b["cycles"]) / 3,
        "cold_start": sum(c["cold_start"] for c in results_b["cycles"]) / 3,
    }

    restore_overhead = avg_a["restore"] - avg_b["restore"]
    ckpt_overhead = avg_a["checkpoint"] - avg_b["checkpoint"]

    print(f"\\n{'=' * 60}", flush=True)
    print("  COMPARISON", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  CUDA graphs  — avg ckpt: {avg_a['checkpoint']:.2f}s, restore: {avg_a['restore']:.2f}s, cold: {avg_a['cold_start']:.2f}s", flush=True)
    print(f"  Enforce eager — avg ckpt: {avg_b['checkpoint']:.2f}s, restore: {avg_b['restore']:.2f}s, cold: {avg_b['cold_start']:.2f}s", flush=True)
    print(f"", flush=True)
    print(f"  CUDA graphs add to checkpoint: {ckpt_overhead:+.2f}s", flush=True)
    print(f"  CUDA graphs add to restore:    {restore_overhead:+.2f}s", flush=True)
    print(f"  Inference diff (graphs faster): {avg_a['inference'] - avg_b['inference']:.3f}s", flush=True)
    print(f"", flush=True)
    print(f"  GPU memory (initial):", flush=True)
    print(f"    CUDA graphs:  {results_a['mem_initial']} MB", flush=True)
    print(f"    Enforce eager: {results_b['mem_initial']} MB", flush=True)
    print(f"    Difference:    {[a - b for a, b in zip(results_a['mem_initial'], results_b['mem_initial'])]} MB", flush=True)
    print(f"", flush=True)
    print(f"  PROJECTIONS (if Foundry graph loading works):", flush=True)
    for load_time in [2, 3, 4, 5, 8]:
        projected = avg_b["restore"] + load_time + avg_a["inference"]
        savings = avg_a["cold_start"] - projected
        print(f"    Graph load {load_time}s → cold start {projected:.1f}s (saves {savings:.1f}s vs current {avg_a['cold_start']:.1f}s)", flush=True)

    all_match = all(c["match"] for c in results_a["cycles"]) and all(c["match"] for c in results_b["cycles"])
    print(f"\\n  All outputs match: {all_match}", flush=True)
    print(f"  Startup: graphs={results_a['startup_time']:.1f}s, eager={results_b['startup_time']:.1f}s", flush=True)
    print(f"  Reduction vs startup: graphs={((1 - avg_a['cold_start']/results_a['startup_time'])*100):.1f}%, eager={((1 - avg_b['cold_start']/results_b['startup_time'])*100):.1f}%", flush=True)

    print(f"\\nRESULT:" + json.dumps({
        "cuda_graphs": {**avg_a, "startup": results_a["startup_time"], "mem_initial": results_a["mem_initial"]},
        "enforce_eager": {**avg_b, "startup": results_b["startup_time"], "mem_initial": results_b["mem_initial"]},
        "graph_overhead_checkpoint": ckpt_overhead,
        "graph_overhead_restore": restore_overhead,
        "all_match": all_match,
    }), flush=True)

except Exception as e:
    traceback.print_exc()
    print(f"RESULT:" + json.dumps({"error": str(e)}), flush=True)
'''


@app.function(gpu="A100:2", image=image, timeout=1200)
def benchmark():
    import subprocess, sys, json, tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(SERVE_SCRIPT)
        script_path = f.name

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, timeout=1100,
    )
    print(result.stdout)
    if result.stderr:
        for line in result.stderr.strip().split('\n')[-20:]:
            print(f"  [stderr] {line}")

    for line in result.stdout.split('\n'):
        if line.startswith('RESULT:'):
            return json.loads(line[7:])
    return {"error": "no result line", "stderr_tail": result.stderr[-500:] if result.stderr else ""}
