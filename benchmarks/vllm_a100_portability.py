"""Phase 42 v33: A100 GPU hardware portability validation.

Tests checkpoint/restore on A100x2 (sm_80) vs our H100 (sm_90) baseline.
Uses Qwen2-7B + enforce-eager + TP=2 for direct comparison with H100 results.
Also captures GPU info and driver version for hardware characterization.

Usage:
    modal run cuda_serializer/modal_multigpu_v33.py
"""

import modal, os
app = modal.App("vllm-ckpt-a100-v33")

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
        "python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-7B-Instruct')\"",
    )
)

PKG_INIT = '''
from gpu_checkpoint_orchestrator.api import CudaCheckpointAPI, VLLMCheckpointer
from gpu_checkpoint_orchestrator.discover import discover_cuda_pids, find_vllm_server
__all__ = ["CudaCheckpointAPI", "VLLMCheckpointer", "discover_cuda_pids", "find_vllm_server"]
'''

PKG_DISCOVER = r'''
import subprocess

def find_vllm_server():
    r = subprocess.run(["pgrep", "-f", "vllm.entrypoints.openai.api_server"],
                      capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError("No vllm serve process found")
    pids = r.stdout.strip().split()
    if len(pids) > 1:
        r2 = subprocess.run(["pgrep", "-f", "vllm.entrypoints.openai.api_server", "--oldest"],
                           capture_output=True, text=True)
        if r2.returncode == 0 and r2.stdout.strip():
            return int(r2.stdout.strip().split()[0])
    return int(pids[0])

def discover_cuda_pids(server_pid):
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
'''

PKG_API = r'''
import ctypes, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor

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
    def lock(self, pid):
        rc = self._fn_lock(pid, ctypes.byref(self._make_args()))
        if rc != 0: raise RuntimeError(f"Lock PID {pid}: rc={rc}")
    def checkpoint(self, pid):
        rc = self._fn_checkpoint(pid, ctypes.byref(self._make_args()))
        if rc != 0: raise RuntimeError(f"Checkpoint PID {pid}: rc={rc}")
    def restore(self, pid):
        rc = self._fn_restore(pid, ctypes.byref(self._make_args()))
        if rc != 0: raise RuntimeError(f"Restore PID {pid}: rc={rc}")
    def unlock(self, pid):
        rc = self._fn_unlock(pid, ctypes.byref(self._make_args()))
        if rc != 0: raise RuntimeError(f"Unlock PID {pid}: rc={rc}")

class VLLMCheckpointer:
    pass
'''

PKG_PYPROJECT = '''
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "vllm-cold-start"
version = "0.4.0"
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
serve = ["httpx"]

[project.scripts]
vllm-ckpt = "gpu_checkpoint_orchestrator.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
'''

SERVE_SCRIPT = r'''
import os, sys, time, json, subprocess, traceback
import httpx

os.environ["CUDA_MODULE_LOADING"] = "EAGER"
os.environ["NCCL_NVLS_ENABLE"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["VLLM_USE_V1"] = "1"
os.environ["PATH"] = "/opt/cuda-checkpoint/bin/x86_64_Linux:" + os.environ["PATH"]

MODEL = "Qwen/Qwen2-7B-Instruct"
PORT = 8000
BASE_URL = f"http://localhost:{PORT}"
NUM_CYCLES = 5


def get_gpu_info():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10,
    )
    gpus = []
    for line in r.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(", ")]
        gpus.append({
            "name": parts[0], "driver": parts[1],
            "memory_mib": parts[2], "compute_cap": parts[3],
        })
    return gpus


def wait_for_server(timeout=300):
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
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()


def get_gpu_memory():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    gpus = []
    for line in r.stdout.strip().split("\n"):
        used, total = line.strip().split(", ")
        gpus.append({"used_mib": int(used), "total_mib": int(total)})
    return gpus


try:
    from gpu_checkpoint_orchestrator import CudaCheckpointAPI, discover_cuda_pids, find_vllm_server
    from concurrent.futures import ThreadPoolExecutor

    gpu_info = get_gpu_info()
    print("=" * 60, flush=True)
    print("  PHASE 42 v33: A100 HARDWARE PORTABILITY", flush=True)
    print("=" * 60, flush=True)
    for i, g in enumerate(gpu_info):
        print(f"  GPU {i}: {g['name']} ({g['memory_mib']}, sm_{g['compute_cap']}, driver {g['driver']})", flush=True)

    print(f"\n--- Starting vllm serve (Qwen2-7B, enforce-eager, TP=2) ---", flush=True)
    server_proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL,
            "--tensor-parallel-size", "2",
            "--enforce-eager",
            "--gpu-memory-utilization", "0.30",
            "--max-model-len", "512",
            "--port", str(PORT),
            "--disable-custom-all-reduce",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    startup_time = wait_for_server(timeout=300)
    print(f"  Server ready in {startup_time:.1f}s (PID {server_proc.pid})", flush=True)

    mem_after = get_gpu_memory()
    print(f"  GPU memory: {[g['used_mib'] for g in mem_after]} MiB", flush=True)

    ref_text = query_server("The capital of France is")
    print(f"  Baseline: {ref_text[:80]}", flush=True)

    pids = discover_cuda_pids(find_vllm_server())
    print(f"  CUDA PIDs: {pids} ({len(pids)})", flush=True)

    api = CudaCheckpointAPI()
    cycles = []
    all_correct = True

    for i in range(NUM_CYCLES):
        mem_pre = get_gpu_memory()

        for pid in pids:
            api.lock(pid)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(pids)) as ex:
            list(ex.map(api.checkpoint, pids))
        ckpt_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(pids)) as ex:
            list(ex.map(api.restore, pids))
        rest_time = time.perf_counter() - t0
        for pid in pids:
            api.unlock(pid)

        t0 = time.perf_counter()
        text = query_server("The capital of France is")
        infer_time = time.perf_counter() - t0
        cold_start = rest_time + infer_time
        match = text.strip() == ref_text.strip()
        if not match:
            all_correct = False

        mem_post = get_gpu_memory()
        mem_delta = [mem_post[j]["used_mib"] - mem_pre[j]["used_mib"] for j in range(len(mem_pre))]

        cycles.append({
            "cycle": i + 1, "checkpoint": round(ckpt_time, 3),
            "restore": round(rest_time, 3), "inference": round(infer_time, 3),
            "cold_start": round(cold_start, 3), "match": match, "mem_delta": mem_delta,
        })

        status = "OK" if match else "FAIL"
        print(f"  Cycle {i+1}/{NUM_CYCLES}: ckpt={ckpt_time:.2f}s rest={rest_time:.2f}s "
              f"cold={cold_start:.2f}s infer={infer_time:.3f}s mem_delta={mem_delta} [{status}]",
              flush=True)

    # Stress test
    from concurrent.futures import as_completed
    stress_prompts = ["Python was created by", "Water boils at", "The speed of light is",
                     "2 + 2 equals", "The largest ocean is", "Linux was created by"]
    stress_ok = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(query_server, p): p for p in stress_prompts}
        for f in as_completed(futures):
            try:
                f.result()
                stress_ok += 1
            except Exception:
                pass
    print(f"\n  Stress: {stress_ok}/{len(stress_prompts)} OK", flush=True)

    # Summary
    cold_starts = [c["cold_start"] for c in cycles]
    avg_cold = sum(cold_starts) / len(cold_starts)
    avg_infer = sum(c["inference"] for c in cycles) / len(cycles)
    avg_rest = sum(c["restore"] for c in cycles) / len(cycles)
    has_leak = any(abs(d) > 50 for c in cycles for d in c["mem_delta"])
    reduction = (1 - avg_cold / startup_time) * 100
    all_pass = all_correct and not has_leak and stress_ok == len(stress_prompts)

    print(f"\n{'=' * 60}", flush=True)
    print("  RESULTS SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  GPU: {gpu_info[0]['name']} (sm_{gpu_info[0]['compute_cap']})", flush=True)
    print(f"  Driver: {gpu_info[0]['driver']}", flush=True)
    print(f"  Model: {MODEL}", flush=True)
    print(f"  CUDA graphs: disabled (eager)", flush=True)
    print(f"  TP: 2, PIDs: {len(pids)}", flush=True)
    print(f"  Startup: {startup_time:.1f}s", flush=True)
    print(f"  Avg restore: {avg_rest:.2f}s", flush=True)
    print(f"  Avg inference: {avg_infer:.3f}s", flush=True)
    print(f"  Avg cold start: {avg_cold:.2f}s", flush=True)
    print(f"  Reduction: {reduction:.1f}%", flush=True)
    print(f"  Memory leak: {'DETECTED' if has_leak else 'none'}", flush=True)
    print(f"  All correct: {all_correct}", flush=True)
    print(f"  Stress: {stress_ok}/{len(stress_prompts)}", flush=True)
    print(f"  VERDICT: {'PASS' if all_pass else 'FAIL'}", flush=True)

    result = {
        "gpu": gpu_info[0]["name"], "compute_cap": gpu_info[0]["compute_cap"],
        "driver": gpu_info[0]["driver"],
        "model": MODEL, "cuda_graphs": False, "tp": 2,
        "startup_time": startup_time, "num_pids": len(pids),
        "avg_restore": round(avg_rest, 3), "avg_inference": round(avg_infer, 3),
        "avg_cold_start": round(avg_cold, 3), "reduction_pct": round(reduction, 1),
        "has_leak": has_leak, "all_correct": all_correct,
        "stress_ok": stress_ok, "stress_total": len(stress_prompts),
        "cycles": cycles, "verdict": "PASS" if all_pass else "FAIL",
    }
    print(f"RESULT:" + json.dumps(result), flush=True)

    server_proc.terminate()
    server_proc.wait(timeout=10)

except Exception as e:
    traceback.print_exc()
    print(f"RESULT:" + json.dumps({"error": str(e)}), flush=True)
    try:
        server_proc.terminate()
    except:
        pass
'''


@app.function(gpu="A100:2", image=image, timeout=1200)
def test_a100_v33():
    """Qwen2-7B + enforce-eager + TP=2 on A100x2."""
    import subprocess, sys, json, tempfile

    pkg_dir = "/tmp/vllm_cold_start_pkg"
    src_dir = f"{pkg_dir}/src/gpu_checkpoint_orchestrator"
    os.makedirs(src_dir, exist_ok=True)

    with open(f"{src_dir}/__init__.py", "w") as f:
        f.write(PKG_INIT)
    with open(f"{src_dir}/api.py", "w") as f:
        f.write(PKG_API)
    with open(f"{src_dir}/discover.py", "w") as f:
        f.write(PKG_DISCOVER)
    with open(f"{pkg_dir}/pyproject.toml", "w") as f:
        f.write(PKG_PYPROJECT)

    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", pkg_dir],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        return {"error": f"pip install failed: {r.stderr[-200:]}"}
    print("pip install OK", flush=True)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(SERVE_SCRIPT)
        serve_path = f.name

    result = subprocess.run(
        [sys.executable, serve_path],
        capture_output=True, text=True, timeout=1100,
    )
    out = result.stdout[-12000:] if len(result.stdout) > 12000 else result.stdout
    print(out, flush=True)

    for line in result.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:"):])

    return {"error": f"No result (exit={result.returncode})", "stderr": result.stderr[-500:]}


@app.local_entrypoint()
def main():
    print("Phase 42 v33: A100 GPU hardware portability")
    print("=" * 60)

    r = test_a100_v33.remote()

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    if 'error' in r:
        print(f"  FAILED: {r['error'][:200]}")
        if 'stderr' in r:
            print(f"  stderr: {r['stderr'][:200]}")
    else:
        print(f"  GPU: {r['gpu']} (sm_{r['compute_cap']})")
        print(f"  Driver: {r['driver']}")
        print(f"  Model: {r['model']}")
        print(f"  CUDA graphs: {r['cuda_graphs']}")
        print(f"  TP: {r['tp']}, PIDs: {r['num_pids']}")
        print(f"  Startup: {r['startup_time']:.1f}s")
        print(f"  Avg restore: {r['avg_restore']:.2f}s")
        print(f"  Avg inference: {r['avg_inference']:.3f}s")
        print(f"  Avg cold start: {r['avg_cold_start']:.2f}s")
        print(f"  Reduction: {r['reduction_pct']:.1f}%")
        print(f"  Leak: {'DETECTED' if r['has_leak'] else 'none'}")
        print(f"  Stress: {r['stress_ok']}/{r['stress_total']}")
        print(f"  Verdict: {r['verdict']}")
