"""Phase 42 v27: CUDA graphs 10-cycle stability + memory leak detection.

Production validation: CUDA graphs enabled (no --enforce-eager), 10
checkpoint/restore cycles with GPU memory monitoring between cycles.
Proves zero memory leaks and stable cold start times.

Usage:
    modal run cuda_serializer/modal_multigpu_v27.py
"""

import modal, os
app = modal.App("vllm-ckpt-stability-v27")

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

PKG_INIT = '''
from vllm_cuda_ckpt.api import CudaCheckpointAPI, VLLMCheckpointer
from vllm_cuda_ckpt.discover import discover_cuda_pids, find_vllm_server
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

PKG_CLI = r'''
import argparse, json, sys, time
from vllm_cuda_ckpt.api import CudaCheckpointAPI
from vllm_cuda_ckpt.discover import discover_cuda_pids, find_vllm_server
from concurrent.futures import ThreadPoolExecutor

def _resolve_pid(args):
    if args.pid:
        return args.pid
    try:
        pid = find_vllm_server()
        print(f"Auto-discovered vllm serve PID: {pid}")
        return pid
    except RuntimeError as e:
        print(f"ERROR: {e}. Use --pid to specify manually.", file=sys.stderr)
        sys.exit(1)

def _check_health(port, timeout=10.0):
    try:
        import httpx
        r = httpx.get(f"http://localhost:{port}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False

def _query_server(port, prompt, model, max_tokens=32):
    import httpx
    r = httpx.post(f"http://localhost:{port}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()

def _do_checkpoint(api, pids, parallel):
    for pid in pids: api.lock(pid)
    t0 = time.perf_counter()
    if parallel and len(pids) > 1:
        with ThreadPoolExecutor(max_workers=len(pids)) as ex:
            list(ex.map(api.checkpoint, pids))
    else:
        for pid in pids: api.checkpoint(pid)
    return time.perf_counter() - t0

def _do_restore(api, pids, parallel):
    t0 = time.perf_counter()
    if parallel and len(pids) > 1:
        with ThreadPoolExecutor(max_workers=len(pids)) as ex:
            list(ex.map(api.restore, pids))
    else:
        for pid in pids: api.restore(pid)
    rest_time = time.perf_counter() - t0
    for pid in pids: api.unlock(pid)
    return rest_time

def cmd_discover(args):
    pids = discover_cuda_pids(_resolve_pid(args))
    print(f"CUDA PIDs: {pids} ({len(pids)} total)")
    if args.json: print(json.dumps({"pids": pids}))

def cmd_cycle(args):
    api = CudaCheckpointAPI()
    pids = discover_cuda_pids(_resolve_pid(args))
    print(f"CUDA PIDs: {pids} ({len(pids)} total)")
    if not pids: print("ERROR: No CUDA-active PIDs", file=sys.stderr); sys.exit(1)
    parallel = not args.sequential
    ckpt_time = _do_checkpoint(api, pids, parallel)
    print(f"Checkpoint: {ckpt_time:.2f}s")
    rest_time = _do_restore(api, pids, parallel)
    print(f"Restore: {rest_time:.2f}s")
    result = {"action": "cycle", "pids": pids, "checkpoint_time": ckpt_time, "restore_time": rest_time}
    if args.port:
        healthy = _check_health(args.port, timeout=30)
        result["healthy"] = healthy
        print(f"Health: {'OK' if healthy else 'FAIL'}")
        if healthy and args.model:
            t0 = time.perf_counter()
            text = _query_server(args.port, "The capital of France is", args.model)
            infer_time = time.perf_counter() - t0
            result["inference_time"] = infer_time
            result["cold_start"] = rest_time + infer_time
            result["response"] = text[:80]
            print(f"Cold start: {result['cold_start']:.2f}s")
    if args.json: print(json.dumps(result))

def cmd_benchmark(args):
    api = CudaCheckpointAPI()
    pids = discover_cuda_pids(_resolve_pid(args))
    print(f"CUDA PIDs: {pids} ({len(pids)} total)")
    if not pids: print("ERROR: No CUDA-active PIDs", file=sys.stderr); sys.exit(1)
    parallel = not args.sequential
    cycles = []
    for i in range(args.cycles):
        print(f"\n--- Cycle {i+1}/{args.cycles} ---")
        ckpt_time = _do_checkpoint(api, pids, parallel)
        rest_time = _do_restore(api, pids, parallel)
        cycle = {"checkpoint": ckpt_time, "restore": rest_time}
        if args.port and args.model:
            t0 = time.perf_counter()
            text = _query_server(args.port, "The capital of France is", args.model)
            infer_time = time.perf_counter() - t0
            cycle["inference"] = infer_time
            cycle["cold_start"] = rest_time + infer_time
            print(f"  Ckpt: {ckpt_time:.2f}s, Restore: {rest_time:.2f}s, Cold: {cycle['cold_start']:.2f}s")
        else:
            print(f"  Ckpt: {ckpt_time:.2f}s, Restore: {rest_time:.2f}s")
        cycles.append(cycle)
    avg_restore = sum(c["restore"] for c in cycles) / len(cycles)
    print(f"\nAvg restore: {avg_restore:.2f}s ({len(cycles)} cycles)")
    if "cold_start" in cycles[0]:
        print(f"Avg cold start: {sum(c['cold_start'] for c in cycles) / len(cycles):.2f}s")
    if args.json: print(json.dumps({"action": "benchmark", "pids": pids, "cycles": cycles}))

def main():
    parser = argparse.ArgumentParser(prog="vllm-ckpt",
        description="Checkpoint/restore running vLLM serve processes")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in [("discover", cmd_discover), ("cycle", cmd_cycle), ("benchmark", cmd_benchmark)]:
        p = sub.add_parser(name)
        p.add_argument("--pid", type=int, help="vllm serve PID (auto-discovered if omitted)")
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=fn)
        if name != "discover":
            p.add_argument("--sequential", action="store_true")
        if name in ("cycle", "benchmark"):
            p.add_argument("--port", type=int)
            p.add_argument("--model", type=str)
        if name == "benchmark":
            p.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
'''

PKG_PYPROJECT = '''
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "vllm-cold-start"
version = "0.3.0"
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
serve = ["httpx"]

[project.scripts]
vllm-ckpt = "vllm_cuda_ckpt.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
'''

SERVE_SCRIPT = '''
import os, sys, time, json, subprocess, traceback
import httpx

os.environ["CUDA_MODULE_LOADING"] = "EAGER"
os.environ["NCCL_NVLS_ENABLE"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["VLLM_USE_V1"] = "1"
os.environ["PATH"] = "/opt/cuda-checkpoint/bin/x86_64_Linux:" + os.environ["PATH"]

MODEL = "Qwen/Qwen2-7B"
PORT = 8000
BASE_URL = f"http://localhost:{PORT}"
NUM_CYCLES = 10


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
    for line in r.stdout.strip().split("\\n"):
        used, total = line.strip().split(", ")
        gpus.append({"used_mib": int(used), "total_mib": int(total)})
    return gpus


try:
    from vllm_cuda_ckpt import CudaCheckpointAPI, discover_cuda_pids, find_vllm_server
    from concurrent.futures import ThreadPoolExecutor

    print("=" * 60, flush=True)
    print("  PHASE 42 v27: CUDA GRAPHS 10-CYCLE STABILITY", flush=True)
    print("=" * 60, flush=True)

    mem_before_server = get_gpu_memory()
    print(f"  GPU memory before server: {[g['used_mib'] for g in mem_before_server]} MiB", flush=True)

    # --- Start server with CUDA graphs (no --enforce-eager) ---
    print("\\n--- Starting vllm serve (TP=2, CUDA graphs enabled) ---", flush=True)
    server_proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL,
            "--tensor-parallel-size", "2",
            "--gpu-memory-utilization", "0.30",
            "--max-model-len", "512",
            "--port", str(PORT),
            "--disable-custom-all-reduce",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    startup_time = wait_for_server(timeout=600)
    print(f"  Server ready in {startup_time:.1f}s (PID {server_proc.pid})", flush=True)

    mem_after_startup = get_gpu_memory()
    print(f"  GPU memory after startup: {[g['used_mib'] for g in mem_after_startup]} MiB", flush=True)

    ref_text = query_server("The capital of France is")
    print(f"  Baseline: {ref_text[:80]}", flush=True)

    # --- Discover PIDs ---
    auto_pid = find_vllm_server()
    pids = discover_cuda_pids(auto_pid)
    print(f"  CUDA PIDs: {pids} ({len(pids)})", flush=True)

    if not pids:
        raise RuntimeError("No CUDA PIDs found")

    # --- 10-cycle benchmark ---
    api = CudaCheckpointAPI()
    cycles = []
    all_correct = True

    prompts = [
        "The capital of France is",
        "The largest planet in our solar system is",
        "Water boils at",
        "The speed of light is approximately",
        "Python was created by",
        "The capital of Japan is",
        "E = mc",
        "The human body has",
        "Linux was created by",
        "The square root of 144 is",
    ]

    for i in range(NUM_CYCLES):
        prompt = prompts[i % len(prompts)]

        # Memory before checkpoint
        mem_pre = get_gpu_memory()

        # Checkpoint
        for pid in pids:
            api.lock(pid)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(pids)) as ex:
            list(ex.map(api.checkpoint, pids))
        ckpt_time = time.perf_counter() - t0

        mem_ckpt = get_gpu_memory()

        # Restore
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(pids)) as ex:
            list(ex.map(api.restore, pids))
        rest_time = time.perf_counter() - t0
        for pid in pids:
            api.unlock(pid)

        # Health check with retry
        healthy = False
        for attempt in range(3):
            try:
                r = httpx.get(f"{BASE_URL}/health", timeout=10)
                if r.status_code == 200:
                    healthy = True
                    break
            except Exception:
                time.sleep(1)

        # Inference
        t0 = time.perf_counter()
        try:
            text = query_server(prompt)
            infer_time = time.perf_counter() - t0
            infer_ok = True
        except Exception as e:
            text = f"ERROR: {e}"
            infer_time = time.perf_counter() - t0
            infer_ok = False
            all_correct = False

        cold_start = rest_time + infer_time

        # Memory after restore
        mem_post = get_gpu_memory()

        cycle = {
            "cycle": i + 1,
            "checkpoint": round(ckpt_time, 3),
            "restore": round(rest_time, 3),
            "inference": round(infer_time, 3),
            "cold_start": round(cold_start, 3),
            "healthy": healthy,
            "infer_ok": infer_ok,
            "gpu_mem_pre": [g["used_mib"] for g in mem_pre],
            "gpu_mem_ckpt": [g["used_mib"] for g in mem_ckpt],
            "gpu_mem_post": [g["used_mib"] for g in mem_post],
        }
        cycles.append(cycle)

        status = "OK" if (healthy and infer_ok) else "FAIL"
        mem_delta = [mem_post[j]["used_mib"] - mem_pre[j]["used_mib"] for j in range(len(mem_pre))]
        print(f"  Cycle {i+1:2d}/{NUM_CYCLES}: ckpt={ckpt_time:.2f}s rest={rest_time:.2f}s "
              f"cold={cold_start:.2f}s infer={infer_time:.3f}s mem_delta={mem_delta} [{status}]",
              flush=True)

    # --- Analysis ---
    print(f"\\n{'=' * 60}", flush=True)
    print("  STABILITY ANALYSIS", flush=True)
    print(f"{'=' * 60}", flush=True)

    cold_starts = [c["cold_start"] for c in cycles]
    restores = [c["restore"] for c in cycles]
    inferences = [c["inference"] for c in cycles]
    checkpoints = [c["checkpoint"] for c in cycles]

    avg_cold = sum(cold_starts) / len(cold_starts)
    min_cold = min(cold_starts)
    max_cold = max(cold_starts)
    std_cold = (sum((x - avg_cold) ** 2 for x in cold_starts) / len(cold_starts)) ** 0.5

    avg_restore = sum(restores) / len(restores)
    avg_infer = sum(inferences) / len(inferences)
    avg_ckpt = sum(checkpoints) / len(checkpoints)

    # Memory leak detection: compare first and last cycle GPU memory
    first_mem = cycles[0]["gpu_mem_pre"]
    last_mem = cycles[-1]["gpu_mem_post"]
    mem_leak = [last_mem[j] - first_mem[j] for j in range(len(first_mem))]
    has_leak = any(abs(d) > 50 for d in mem_leak)  # >50 MiB = leak

    # Trend detection: is cold start drifting?
    first_half_avg = sum(cold_starts[:5]) / 5
    second_half_avg = sum(cold_starts[5:]) / 5
    drift_pct = abs(second_half_avg - first_half_avg) / first_half_avg * 100

    all_healthy = all(c["healthy"] for c in cycles)
    all_infer_ok = all(c["infer_ok"] for c in cycles)
    reduction = (1 - avg_cold / startup_time) * 100

    verdict = "PASS" if (all_healthy and all_infer_ok and not has_leak and drift_pct < 15) else "FAIL"

    print(f"  Cycles: {NUM_CYCLES}", flush=True)
    print(f"  All healthy: {all_healthy}", flush=True)
    print(f"  All inference OK: {all_infer_ok}", flush=True)
    print(f"  Avg checkpoint: {avg_ckpt:.2f}s", flush=True)
    print(f"  Avg restore: {avg_restore:.2f}s", flush=True)
    print(f"  Avg inference: {avg_infer:.3f}s", flush=True)
    print(f"  Cold start: avg={avg_cold:.2f}s min={min_cold:.2f}s max={max_cold:.2f}s std={std_cold:.2f}s", flush=True)
    print(f"  Reduction: {reduction:.1f}% (vs {startup_time:.1f}s startup)", flush=True)
    print(f"  Memory leak: {mem_leak} MiB ({'LEAK DETECTED' if has_leak else 'none'})", flush=True)
    print(f"  Cold start drift: {drift_pct:.1f}% (1st half {first_half_avg:.2f}s vs 2nd half {second_half_avg:.2f}s)", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)

    result = {
        "startup_time": startup_time,
        "num_cycles": NUM_CYCLES,
        "cuda_graphs": True,
        "num_pids": len(pids),
        "pids": pids,
        "all_healthy": all_healthy,
        "all_infer_ok": all_infer_ok,
        "avg_checkpoint": round(avg_ckpt, 3),
        "avg_restore": round(avg_restore, 3),
        "avg_inference": round(avg_infer, 3),
        "avg_cold_start": round(avg_cold, 3),
        "min_cold_start": round(min_cold, 3),
        "max_cold_start": round(max_cold, 3),
        "std_cold_start": round(std_cold, 3),
        "reduction_pct": round(reduction, 1),
        "mem_leak_mib": mem_leak,
        "has_leak": has_leak,
        "drift_pct": round(drift_pct, 1),
        "cycles": cycles,
        "verdict": verdict,
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


@app.function(gpu="H100:2", image=image, timeout=900)
def test_stability_v27():
    """10-cycle CUDA graphs stability test."""
    import subprocess, sys, json, tempfile

    pkg_dir = "/tmp/vllm_cold_start_pkg"
    src_dir = f"{pkg_dir}/src/vllm_cuda_ckpt"
    os.makedirs(src_dir, exist_ok=True)

    with open(f"{src_dir}/__init__.py", "w") as f:
        f.write(PKG_INIT)
    with open(f"{src_dir}/api.py", "w") as f:
        f.write(PKG_API)
    with open(f"{src_dir}/discover.py", "w") as f:
        f.write(PKG_DISCOVER)
    with open(f"{src_dir}/cli.py", "w") as f:
        f.write(PKG_CLI)
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
        capture_output=True, text=True, timeout=800,
    )
    out = result.stdout[-12000:] if len(result.stdout) > 12000 else result.stdout
    print(out, flush=True)

    for line in result.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:"):])

    return {"error": f"No result (exit={result.returncode})", "stderr": result.stderr[-500:]}


@app.local_entrypoint()
def main():
    print("Phase 42 v27: CUDA graphs 10-cycle stability")
    print("=" * 60)

    r = test_stability_v27.remote()

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    if 'error' in r:
        print(f"  FAILED: {r['error'][:200]}")
        if 'stderr' in r:
            print(f"  stderr: {r['stderr'][:200]}")
    else:
        print(f"  Startup: {r['startup_time']:.1f}s (CUDA graphs)")
        print(f"  CUDA PIDs: {r['num_pids']}")
        print(f"  Cycles: {r['num_cycles']}")
        print(f"  All healthy: {r['all_healthy']}")
        print(f"  All inference OK: {r['all_infer_ok']}")
        print(f"  Avg checkpoint: {r['avg_checkpoint']:.2f}s")
        print(f"  Avg restore: {r['avg_restore']:.2f}s")
        print(f"  Avg inference: {r['avg_inference']:.3f}s")
        print(f"  Cold start: avg={r['avg_cold_start']:.2f}s min={r['min_cold_start']:.2f}s "
              f"max={r['max_cold_start']:.2f}s std={r['std_cold_start']:.2f}s")
        print(f"  Reduction: {r['reduction_pct']:.1f}%")
        print(f"  Memory leak: {r['mem_leak_mib']} MiB ({'LEAK' if r['has_leak'] else 'none'})")
        print(f"  Cold start drift: {r['drift_pct']:.1f}%")
        print(f"  Verdict: {r['verdict']}")
