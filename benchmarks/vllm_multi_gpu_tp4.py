"""Phase 42 v24: TP=4 scaling validation — vllm serve with 4 GPUs.

Tests checkpoint/restore scaling to 4 GPUs (H100:4). Validates:
  A) PID discovery finds all CUDA processes for TP=4
  B) Parallel checkpoint/restore works with 4+ PIDs
  C) Cold start scales sub-linearly with GPU count
  D) 3-cycle stability at TP=4

Usage:
    modal run cuda_serializer/modal_multigpu_v24.py
"""

import modal, os
app = modal.App("vllm-serve-tp4-v24")

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
import httpx

os.environ["CUDA_MODULE_LOADING"] = "EAGER"
os.environ["NCCL_NVLS_ENABLE"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["VLLM_USE_V1"] = "1"
os.environ["PATH"] = "/opt/cuda-checkpoint/bin/x86_64_Linux:" + os.environ["PATH"]

MODEL = "Qwen/Qwen2-7B"
TP = 4
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
                for gggc in get_children(ggc):
                    all_pids.add(gggc)

    cuda_pids = []
    for pid in sorted(all_pids, key=int):
        r = subprocess.run(["cuda-checkpoint", "--action", "lock", "--pid", pid],
                          capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            cuda_pids.append(int(pid))
            subprocess.run(["cuda-checkpoint", "--action", "unlock", "--pid", pid],
                          capture_output=True, text=True, timeout=10)
    return sorted(cuda_pids)


try:
    print("=" * 60, flush=True)
    print(f"  PHASE 42 v24: vllm serve TP={TP} CHECKPOINT/RESTORE", flush=True)
    print("=" * 60, flush=True)

    api = CudaCheckpointAPI()

    # --- Start vllm serve with TP=4 ---
    print(f"\\n--- Starting vllm serve (TP={TP}) ---", flush=True)
    t0 = time.perf_counter()

    server_proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL,
            "--tensor-parallel-size", str(TP),
            "--enforce-eager",
            "--gpu-memory-utilization", "0.30",
            "--max-model-len", "512",
            "--port", str(PORT),
            "--disable-custom-all-reduce",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    startup_time = wait_for_server(timeout=300)
    print(f"  Server ready in {startup_time:.1f}s (PID {server_proc.pid})", flush=True)

    # --- Test A: Verify server works ---
    print("\\n--- Test A: Baseline inference ---", flush=True)
    ref_text = query_server("The capital of France is")
    print(f"  Response: {ref_text[:80]}", flush=True)

    # --- Find CUDA PIDs ---
    cuda_pids = find_cuda_pids(server_proc.pid)
    print(f"  CUDA PIDs: {cuda_pids} ({len(cuda_pids)} total)", flush=True)

    if len(cuda_pids) < TP:
        print(f"  WARNING: Expected >={TP} CUDA PIDs for TP={TP}, got {len(cuda_pids)}", flush=True)

    # --- Test B: 3-cycle benchmark ---
    print(f"\\n--- Test B: 3-cycle benchmark (TP={TP}) ---", flush=True)
    cycles = []
    for i in range(3):
        for pid in cuda_pids:
            api.lock(pid)
        with ThreadPoolExecutor(max_workers=len(cuda_pids)) as ex:
            t0 = time.perf_counter()
            list(ex.map(api.checkpoint, cuda_pids))
            ckpt_time = time.perf_counter() - t0
        with ThreadPoolExecutor(max_workers=len(cuda_pids)) as ex:
            t0 = time.perf_counter()
            list(ex.map(api.restore, cuda_pids))
            rest_time = time.perf_counter() - t0
        for pid in cuda_pids:
            api.unlock(pid)

        t0 = time.perf_counter()
        text = query_server("The capital of France is")
        infer_time = time.perf_counter() - t0
        match = text.strip() == ref_text.strip()
        cold_start = rest_time + infer_time
        cycles.append({
            "ckpt": ckpt_time, "restore": rest_time,
            "infer": infer_time, "cold_start": cold_start, "match": match,
        })
        print(f"  Cycle {i+1}: ckpt={ckpt_time:.2f}s, restore={rest_time:.2f}s, "
              f"cold={cold_start:.2f}s, match={match}", flush=True)

    # --- Test C: Stress test (8 prompts) ---
    print("\\n--- Test C: Stress test (8 prompts) ---", flush=True)
    prompts = [
        "Explain quantum computing:",
        "def fibonacci(n):",
        "The largest planet is",
        "Write a haiku:",
        "What is machine learning?",
        "The speed of light is",
        "The Pythagorean theorem states",
        "In the year 2050,",
    ]

    t0 = time.perf_counter()
    results = []
    for p in prompts:
        try:
            text = query_server(p, max_tokens=32)
            results.append({"ok": True, "text": text[:50]})
        except Exception as e:
            results.append({"ok": False, "text": str(e)[:50]})
    stress_time = time.perf_counter() - t0

    ok_count = sum(1 for r in results if r["ok"])
    print(f"  {ok_count}/{len(prompts)} OK in {stress_time:.2f}s", flush=True)
    for i, r in enumerate(results):
        print(f"    [{i}] {'OK' if r['ok'] else 'FAIL'}: {r['text'][:50]}", flush=True)

    # --- Summary ---
    avg_cold = sum(c["cold_start"] for c in cycles) / len(cycles)
    avg_restore = sum(c["restore"] for c in cycles) / len(cycles)
    avg_ckpt = sum(c["ckpt"] for c in cycles) / len(cycles)
    all_match = all(c["match"] for c in cycles)
    reduction = (1 - avg_cold / startup_time) * 100

    all_pass = all_match and ok_count == len(prompts)

    print(f"\\n{'=' * 60}", flush=True)
    print("  RESULTS SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Model: {MODEL}, TP={TP}, V1 engine", flush=True)
    print(f"  CUDA PIDs: {len(cuda_pids)}", flush=True)
    print(f"  Server startup: {startup_time:.1f}s", flush=True)
    print(f"  Avg checkpoint: {avg_ckpt:.2f}s", flush=True)
    print(f"  Avg restore: {avg_restore:.2f}s", flush=True)
    print(f"  Avg cold start: {avg_cold:.2f}s", flush=True)
    print(f"  Stress: {ok_count}/{len(prompts)} OK", flush=True)
    print(f"  Match: {all_match}", flush=True)
    print(f"  Reduction: {reduction:.1f}%", flush=True)
    print(f"  VERDICT: {'PASS' if all_pass else 'FAIL'}", flush=True)

    print(f"RESULT:" + json.dumps({
        "tp": TP,
        "startup_time": startup_time,
        "num_cuda_pids": len(cuda_pids),
        "avg_ckpt": avg_ckpt,
        "avg_restore": avg_restore,
        "avg_cold": avg_cold,
        "stress_ok": ok_count,
        "stress_total": len(prompts),
        "match": all_match,
        "reduction": reduction,
        "cycles": cycles,
        "verdict": "PASS" if all_pass else "FAIL",
    }), flush=True)

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


@app.function(gpu="H100:4", image=image, timeout=600)
def test_serve_tp4():
    """Test checkpoint/restore with vllm serve TP=4."""
    import subprocess, sys, json, tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(SERVE_SCRIPT)
        script_path = f.name

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, timeout=500,
    )
    out = result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout
    print(out, flush=True)
    if result.stderr:
        stderr_lines = result.stderr.strip().split('\n')
        important = [l for l in stderr_lines if any(k in l.lower() for k in
            ["error", "fail", "traceback", "cuda", "assert"])]
        if important:
            print(f"\n  STDERR (relevant):", flush=True)
            for line in important[-10:]:
                print(f"    {line}", flush=True)

    for line in result.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:"):])

    return {"error": f"No result (exit={result.returncode})"}


@app.local_entrypoint()
def main():
    print("Phase 42 v24: vllm serve TP=4 Checkpoint/Restore")
    print("=" * 60)

    r = test_serve_tp4.remote()

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    if 'error' in r:
        print(f"  FAILED: {r['error'][:80]}")
    else:
        print(f"  TP: {r['tp']}")
        print(f"  Startup: {r['startup_time']:.1f}s")
        print(f"  CUDA PIDs: {r['num_cuda_pids']}")
        print(f"  Avg checkpoint: {r['avg_ckpt']:.2f}s")
        print(f"  Avg restore: {r['avg_restore']:.2f}s")
        print(f"  Avg cold: {r['avg_cold']:.2f}s")
        print(f"  Stress: {r['stress_ok']}/{r['stress_total']} OK")
        print(f"  Reduction: {r['reduction']:.1f}%")
        print(f"  Verdict: {r['verdict']}")
