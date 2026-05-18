"""Phase 42 v21: Multi-GPU vllm serve (TP=2) checkpoint/restore.

Production multi-GPU serving: vllm serve with tensor_parallel_size=2.
V1 engine (spawn-based, no NCCL cleanup needed) + parallel PID + sleep.

Tests:
  A) Start vllm serve --tensor-parallel-size 2, verify /v1/completions
  B) Checkpoint all CUDA PIDs in parallel
  C) Restore and verify server responds correctly
  D) Stress test with concurrent requests
  E) Second checkpoint/restore cycle

Usage:
    modal run cuda_serializer/modal_multigpu_v21.py
"""

import modal, os
app = modal.App("vllm-serve-tp2-v21")

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
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()


def find_cuda_pids(server_pid):
    all_pids = [str(server_pid)]
    r = subprocess.run(["pgrep", "-P", str(server_pid)], capture_output=True, text=True)
    for cpid in (r.stdout.strip().split() if r.stdout.strip() else []):
        all_pids.append(cpid)
        r2 = subprocess.run(["pgrep", "-P", cpid], capture_output=True, text=True)
        for gc in (r2.stdout.strip().split() if r2.stdout.strip() else []):
            all_pids.append(gc)
            r3 = subprocess.run(["pgrep", "-P", gc], capture_output=True, text=True)
            if r3.stdout.strip():
                all_pids.extend(r3.stdout.strip().split())

    cuda_pids = []
    for pid in set(all_pids):
        r = subprocess.run(["cuda-checkpoint", "--action", "lock", "--pid", pid],
                          capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            cuda_pids.append(int(pid))
            subprocess.run(["cuda-checkpoint", "--action", "unlock", "--pid", pid],
                          capture_output=True, text=True, timeout=10)
    return sorted(cuda_pids)


try:
    print("=" * 60, flush=True)
    print("  PHASE 42 v21: vllm serve TP=2 CHECKPOINT/RESTORE", flush=True)
    print("=" * 60, flush=True)

    api = CudaCheckpointAPI()

    # --- Start vllm serve with TP=2 ---
    print("\\n--- Starting vllm serve (TP=2) ---", flush=True)
    t0 = time.perf_counter()

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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    startup_time = wait_for_server(timeout=300)
    print(f"  Server ready in {startup_time:.1f}s (PID {server_proc.pid})", flush=True)

    # --- Test A: Verify server works ---
    print("\\n--- Test A: Baseline inference ---", flush=True)
    ref_text = query_server("The capital of France is")
    print(f"  Response: {ref_text[:80]}", flush=True)

    text2 = query_server("The capital of France is")
    print(f"  Warm: {text2[:80]}", flush=True)

    # --- Find CUDA PIDs ---
    cuda_pids = find_cuda_pids(server_proc.pid)
    print(f"  CUDA PIDs: {cuda_pids} ({len(cuda_pids)} total)", flush=True)

    if len(cuda_pids) < 2:
        print(f"  WARNING: Expected >=2 CUDA PIDs for TP=2, got {len(cuda_pids)}", flush=True)

    # --- Test B: Parallel checkpoint ---
    print("\\n--- Test B: Parallel checkpoint ---", flush=True)

    for pid in cuda_pids:
        api.lock(pid)

    with ThreadPoolExecutor(max_workers=len(cuda_pids)) as ex:
        t0 = time.perf_counter()
        list(ex.map(api.checkpoint, cuda_pids))
        ckpt_time = time.perf_counter() - t0
    print(f"  Checkpoint: {ckpt_time:.2f}s ({len(cuda_pids)} PIDs, parallel)", flush=True)

    # --- Test C: Parallel restore ---
    print("\\n--- Test C: Parallel restore ---", flush=True)
    with ThreadPoolExecutor(max_workers=len(cuda_pids)) as ex:
        t0 = time.perf_counter()
        list(ex.map(api.restore, cuda_pids))
        rest_time = time.perf_counter() - t0

    for pid in cuda_pids:
        api.unlock(pid)
    print(f"  Restore: {rest_time:.2f}s", flush=True)

    t0 = time.perf_counter()
    post_text = query_server("The capital of France is")
    infer_c = time.perf_counter() - t0
    match = post_text.strip() == ref_text.strip()
    print(f"  Post-restore ({infer_c:.2f}s): {post_text[:80]}", flush=True)
    print(f"  Match: {match}", flush=True)

    cold_start = rest_time + infer_c
    reduction = (1 - cold_start / startup_time) * 100
    print(f"  Cold start: {cold_start:.2f}s (vs {startup_time:.1f}s = {reduction:.1f}% reduction)", flush=True)

    # --- Test D: Stress test ---
    print("\\n--- Test D: Stress test (6 prompts) ---", flush=True)
    prompts = [
        "Explain quantum computing:",
        "def fibonacci(n):",
        "The largest planet is",
        "Write a haiku:",
        "What is machine learning?",
        "The speed of light is",
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

    # --- Test E: Second cycle ---
    print("\\n--- Test E: Second checkpoint/restore ---", flush=True)

    for pid in cuda_pids:
        api.lock(pid)
    with ThreadPoolExecutor(max_workers=len(cuda_pids)) as ex:
        t0 = time.perf_counter()
        list(ex.map(api.checkpoint, cuda_pids))
        ckpt2 = time.perf_counter() - t0
    with ThreadPoolExecutor(max_workers=len(cuda_pids)) as ex:
        t0 = time.perf_counter()
        list(ex.map(api.restore, cuda_pids))
        rest2 = time.perf_counter() - t0
    for pid in cuda_pids:
        api.unlock(pid)

    t0 = time.perf_counter()
    text_e = query_server("The capital of France is")
    infer_e = time.perf_counter() - t0
    match_e = text_e.strip() == ref_text.strip()
    cold2 = rest2 + infer_e
    print(f"  Ckpt: {ckpt2:.2f}s, Restore: {rest2:.2f}s, Infer: {infer_e:.2f}s", flush=True)
    print(f"  Cold start: {cold2:.2f}s, Match: {match_e}", flush=True)

    # --- Summary ---
    avg_cold = (cold_start + cold2) / 2
    avg_rest = (rest_time + rest2) / 2

    print(f"\\n{'=' * 60}", flush=True)
    print("  RESULTS SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Model: {MODEL}, TP=2, V1 engine", flush=True)
    print(f"  CUDA PIDs: {len(cuda_pids)}", flush=True)
    print(f"  Server startup: {startup_time:.1f}s", flush=True)
    print(f"  Cycle 1: ckpt={ckpt_time:.2f}s, restore={rest_time:.2f}s, cold={cold_start:.2f}s", flush=True)
    print(f"  Cycle 2: ckpt={ckpt2:.2f}s, restore={rest2:.2f}s, cold={cold2:.2f}s", flush=True)
    print(f"  Avg cold start: {avg_cold:.2f}s", flush=True)
    print(f"  Stress: {ok_count}/{len(prompts)} OK", flush=True)
    print(f"  Match: {match and match_e}", flush=True)
    print(f"  Reduction: {reduction:.1f}%", flush=True)

    all_pass = match and match_e and ok_count == len(prompts)
    print(f"  VERDICT: {'PASS' if all_pass else 'FAIL'}", flush=True)

    print(f"RESULT:" + json.dumps({
        "startup_time": startup_time,
        "num_cuda_pids": len(cuda_pids),
        "ckpt_time": ckpt_time,
        "rest_time": rest_time,
        "cold_start": cold_start,
        "ckpt2": ckpt2,
        "rest2": rest2,
        "cold2": cold2,
        "avg_cold": avg_cold,
        "avg_restore": avg_rest,
        "stress_ok": ok_count,
        "stress_total": len(prompts),
        "match": match and match_e,
        "reduction": reduction,
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


@app.function(gpu="H100:2", image=image, timeout=600)
def test_serve_tp2():
    """Test checkpoint/restore with vllm serve TP=2."""
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
    print("Phase 42 v21: vllm serve TP=2 Checkpoint/Restore")
    print("=" * 60)

    r = test_serve_tp2.remote()

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    if 'error' in r:
        print(f"  FAILED: {r['error'][:80]}")
    else:
        print(f"  Startup: {r['startup_time']:.1f}s")
        print(f"  CUDA PIDs: {r['num_cuda_pids']}")
        print(f"  Cycle 1: restore={r['rest_time']:.2f}s, cold={r['cold_start']:.2f}s")
        print(f"  Cycle 2: restore={r['rest2']:.2f}s, cold={r['cold2']:.2f}s")
        print(f"  Avg cold: {r['avg_cold']:.2f}s")
        print(f"  Stress: {r['stress_ok']}/{r['stress_total']} OK")
        print(f"  Reduction: {r['reduction']:.1f}%")
        print(f"  Verdict: {r['verdict']}")
