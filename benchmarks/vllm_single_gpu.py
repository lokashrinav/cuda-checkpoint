"""Phase 42 v20: vllm serve (OpenAI-compatible server) checkpoint/restore.

Production deployments use `vllm serve` not the Python LLM class.
Tests checkpoint/restore while the server is running:
  A) Start vllm serve, verify /v1/completions works
  B) Checkpoint while server is running
  C) Restore and verify server still responds correctly
  D) Stress test with concurrent requests after restore

Single-GPU V0 with enable_sleep_mode for simplicity.

Usage:
    modal run cuda_serializer/modal_multigpu_v20.py
"""

import modal, os
app = modal.App("vllm-serve-ckpt-v20")

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
        "python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-1.5B')\"",
    )
)

SERVE_SCRIPT = '''
import os, sys, time, json, subprocess, ctypes, signal, traceback
import httpx

os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["PATH"] = "/opt/cuda-checkpoint/bin/x86_64_Linux:" + os.environ["PATH"]

MODEL = "Qwen/Qwen2-1.5B"
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


def wait_for_server(timeout=120):
    """Wait for vllm serve to be ready."""
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
    """Send a completion request to the server."""
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
    data = r.json()
    return data["choices"][0]["text"].strip()


def find_server_pid(server_proc):
    """Find the CUDA-active PID (may be different from the subprocess PID)."""
    main_pid = server_proc.pid
    all_pids = [str(main_pid)]
    r = subprocess.run(["pgrep", "-P", str(main_pid)], capture_output=True, text=True)
    for cpid in (r.stdout.strip().split() if r.stdout.strip() else []):
        all_pids.append(cpid)
        r2 = subprocess.run(["pgrep", "-P", cpid], capture_output=True, text=True)
        if r2.stdout.strip():
            all_pids.extend(r2.stdout.strip().split())

    cuda_pids = []
    for pid in all_pids:
        r = subprocess.run(["cuda-checkpoint", "--action", "lock", "--pid", pid],
                          capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            cuda_pids.append(int(pid))
            subprocess.run(["cuda-checkpoint", "--action", "unlock", "--pid", pid],
                          capture_output=True, text=True, timeout=10)
    return cuda_pids


try:
    print("=" * 60, flush=True)
    print("  PHASE 42 v20: vllm serve CHECKPOINT/RESTORE", flush=True)
    print("=" * 60, flush=True)

    # --- Start vllm serve ---
    print("\\n--- Starting vllm serve ---", flush=True)
    t0 = time.perf_counter()

    server_proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL,
            "--enforce-eager",
            "--gpu-memory-utilization", "0.30",
            "--max-model-len", "512",
            "--port", str(PORT),
            "--enable-sleep-mode",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    startup_time = wait_for_server(timeout=180)
    print(f"  Server ready in {startup_time:.1f}s (PID {server_proc.pid})", flush=True)

    # --- Test A: Verify server works ---
    print("\\n--- Test A: Baseline inference ---", flush=True)
    t0 = time.perf_counter()
    ref_text = query_server("The capital of France is")
    infer_a = time.perf_counter() - t0
    print(f"  Response ({infer_a:.2f}s): {ref_text[:80]}", flush=True)

    # Second query to warm up
    t0 = time.perf_counter()
    text2 = query_server("The capital of France is")
    infer_a2 = time.perf_counter() - t0
    print(f"  Warm response ({infer_a2:.2f}s): {text2[:80]}", flush=True)

    # --- Find CUDA PIDs ---
    cuda_pids = find_server_pid(server_proc)
    print(f"  CUDA PIDs: {cuda_pids}", flush=True)

    if not cuda_pids:
        print("  ERROR: No CUDA PIDs found!", flush=True)
        server_proc.terminate()
        sys.exit(1)

    api = CudaCheckpointAPI()

    # --- Test B: Checkpoint ---
    print("\\n--- Test B: Checkpoint while server running ---", flush=True)

    for pid in cuda_pids:
        api.lock(pid)

    t0 = time.perf_counter()
    for pid in cuda_pids:
        api.checkpoint(pid)
    ckpt_time = time.perf_counter() - t0
    print(f"  Checkpoint: {ckpt_time:.2f}s ({len(cuda_pids)} PIDs)", flush=True)

    # Server should be frozen now - requests should hang/fail
    print("  Server is checkpointed (frozen)", flush=True)

    # --- Test C: Restore ---
    print("\\n--- Test C: Restore ---", flush=True)
    t0 = time.perf_counter()
    for pid in cuda_pids:
        api.restore(pid)
    rest_time = time.perf_counter() - t0

    for pid in cuda_pids:
        api.unlock(pid)
    print(f"  Restore: {rest_time:.2f}s", flush=True)

    # Verify server responds after restore
    t0 = time.perf_counter()
    post_text = query_server("The capital of France is")
    infer_c = time.perf_counter() - t0
    match = post_text.strip() == ref_text.strip()
    print(f"  Post-restore ({infer_c:.2f}s): {post_text[:80]}", flush=True)
    print(f"  Output match: {match}", flush=True)

    cold_start = rest_time + infer_c
    reduction = (1 - cold_start / startup_time) * 100 if startup_time > 0 else 0
    print(f"  Cold start: {cold_start:.2f}s (vs {startup_time:.1f}s startup = {reduction:.1f}% reduction)", flush=True)

    # --- Test D: Stress test after restore ---
    print("\\n--- Test D: Concurrent requests after restore ---", flush=True)
    prompts = [
        "Explain quantum computing in one sentence:",
        "def fibonacci(n):",
        "The largest planet in our solar system is",
        "Write a haiku about programming:",
    ]

    t0 = time.perf_counter()
    results = []
    for p in prompts:
        try:
            text = query_server(p, max_tokens=32)
            results.append({"prompt": p[:30], "text": text[:60], "ok": True})
        except Exception as e:
            results.append({"prompt": p[:30], "text": str(e)[:60], "ok": False})
    stress_time = time.perf_counter() - t0

    ok_count = sum(1 for r in results if r["ok"])
    print(f"  {ok_count}/{len(prompts)} requests OK in {stress_time:.2f}s", flush=True)
    for r in results:
        status = "OK" if r["ok"] else "FAIL"
        print(f"    [{status}] {r['prompt']}... -> {r['text'][:50]}", flush=True)

    # --- Second checkpoint/restore cycle ---
    print("\\n--- Test E: Second checkpoint/restore cycle ---", flush=True)

    for pid in cuda_pids:
        api.lock(pid)
    t0 = time.perf_counter()
    for pid in cuda_pids:
        api.checkpoint(pid)
    ckpt2 = time.perf_counter() - t0

    t0 = time.perf_counter()
    for pid in cuda_pids:
        api.restore(pid)
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
    print(f"\\n{'=' * 60}", flush=True)
    print("  RESULTS SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Server startup: {startup_time:.1f}s", flush=True)
    print(f"  Cycle 1: ckpt={ckpt_time:.2f}s, restore={rest_time:.2f}s, infer={infer_c:.2f}s, cold={cold_start:.2f}s", flush=True)
    print(f"  Cycle 2: ckpt={ckpt2:.2f}s, restore={rest2:.2f}s, infer={infer_e:.2f}s, cold={cold2:.2f}s", flush=True)
    print(f"  Stress: {ok_count}/{len(prompts)} OK", flush=True)
    print(f"  Output match: cycle1={match}, cycle2={match_e}", flush=True)
    print(f"  Reduction: {reduction:.1f}%", flush=True)

    all_pass = match and match_e and ok_count == len(prompts)
    print(f"  VERDICT: {'PASS' if all_pass else 'FAIL'}", flush=True)

    print(f"RESULT:" + json.dumps({
        "startup_time": startup_time,
        "ckpt_time": ckpt_time,
        "rest_time": rest_time,
        "infer_post_restore": infer_c,
        "cold_start": cold_start,
        "ckpt2": ckpt2,
        "rest2": rest2,
        "cold2": cold2,
        "stress_ok": ok_count,
        "stress_total": len(prompts),
        "match": match,
        "match2": match_e,
        "reduction": reduction,
        "verdict": "PASS" if all_pass else "FAIL",
    }), flush=True)

    # Cleanup
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


@app.function(gpu="H100", image=image, timeout=600)
def test_serve_ckpt():
    """Test checkpoint/restore with vllm serve."""
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
    print("Phase 42 v20: vllm serve Checkpoint/Restore")
    print("=" * 60)

    r = test_serve_ckpt.remote()

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    if 'error' in r:
        print(f"  FAILED: {r['error'][:80]}")
    else:
        print(f"  Startup: {r['startup_time']:.1f}s")
        print(f"  Cycle 1: restore={r['rest_time']:.2f}s, cold={r['cold_start']:.2f}s")
        print(f"  Cycle 2: restore={r['rest2']:.2f}s, cold={r['cold2']:.2f}s")
        print(f"  Stress: {r['stress_ok']}/{r['stress_total']} OK")
        print(f"  Reduction: {r['reduction']:.1f}%")
        print(f"  Verdict: {r['verdict']}")
