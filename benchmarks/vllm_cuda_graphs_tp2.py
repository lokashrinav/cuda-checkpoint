"""Phase 42 v25: CUDA graphs + TP=2 — production default configuration.

All prior multi-GPU tests used --enforce-eager. Production vllm serve uses
CUDA graphs by default. This test validates checkpoint/restore with CUDA
graphs enabled on V1 engine TP=2.

CUDA graphs add significant checkpoint overhead (45s vs 23s single-GPU),
but they survive checkpoint/restore (proven in Phase 38d single-GPU).
This test verifies multi-GPU CUDA graph checkpoint/restore.

Usage:
    modal run cuda_serializer/modal_multigpu_v25.py
"""

import modal, os
app = modal.App("vllm-serve-cudagraphs-v25")

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


try:
    print("=" * 60, flush=True)
    print("  PHASE 42 v25: CUDA GRAPHS + TP=2 (production default)", flush=True)
    print("=" * 60, flush=True)

    api = CudaCheckpointAPI()

    # --- Config A: CUDA graphs (production default) ---
    print("\\n--- Config A: CUDA graphs + TP=2 (0.30 util) ---", flush=True)
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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    startup_time_a = wait_for_server(timeout=600)
    print(f"  Server ready in {startup_time_a:.1f}s (PID {server_proc.pid})", flush=True)

    ref_text = query_server("The capital of France is")
    print(f"  Baseline: {ref_text[:80]}", flush=True)

    cuda_pids = find_cuda_pids(server_proc.pid)
    print(f"  CUDA PIDs: {cuda_pids} ({len(cuda_pids)} total)", flush=True)

    # Cycle 1
    for pid in cuda_pids:
        api.lock(pid)
    with ThreadPoolExecutor(max_workers=len(cuda_pids)) as ex:
        t0 = time.perf_counter()
        list(ex.map(api.checkpoint, cuda_pids))
        ckpt_a = time.perf_counter() - t0
    with ThreadPoolExecutor(max_workers=len(cuda_pids)) as ex:
        t0 = time.perf_counter()
        list(ex.map(api.restore, cuda_pids))
        rest_a = time.perf_counter() - t0
    for pid in cuda_pids:
        api.unlock(pid)

    t0 = time.perf_counter()
    post_text = query_server("The capital of France is")
    infer_a = time.perf_counter() - t0
    match_a = post_text.strip() == ref_text.strip()
    cold_a = rest_a + infer_a
    print(f"  Checkpoint: {ckpt_a:.2f}s", flush=True)
    print(f"  Restore: {rest_a:.2f}s", flush=True)
    print(f"  Post-restore infer: {infer_a:.2f}s", flush=True)
    print(f"  Cold start: {cold_a:.2f}s, Match: {match_a}", flush=True)
    reduction_a = (1 - cold_a / startup_time_a) * 100
    print(f"  Reduction: {reduction_a:.1f}%", flush=True)

    # Cycle 2
    for pid in cuda_pids:
        api.lock(pid)
    with ThreadPoolExecutor(max_workers=len(cuda_pids)) as ex:
        t0 = time.perf_counter()
        list(ex.map(api.checkpoint, cuda_pids))
        ckpt_a2 = time.perf_counter() - t0
    with ThreadPoolExecutor(max_workers=len(cuda_pids)) as ex:
        t0 = time.perf_counter()
        list(ex.map(api.restore, cuda_pids))
        rest_a2 = time.perf_counter() - t0
    for pid in cuda_pids:
        api.unlock(pid)

    t0 = time.perf_counter()
    text_a2 = query_server("The capital of France is")
    infer_a2 = time.perf_counter() - t0
    match_a2 = text_a2.strip() == ref_text.strip()
    cold_a2 = rest_a2 + infer_a2
    print(f"  Cycle 2: ckpt={ckpt_a2:.2f}s, restore={rest_a2:.2f}s, cold={cold_a2:.2f}s, match={match_a2}", flush=True)

    # Stress test
    print("\\n--- Stress test (6 prompts) ---", flush=True)
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

    server_proc.terminate()
    server_proc.wait(timeout=10)

    # --- Summary ---
    avg_cold = (cold_a + cold_a2) / 2

    all_pass = match_a and match_a2 and ok_count == len(prompts)

    print(f"\\n{'=' * 60}", flush=True)
    print("  RESULTS SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Config: CUDA graphs (default) + TP=2, V1 engine", flush=True)
    print(f"  CUDA PIDs: {len(cuda_pids)}", flush=True)
    print(f"  Server startup: {startup_time_a:.1f}s", flush=True)
    print(f"  Cycle 1: ckpt={ckpt_a:.2f}s, restore={rest_a:.2f}s, cold={cold_a:.2f}s", flush=True)
    print(f"  Cycle 2: ckpt={ckpt_a2:.2f}s, restore={rest_a2:.2f}s, cold={cold_a2:.2f}s", flush=True)
    print(f"  Avg cold start: {avg_cold:.2f}s", flush=True)
    print(f"  Stress: {ok_count}/{len(prompts)} OK", flush=True)
    print(f"  Match: {match_a and match_a2}", flush=True)
    print(f"  Reduction: {reduction_a:.1f}%", flush=True)
    print(f"  VERDICT: {'PASS' if all_pass else 'FAIL'}", flush=True)

    print(f"RESULT:" + json.dumps({
        "config": "cuda_graphs_tp2",
        "startup_time": startup_time_a,
        "num_cuda_pids": len(cuda_pids),
        "ckpt1": ckpt_a, "rest1": rest_a, "cold1": cold_a,
        "ckpt2": ckpt_a2, "rest2": rest_a2, "cold2": cold_a2,
        "avg_cold": avg_cold,
        "stress_ok": ok_count, "stress_total": len(prompts),
        "match": match_a and match_a2,
        "reduction": reduction_a,
        "verdict": "PASS" if all_pass else "FAIL",
    }), flush=True)

except Exception as e:
    traceback.print_exc()
    print(f"RESULT:" + json.dumps({"error": str(e)}), flush=True)
    try:
        server_proc.terminate()
    except:
        pass
'''


@app.function(gpu="H100:2", image=image, timeout=900)
def test_cudagraphs_tp2():
    """Test checkpoint/restore with CUDA graphs + TP=2."""
    import subprocess, sys, json, tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(SERVE_SCRIPT)
        script_path = f.name

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, timeout=800,
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
    print("Phase 42 v25: CUDA graphs + TP=2 (production default)")
    print("=" * 60)

    r = test_cudagraphs_tp2.remote()

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    if 'error' in r:
        print(f"  FAILED: {r['error'][:80]}")
    else:
        print(f"  Config: {r['config']}")
        print(f"  Startup: {r['startup_time']:.1f}s")
        print(f"  CUDA PIDs: {r['num_cuda_pids']}")
        print(f"  Cycle 1: ckpt={r['ckpt1']:.2f}s, restore={r['rest1']:.2f}s, cold={r['cold1']:.2f}s")
        print(f"  Cycle 2: ckpt={r['ckpt2']:.2f}s, restore={r['rest2']:.2f}s, cold={r['cold2']:.2f}s")
        print(f"  Avg cold: {r['avg_cold']:.2f}s")
        print(f"  Stress: {r['stress_ok']}/{r['stress_total']} OK")
        print(f"  Reduction: {r['reduction']:.1f}%")
        print(f"  Verdict: {r['verdict']}")
