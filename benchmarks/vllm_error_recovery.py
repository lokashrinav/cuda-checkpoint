"""Phase 42 v29: Error recovery and partial failure handling.

Production validation: What happens when checkpoint/restore operations
fail partway through? Tests graceful degradation and recovery.

Tests:
  A. Normal cycle baseline (sanity)
  B. Checkpoint with invalid PID (should not affect valid PIDs)
  C. Double-checkpoint (checkpoint already-checkpointed process)
  D. Double-restore (restore already-running process)
  E. Checkpoint → restore with health-gated retry
  F. Recovery after all error scenarios (server still works)

Usage:
    modal run cuda_serializer/modal_multigpu_v29.py
"""

import modal, os
app = modal.App("vllm-ckpt-recovery-v29")

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

    def safe_lock(self, pid):
        try:
            self.lock(pid)
            return True
        except RuntimeError:
            return False

    def safe_checkpoint(self, pid):
        try:
            self.checkpoint(pid)
            return True
        except RuntimeError:
            return False

    def safe_restore(self, pid):
        try:
            self.restore(pid)
            return True
        except RuntimeError:
            return False

    def safe_unlock(self, pid):
        try:
            self.unlock(pid)
            return True
        except RuntimeError:
            return False

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
        return pid
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

def _check_health(port, timeout=10.0):
    try:
        import httpx
        r = httpx.get(f"http://localhost:{port}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False

def cmd_discover(args):
    pids = discover_cuda_pids(_resolve_pid(args))
    print(f"CUDA PIDs: {pids} ({len(pids)} total)")
    if args.json: print(json.dumps({"pids": pids}))

def main():
    parser = argparse.ArgumentParser(prog="vllm-ckpt")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("discover")
    p.add_argument("--pid", type=int)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_discover)
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

SERVE_SCRIPT = r'''
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


def health_check_with_retry(max_retries=5, delay=2.0):
    for i in range(max_retries):
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=10)
            if r.status_code == 200:
                return True, i + 1
        except Exception:
            pass
        if i < max_retries - 1:
            time.sleep(delay)
    return False, max_retries


try:
    from vllm_cuda_ckpt import CudaCheckpointAPI, discover_cuda_pids, find_vllm_server
    from concurrent.futures import ThreadPoolExecutor

    print("=" * 60, flush=True)
    print("  PHASE 42 v29: ERROR RECOVERY + PARTIAL FAILURE", flush=True)
    print("=" * 60, flush=True)

    # Start server
    print("\n--- Starting vllm serve (TP=2, enforce-eager) ---", flush=True)
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

    ref_text = query_server("The capital of France is")
    print(f"  Baseline: {ref_text[:80]}", flush=True)

    pids = discover_cuda_pids(find_vllm_server())
    print(f"  CUDA PIDs: {pids} ({len(pids)})", flush=True)

    api = CudaCheckpointAPI()
    tests = {}

    # ============================================================
    # TEST A: Normal cycle baseline
    # ============================================================
    print("\n--- Test A: Normal cycle (baseline) ---", flush=True)
    for pid in pids:
        api.lock(pid)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(pids)) as ex:
        list(ex.map(api.checkpoint, pids))
    ckpt_a = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(pids)) as ex:
        list(ex.map(api.restore, pids))
    rest_a = time.perf_counter() - t0
    for pid in pids:
        api.unlock(pid)

    text_a = query_server("The capital of France is")
    match_a = text_a.strip() == ref_text.strip()
    tests["A_baseline"] = {"ckpt": round(ckpt_a, 3), "rest": round(rest_a, 3), "match": match_a, "pass": match_a}
    print(f"  ckpt={ckpt_a:.2f}s rest={rest_a:.2f}s match={match_a} [{'PASS' if match_a else 'FAIL'}]", flush=True)

    # ============================================================
    # TEST B: Checkpoint with invalid PID mixed in
    # ============================================================
    print("\n--- Test B: Invalid PID handling ---", flush=True)
    invalid_pid = 99999
    b_errors = []

    # Lock valid PIDs
    for pid in pids:
        api.lock(pid)

    # Try to lock invalid PID
    try:
        api.lock(invalid_pid)
        b_errors.append("lock_invalid_should_fail")
    except RuntimeError as e:
        print(f"  Lock invalid PID {invalid_pid}: correctly raised {e}", flush=True)

    # Checkpoint valid PIDs should still work
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(pids)) as ex:
        list(ex.map(api.checkpoint, pids))
    ckpt_b = time.perf_counter() - t0

    # Try checkpoint on invalid PID
    try:
        api.checkpoint(invalid_pid)
        b_errors.append("ckpt_invalid_should_fail")
    except RuntimeError as e:
        print(f"  Checkpoint invalid PID: correctly raised {e}", flush=True)

    # Restore valid PIDs
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(pids)) as ex:
        list(ex.map(api.restore, pids))
    rest_b = time.perf_counter() - t0
    for pid in pids:
        api.unlock(pid)

    text_b = query_server("The capital of France is")
    match_b = text_b.strip() == ref_text.strip()
    b_pass = match_b and len(b_errors) == 0
    tests["B_invalid_pid"] = {"ckpt": round(ckpt_b, 3), "rest": round(rest_b, 3), "match": match_b,
                               "errors": b_errors, "pass": b_pass}
    print(f"  Valid PIDs ckpt={ckpt_b:.2f}s rest={rest_b:.2f}s match={match_b} [{'PASS' if b_pass else 'FAIL'}]", flush=True)

    # ============================================================
    # TEST C: Double checkpoint (checkpoint already-checkpointed process)
    # ============================================================
    print("\n--- Test C: Double checkpoint ---", flush=True)
    for pid in pids:
        api.lock(pid)
    with ThreadPoolExecutor(max_workers=len(pids)) as ex:
        list(ex.map(api.checkpoint, pids))
    print(f"  First checkpoint OK", flush=True)

    # Try to checkpoint again while already checkpointed
    double_ckpt_errors = []
    for pid in pids:
        ok = api.safe_checkpoint(pid)
        if not ok:
            double_ckpt_errors.append(pid)
    if double_ckpt_errors:
        print(f"  Double checkpoint correctly failed for PIDs: {double_ckpt_errors}", flush=True)
    else:
        print(f"  Double checkpoint succeeded (driver handles it)", flush=True)

    # Restore
    with ThreadPoolExecutor(max_workers=len(pids)) as ex:
        list(ex.map(api.restore, pids))
    for pid in pids:
        api.unlock(pid)

    text_c = query_server("The capital of France is")
    match_c = text_c.strip() == ref_text.strip()
    tests["C_double_ckpt"] = {"double_ckpt_failed_pids": double_ckpt_errors, "match": match_c, "pass": match_c}
    print(f"  Recovery: match={match_c} [{'PASS' if match_c else 'FAIL'}]", flush=True)

    # ============================================================
    # TEST D: Double restore (restore already-running process)
    # ============================================================
    print("\n--- Test D: Double restore ---", flush=True)
    # First do a normal cycle
    for pid in pids:
        api.lock(pid)
    with ThreadPoolExecutor(max_workers=len(pids)) as ex:
        list(ex.map(api.checkpoint, pids))
    with ThreadPoolExecutor(max_workers=len(pids)) as ex:
        list(ex.map(api.restore, pids))
    for pid in pids:
        api.unlock(pid)
    print(f"  Normal cycle OK", flush=True)

    # Try to restore again (process is already running, not checkpointed)
    double_rest_errors = []
    for pid in pids:
        ok = api.safe_restore(pid)
        if not ok:
            double_rest_errors.append(pid)
    if double_rest_errors:
        print(f"  Double restore correctly failed for PIDs: {double_rest_errors}", flush=True)
    else:
        print(f"  Double restore succeeded (driver handles it)", flush=True)

    text_d = query_server("The capital of France is")
    match_d = text_d.strip() == ref_text.strip()
    tests["D_double_restore"] = {"double_rest_failed_pids": double_rest_errors, "match": match_d, "pass": match_d}
    print(f"  Recovery: match={match_d} [{'PASS' if match_d else 'FAIL'}]", flush=True)

    # ============================================================
    # TEST E: Health-gated restore with retry
    # ============================================================
    print("\n--- Test E: Health-gated restore with retry ---", flush=True)
    for pid in pids:
        api.lock(pid)
    with ThreadPoolExecutor(max_workers=len(pids)) as ex:
        list(ex.map(api.checkpoint, pids))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(pids)) as ex:
        list(ex.map(api.restore, pids))
    rest_e = time.perf_counter() - t0
    for pid in pids:
        api.unlock(pid)

    healthy, attempts = health_check_with_retry(max_retries=5, delay=2.0)
    t0 = time.perf_counter()
    text_e = query_server("The capital of France is")
    infer_e = time.perf_counter() - t0
    match_e = text_e.strip() == ref_text.strip()
    tests["E_health_gated"] = {"rest": round(rest_e, 3), "healthy": healthy, "attempts": attempts,
                                "infer": round(infer_e, 3), "match": match_e, "pass": healthy and match_e}
    print(f"  rest={rest_e:.2f}s healthy={healthy} (attempt {attempts}) infer={infer_e:.3f}s match={match_e} [{'PASS' if healthy and match_e else 'FAIL'}]", flush=True)

    # ============================================================
    # TEST F: Final recovery check — server still works after all error scenarios
    # ============================================================
    print("\n--- Test F: Final recovery (server stable after all tests) ---", flush=True)

    # Run 3 rapid cycles
    for i in range(3):
        for pid in pids:
            api.lock(pid)
        with ThreadPoolExecutor(max_workers=len(pids)) as ex:
            list(ex.map(api.checkpoint, pids))
        with ThreadPoolExecutor(max_workers=len(pids)) as ex:
            list(ex.map(api.restore, pids))
        for pid in pids:
            api.unlock(pid)

    text_f = query_server("The capital of France is")
    match_f = text_f.strip() == ref_text.strip()

    # Concurrent stress
    from concurrent.futures import as_completed
    stress_prompts = ["Python was created by", "Water boils at", "E = mc", "Linux was created by"]
    stress_ok = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(query_server, p): p for p in stress_prompts}
        for f in as_completed(futures):
            try:
                f.result()
                stress_ok += 1
            except Exception:
                pass

    tests["F_final_recovery"] = {"match": match_f, "stress_ok": stress_ok,
                                  "stress_total": len(stress_prompts),
                                  "pass": match_f and stress_ok == len(stress_prompts)}
    print(f"  3 rapid cycles OK, match={match_f}, stress={stress_ok}/{len(stress_prompts)} [{'PASS' if tests['F_final_recovery']['pass'] else 'FAIL'}]", flush=True)

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'=' * 60}", flush=True)
    print("  RESULTS SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)

    all_pass = all(t["pass"] for t in tests.values())
    for name, t in tests.items():
        status = "PASS" if t["pass"] else "FAIL"
        print(f"  {name}: {status}", flush=True)
    print(f"  VERDICT: {'PASS' if all_pass else 'FAIL'}", flush=True)

    result = {
        "startup_time": startup_time,
        "num_pids": len(pids),
        "tests": tests,
        "all_pass": all_pass,
        "verdict": "PASS" if all_pass else "FAIL",
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


@app.function(gpu="H100:2", image=image, timeout=600)
def test_recovery_v29():
    """Error recovery and partial failure test."""
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
        capture_output=True, text=True, timeout=500,
    )
    out = result.stdout[-12000:] if len(result.stdout) > 12000 else result.stdout
    print(out, flush=True)

    for line in result.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:"):])

    return {"error": f"No result (exit={result.returncode})", "stderr": result.stderr[-500:]}


@app.local_entrypoint()
def main():
    print("Phase 42 v29: Error recovery + partial failure")
    print("=" * 60)

    r = test_recovery_v29.remote()

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    if 'error' in r:
        print(f"  FAILED: {r['error'][:200]}")
        if 'stderr' in r:
            print(f"  stderr: {r['stderr'][:200]}")
    else:
        print(f"  Startup: {r['startup_time']:.1f}s")
        print(f"  CUDA PIDs: {r['num_pids']}")
        for name, t in r['tests'].items():
            status = "PASS" if t["pass"] else "FAIL"
            print(f"  {name}: {status}")
        print(f"  Verdict: {r['verdict']}")
