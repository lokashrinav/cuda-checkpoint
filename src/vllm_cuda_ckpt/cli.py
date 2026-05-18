"""CLI entrypoint for vllm-ckpt command.

Usage:
    vllm-ckpt discover                          # auto-find vllm serve
    vllm-ckpt discover --pid <PID>              # explicit PID
    vllm-ckpt cycle --port 8000 --model <MODEL> # auto-find + test
    vllm-ckpt benchmark --port 8000 --model <MODEL> --cycles 3
    vllm-ckpt watch --port 8000 --interval 60   # sidecar daemon
    vllm-ckpt recommend                         # GPU-specific settings
"""

import argparse
import json
import sys
import time

from vllm_cuda_ckpt.api import CudaCheckpointAPI
from vllm_cuda_ckpt.discover import discover_cuda_pids, find_vllm_server
from concurrent.futures import ThreadPoolExecutor


def _resolve_pid(args) -> int:
    if args.pid:
        return args.pid
    try:
        pid = find_vllm_server()
        print(f"Auto-discovered vllm serve PID: {pid}")
        return pid
    except RuntimeError as e:
        print(f"ERROR: {e}. Use --pid to specify manually.", file=sys.stderr)
        sys.exit(1)


def _check_health(port: int, timeout: float = 10.0) -> bool:
    try:
        import httpx
        r = httpx.get(f"http://localhost:{port}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _query_server(port: int, prompt: str, model: str, max_tokens: int = 32) -> str:
    import httpx
    r = httpx.post(
        f"http://localhost:{port}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()


def _do_checkpoint(api: CudaCheckpointAPI, pids: list[int], parallel: bool) -> float:
    for pid in pids:
        api.lock(pid)
    t0 = time.perf_counter()
    if parallel and len(pids) > 1:
        with ThreadPoolExecutor(max_workers=len(pids)) as ex:
            list(ex.map(api.checkpoint, pids))
    else:
        for pid in pids:
            api.checkpoint(pid)
    return time.perf_counter() - t0


def _do_restore(api: CudaCheckpointAPI, pids: list[int], parallel: bool) -> float:
    t0 = time.perf_counter()
    if parallel and len(pids) > 1:
        with ThreadPoolExecutor(max_workers=len(pids)) as ex:
            list(ex.map(api.restore, pids))
    else:
        for pid in pids:
            api.restore(pid)
    rest_time = time.perf_counter() - t0
    for pid in pids:
        api.unlock(pid)
    return rest_time


def cmd_discover(args):
    pids = discover_cuda_pids(_resolve_pid(args))
    print(f"CUDA PIDs: {pids} ({len(pids)} total)")
    if args.json:
        print(json.dumps({"pids": pids}))


def cmd_checkpoint(args):
    api = CudaCheckpointAPI()
    pids = discover_cuda_pids(_resolve_pid(args))
    print(f"CUDA PIDs: {pids} ({len(pids)} total)")
    if not pids:
        print("ERROR: No CUDA-active PIDs found", file=sys.stderr)
        sys.exit(1)
    ckpt_time = _do_checkpoint(api, pids, parallel=not args.sequential)
    print(f"Checkpoint: {ckpt_time:.2f}s ({len(pids)} PIDs)")
    if args.json:
        print(json.dumps({"action": "checkpoint", "pids": pids, "time": ckpt_time}))


def cmd_restore(args):
    api = CudaCheckpointAPI()
    pids = discover_cuda_pids(_resolve_pid(args))
    print(f"CUDA PIDs: {pids} ({len(pids)} total)")
    if not pids:
        print("ERROR: No CUDA-active PIDs found", file=sys.stderr)
        sys.exit(1)
    rest_time = _do_restore(api, pids, parallel=not args.sequential)
    result = {"action": "restore", "pids": pids, "time": rest_time}
    print(f"Restore: {rest_time:.2f}s ({len(pids)} PIDs)")
    if args.port:
        healthy = _check_health(args.port, timeout=30)
        result["healthy"] = healthy
        print(f"Health: {'OK' if healthy else 'FAIL'}")
    if args.json:
        print(json.dumps(result))


def cmd_cycle(args):
    api = CudaCheckpointAPI()
    pids = discover_cuda_pids(_resolve_pid(args))
    print(f"CUDA PIDs: {pids} ({len(pids)} total)")
    if not pids:
        print("ERROR: No CUDA-active PIDs found", file=sys.stderr)
        sys.exit(1)
    parallel = not args.sequential
    ckpt_time = _do_checkpoint(api, pids, parallel=parallel)
    print(f"Checkpoint: {ckpt_time:.2f}s")
    rest_time = _do_restore(api, pids, parallel=parallel)
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
    if args.json:
        print(json.dumps(result))


def cmd_watch(args):
    import signal

    api = CudaCheckpointAPI()
    interval = args.interval
    port = args.port
    model = args.model

    print(f"Watch mode: checkpoint every {interval}s", flush=True)

    # Wait for server to be ready
    if port:
        print(f"Waiting for server on port {port}...", flush=True)
        for _ in range(300):
            if _check_health(port, timeout=2):
                break
            time.sleep(1)
        else:
            print("ERROR: Server never became healthy", file=sys.stderr)
            sys.exit(1)
        print("Server healthy", flush=True)

    pid = _resolve_pid(args)
    pids = discover_cuda_pids(pid)
    print(f"CUDA PIDs: {pids} ({len(pids)} total)", flush=True)
    if not pids:
        print("ERROR: No CUDA-active PIDs found", file=sys.stderr)
        sys.exit(1)

    parallel = not args.sequential
    stop_event = False
    last_ckpt = None

    def handle_sigterm(signum, frame):
        nonlocal stop_event
        print("\nSIGTERM received — final checkpoint...", flush=True)
        try:
            _do_checkpoint(api, pids, parallel)
            print("Final checkpoint OK", flush=True)
        except Exception as e:
            print(f"Final checkpoint failed: {e}", file=sys.stderr)
        stop_event = True

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    cycle_count = 0
    while not stop_event:
        try:
            ckpt_time = _do_checkpoint(api, pids, parallel)
            rest_time = _do_restore(api, pids, parallel)
            cycle_count += 1
            last_ckpt = time.time()

            status = {"cycle": cycle_count, "checkpoint": round(ckpt_time, 3),
                      "restore": round(rest_time, 3), "timestamp": last_ckpt}

            if port:
                healthy = _check_health(port, timeout=10)
                status["healthy"] = healthy
                if not healthy:
                    print(f"WARNING: Server unhealthy after restore (cycle {cycle_count})", file=sys.stderr)

            if args.json:
                print(json.dumps(status), flush=True)
            else:
                health_str = f" health={'OK' if status.get('healthy', True) else 'FAIL'}" if port else ""
                print(f"Cycle {cycle_count}: ckpt={ckpt_time:.2f}s rest={rest_time:.2f}s{health_str}", flush=True)

        except Exception as e:
            print(f"ERROR in cycle {cycle_count + 1}: {e}", file=sys.stderr)

        # Sleep in small increments so SIGTERM is responsive
        for _ in range(int(interval)):
            if stop_event:
                break
            time.sleep(1)

    print(f"Watch stopped after {cycle_count} cycles", flush=True)


def cmd_recommend(args):
    import subprocess as sp

    r = sp.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode != 0:
        print("ERROR: nvidia-smi not found or no GPUs detected", file=sys.stderr)
        sys.exit(1)

    gpus = []
    for line in r.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(", ")]
        gpus.append({"name": parts[0], "memory_mib": int(parts[1]), "compute_cap": parts[2]})

    num_gpus = len(gpus)
    gpu = gpus[0]
    sm = gpu["compute_cap"]
    mem_gib = gpu["memory_mib"] / 1024

    use_graphs = sm.startswith("9.")
    tp = min(num_gpus, 2) if num_gpus > 1 else 1

    if mem_gib >= 70:
        util = "0.80"
        expected_restore = "3-5s"
    elif mem_gib >= 35:
        util = "0.60"
        expected_restore = "3-5s"
    elif mem_gib >= 20:
        util = "0.80"
        expected_restore = "2-4s"
    else:
        util = "0.85"
        expected_restore = "2-3s"

    print(f"GPU: {gpu['name']} (sm_{sm}, {mem_gib:.0f} GiB) x{num_gpus}")
    print()
    print("Recommended vllm serve flags:")
    flags = [f"--tensor-parallel-size {tp}"]
    if not use_graphs:
        flags.append("--enforce-eager")
    flags.append(f"--gpu-memory-utilization {util}")
    if tp > 1:
        flags.append("--disable-custom-all-reduce")
    print(f"  vllm serve <MODEL> {' '.join(flags)}")
    print()
    print("Required environment:")
    print("  CUDA_MODULE_LOADING=EAGER")
    if tp > 1:
        print("  NCCL_NVLS_ENABLE=0")
        print("  NCCL_P2P_DISABLE=1")
    print("  VLLM_USE_V1=1")
    print()
    print(f"CUDA graphs: {'enabled (H100 benefits from graphs)' if use_graphs else 'disabled (enforce-eager recommended for this GPU)'}")
    print(f"Expected restore: {expected_restore}")
    print(f"Expected cold start reduction: 92-98%")

    if args.json:
        result = {
            "gpu": gpu["name"], "compute_cap": sm, "memory_gib": round(mem_gib, 1),
            "num_gpus": num_gpus, "tp": tp, "cuda_graphs": use_graphs,
            "gpu_memory_utilization": float(util), "expected_restore": expected_restore,
            "flags": flags,
        }
        print(json.dumps(result))


def cmd_benchmark(args):
    api = CudaCheckpointAPI()
    pids = discover_cuda_pids(_resolve_pid(args))
    print(f"CUDA PIDs: {pids} ({len(pids)} total)")
    if not pids:
        print("ERROR: No CUDA-active PIDs found", file=sys.stderr)
        sys.exit(1)
    parallel = not args.sequential
    cycles = []
    for i in range(args.cycles):
        print(f"\n--- Cycle {i+1}/{args.cycles} ---")
        ckpt_time = _do_checkpoint(api, pids, parallel=parallel)
        rest_time = _do_restore(api, pids, parallel=parallel)
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
        avg_cold = sum(c["cold_start"] for c in cycles) / len(cycles)
        print(f"Avg cold start: {avg_cold:.2f}s")
    if args.json:
        print(json.dumps({"action": "benchmark", "pids": pids, "cycles": cycles}))


def main():
    parser = argparse.ArgumentParser(
        prog="vllm-ckpt",
        description="Checkpoint/restore running vLLM serve processes via NVIDIA cuda-checkpoint",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("recommend")
    rec.add_argument("--json", action="store_true", help="JSON output")
    rec.set_defaults(func=cmd_recommend)

    for name, fn in [
        ("discover", cmd_discover),
        ("checkpoint", cmd_checkpoint),
        ("restore", cmd_restore),
        ("cycle", cmd_cycle),
        ("benchmark", cmd_benchmark),
        ("watch", cmd_watch),
    ]:
        p = sub.add_parser(name)
        p.add_argument("--pid", type=int, help="vllm serve root PID (auto-discovered if omitted)")
        p.add_argument("--json", action="store_true", help="JSON output")
        p.set_defaults(func=fn)

        if name not in ("discover",):
            p.add_argument("--sequential", action="store_true",
                          help="Sequential PID processing (default: parallel)")

        if name in ("restore", "cycle", "benchmark", "watch"):
            p.add_argument("--port", type=int, help="vllm serve port for health check")

        if name in ("cycle", "benchmark", "watch"):
            p.add_argument("--model", type=str, help="Model name for inference test")

        if name == "benchmark":
            p.add_argument("--cycles", type=int, default=3, help="Number of cycles")

        if name == "watch":
            p.add_argument("--interval", type=int, default=60,
                          help="Seconds between checkpoint cycles (default: 60)")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
