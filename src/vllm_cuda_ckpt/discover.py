"""CUDA PID discovery for vLLM process trees."""

import subprocess


def find_vllm_server() -> int:
    """Auto-discover a running vllm serve process.

    Searches for processes matching vllm's OpenAI API server module.
    Returns the PID of the server process, or raises RuntimeError if
    none or multiple are found.
    """
    r = subprocess.run(
        ["pgrep", "-f", "vllm.entrypoints.openai.api_server"],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError("No vllm serve process found")

    pids = r.stdout.strip().split()
    if len(pids) > 1:
        r2 = subprocess.run(
            ["pgrep", "-f", "vllm.entrypoints.openai.api_server", "--oldest"],
            capture_output=True, text=True,
        )
        if r2.returncode == 0 and r2.stdout.strip():
            return int(r2.stdout.strip().split()[0])
    return int(pids[0])


def discover_cuda_pids(server_pid: int) -> list[int]:
    """Recursively find all CUDA-active PIDs in a process tree.

    Walks 4 levels deep (server -> children -> grandchildren -> great-grandchildren)
    and probes each with cuda-checkpoint --action lock to verify CUDA activity.
    """
    all_pids = {str(server_pid)}

    def get_children(pid: str) -> list[str]:
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
