"""CUDA PID discovery for arbitrary process trees.

Framework-agnostic — walks any process tree and probes each PID
for CUDA activity via cuda-checkpoint --action lock.
"""

import subprocess


def find_cuda_pids_for_process(root_pid: int, depth: int = 4) -> list[int]:
    """Recursively find all CUDA-active PIDs in a process tree.

    Walks `depth` levels deep from root_pid and probes each PID with
    cuda-checkpoint --action lock to verify CUDA activity.

    Args:
        root_pid: The root process PID to start from.
        depth: How many levels of children to walk (default 4).

    Returns:
        Sorted list of CUDA-active PIDs.
    """
    all_pids = {str(root_pid)}

    def get_children(pid: str) -> list[str]:
        r = subprocess.run(["pgrep", "-P", pid], capture_output=True, text=True)
        return r.stdout.strip().split() if r.stdout.strip() else []

    frontier = [str(root_pid)]
    for _ in range(depth):
        next_frontier = []
        for pid in frontier:
            children = get_children(pid)
            for c in children:
                all_pids.add(c)
                next_frontier.append(c)
        frontier = next_frontier

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


def discover_cuda_pids(root_pid: int) -> list[int]:
    """Alias for find_cuda_pids_for_process with default depth."""
    return find_cuda_pids_for_process(root_pid)


def find_process_by_name(pattern: str) -> int:
    """Find a process by command-line pattern via pgrep.

    Args:
        pattern: Regex pattern to match against process command lines.

    Returns:
        PID of the matching process.

    Raises:
        RuntimeError: If no process or multiple ambiguous processes found.
    """
    r = subprocess.run(
        ["pgrep", "-f", pattern],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"No process found matching '{pattern}'")

    pids = r.stdout.strip().split()
    if len(pids) > 1:
        r2 = subprocess.run(
            ["pgrep", "-f", pattern, "--oldest"],
            capture_output=True, text=True,
        )
        if r2.returncode == 0 and r2.stdout.strip():
            return int(r2.stdout.strip().split()[0])
    return int(pids[0])
