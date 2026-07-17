# GPU Reservation Guard Design

## Goal

Provide an opt-in local utility under `/home/yxz/CoT/gpu_script` that reserves any four currently available A800 GPUs with a bounded PyTorch stress process, releases only its own processes for a training command, and automatically re-acquires four GPUs when that command exits.

## Scope and safety boundary

- The utility only selects GPUs that have no reported compute processes, satisfy a minimum free-memory threshold, and are revalidated immediately before allocation.
- It never kills arbitrary PIDs, uses `killall`, or changes another process's CUDA state.
- It owns all child workers through a state file and process-group cleanup. Cleanup is idempotent and runs on normal exit, Ctrl-C, and shell termination traps where possible.
- It uses a conservative memory target of 70,000 MiB per GPU, a 1,500 MiB selection margin, and a utilization target of 85% rather than attempting to force 100%.
- If four GPUs are not simultaneously safe, it waits and does not partially reserve a set.

## User interface

```bash
cd /home/yxz/CoT/gpu_script
./gpu_guard.sh hold
./gpu_guard.sh status
./gpu_guard.sh release
./gpu_guard.sh run -- accelerate launch ...
```

`hold` blocks while waiting for a set of four GPUs and keeps the workers alive until interrupted. `release` stops only workers recorded by this utility. `run -- COMMAND ...` releases an existing hold, runs the command, and reacquires a reservation after the command exits or is interrupted. Configuration is available through environment variables, including `GPU_COUNT`, `TARGET_MEMORY_MIB`, `MIN_FREE_MEMORY_MIB`, `MIN_UTILIZATION_PERCENT`, `POLL_SECONDS`, and `PYTHON_BIN`.

## Architecture

`gpu_guard.sh` is the lifecycle and selection layer. It queries `nvidia-smi` for all GPU indices, free memory, and compute PIDs; filters candidates; atomically records the candidate list; then starts one worker per selected GPU. It verifies that each worker remains alive and reports the selected `CUDA_VISIBLE_DEVICES` set. The shell wrapper owns a lock directory and state file to prevent two reservations from racing.

`gpu_worker.py` is a single-GPU process. It allocates a bounded byte buffer in chunks, leaving a configured safety margin, then repeatedly performs FP16 matrix multiplications. A watchdog-like heartbeat is written to the worker's state only through process liveness; no external process is touched. OOM during allocation causes that worker to exit cleanly so the shell layer can release the partial reservation and retry.

## Lifecycle

```text
hold/run -> poll candidates -> revalidate -> start 4 workers
       ^                                  |
       |                                  v
       +----------- command exit <---- release own workers
```

For `run`, workers are stopped before the user's command starts. A shell `trap` always attempts cleanup and, for `run`, starts a new `hold` after the command's exit status has been captured. The wrapper does not automatically take GPUs that become occupied by another process.

## Failure handling

- Missing `nvidia-smi`, missing Python, or missing PyTorch produces an actionable error before a reservation is claimed.
- A partial worker-start failure stops all workers from the current reservation and returns to polling.
- Stale state files are ignored after verifying recorded PIDs are no longer alive.
- A command exit status is preserved after the post-command reacquisition has been started.
- The worker target is bounded and configurable; allocation is based on available memory minus the safety margin, never on total memory alone.

## Validation

- Shell syntax check and Python bytecode compilation.
- Dry-run GPU discovery against the current eight A800s without allocating memory.
- One short hold/release smoke test with a reduced memory target in a test-only environment.
- Manual `run -- true` lifecycle test to verify release, command execution, and re-acquisition behavior.
- Verify that status reports only utility-owned PIDs and that a second invocation refuses to start.
