# GPU Reservation Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a safe local GPU reservation wrapper that uses any four available A800 GPUs, targets 70,000 MiB and at least 85% utilization, and restores the reservation after a wrapped command exits.

**Architecture:** A Bash controller owns selection, locking, state, lifecycle, and the `hold/release/run/status` interface. A separate Python/PyTorch worker is launched once per selected GPU and performs bounded memory allocation plus FP16 GEMM. The controller rechecks candidates before starting workers and cleans up only recorded child PIDs.

**Tech Stack:** Bash, `nvidia-smi`, Python 3, PyTorch from the `CoT` environment.

## Global Constraints

- Reserve exactly 4 GPUs, selected from any indices visible to `nvidia-smi`.
- Target `70,000 MiB` per GPU and require `1,500 MiB` free headroom before selection.
- Target utilization is `85%`; 100% is not required.
- Never kill arbitrary processes; only terminate PIDs recorded by this utility.
- Do not partially reserve a set; on startup failure clean up all workers and retry.
- Default Python command is `python` when the caller has activated `CoT`; `PYTHON_BIN` can point to the CoT interpreter explicitly.

---

### Task 1: Add the worker contract and tests

**Files:**
- Create: `/home/yxz/CoT/gpu_script/gpu_worker.py`
- Create: `/home/yxz/CoT/gpu_script/tests/test_gpu_worker.py`

**Interfaces:**
- CLI: `gpu_worker.py --gpu-index INDEX --target-memory-mib N --safety-mib N --heartbeat PATH --matrix-size N`.
- Exit code 0 on SIGTERM/SIGINT cleanup; nonzero on CUDA allocation failure or invalid arguments.
- No global process termination and no writes outside the supplied heartbeat path.

- [ ] **Step 1: Write tests for pure worker helpers.** Test that the allocation budget is `max(0, target - safety)`, invalid nonpositive sizes are rejected, and the heartbeat parent directory is created.
- [ ] **Step 2: Run the worker unit tests.** Run `PYTHONPATH=/home/yxz/CoT/gpu_script python -m pytest /home/yxz/CoT/gpu_script/tests/test_gpu_worker.py -q`. Expected: the tests initially fail because the module does not exist.
- [ ] **Step 3: Implement the worker.** Add typed argument parsing, `compute_allocation_bytes`, chunked `torch.empty(..., dtype=torch.uint8)` allocation up to the bounded budget, and a loop using two FP16 square matrices and `torch.matmul`. Use `torch.cuda.set_device(args.gpu_index)` and `torch.cuda.synchronize()` after each iteration. Install SIGTERM/SIGINT handlers that set a stop event and release tensors. Write a heartbeat timestamp only once after initialization; the controller will use PID liveness for cleanup.
- [ ] **Step 4: Run tests and compile.** Run `PYTHONPATH=/home/yxz/CoT/gpu_script python -m pytest /home/yxz/CoT/gpu_script/tests/test_gpu_worker.py -q` and `python -m py_compile /home/yxz/CoT/gpu_script/gpu_worker.py`. Expected: unit tests pass and compilation exits 0 when run inside `CoT`.

### Task 2: Add GPU discovery and lifecycle controller

**Files:**
- Create: `/home/yxz/CoT/gpu_script/gpu_guard.sh`
- Create: `/home/yxz/CoT/gpu_script/README.md`

**Interfaces:**
- `./gpu_guard.sh hold`: wait until exactly four safe candidates exist, start workers, and block.
- `./gpu_guard.sh release`: terminate only workers in the state file.
- `./gpu_guard.sh status`: show state, selected indices, and owned PIDs.
- `./gpu_guard.sh run -- COMMAND [ARGS...]`: release, run the command, reacquire in the background/foreground hold path after exit, and return the command's original status.

- [ ] **Step 1: Add a shell-level smoke-test checklist to README.** Document `conda activate CoT`, `PYTHON_BIN`, conservative defaults, `GPU_COUNT=4`, `TARGET_MEMORY_MIB=70000`, `MIN_FREE_MEMORY_MIB=71500`, `MIN_UTILIZATION_PERCENT=85`, and the fact that only idle GPUs are selected.
- [ ] **Step 2: Implement lock and state helpers.** Use `mkdir "$STATE_DIR/lock"` as an atomic lock, store selected indices and worker PIDs in a state file with mode 600, and use `kill -0` plus `/proc/$pid/cmdline` checks before cleanup. Treat missing or stale state as released.
- [ ] **Step 3: Implement discovery.** Query `nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits`, query compute PIDs per GPU, reject GPUs with any compute PID, require `memory.free >= MIN_FREE_MEMORY_MIB`, and sort by index. Select the first four candidates and re-query all four immediately before launch.
- [ ] **Step 4: Implement worker start and cleanup.** Start one worker per selected GPU with `CUDA_VISIBLE_DEVICES` set to that physical index, redirect each worker's log to `logs/gpu-INDEX.log`, record PIDs, wait briefly, and abort/release if any child exits. Cleanup sends TERM only to recorded matching workers, waits up to 10 seconds, then sends KILL only to still-matching recorded workers.
- [ ] **Step 5: Implement commands and traps.** `hold` loops with `POLL_SECONDS=5`; `release` is idempotent; `status` prints state and process liveness; `run` releases, runs the exact user command, captures `$?`, and invokes the hold loop again after exit while preserving the original status. Ctrl-C must stop the wrapped command and still execute cleanup.
- [ ] **Step 6: Run shell checks.** Run `bash -n /home/yxz/CoT/gpu_script/gpu_guard.sh` and `./gpu_guard.sh status`. Expected: syntax passes and status reports `released` or a valid owned reservation without changing unrelated GPU processes.

### Task 3: Verify against the live eight-GPU host

**Files:**
- Modify: `/home/yxz/CoT/gpu_script/README.md` only if observed behavior requires a documented adjustment.

- [ ] **Step 1: Run dry discovery.** Use `DRY_RUN=1 ./gpu_guard.sh status` or the controller's discovery mode to print candidates without allocating memory. Expected: no worker starts and no other process is terminated.
- [ ] **Step 2: Run a reduced smoke reservation.** Set `TARGET_MEMORY_MIB=1000 MIN_FREE_MEMORY_MIB=3000` and run `./gpu_guard.sh hold`, then interrupt it. Expected: four workers start only if four cards pass the idle checks, and `release` removes only those workers.
- [ ] **Step 3: Run lifecycle smoke test.** Run `TARGET_MEMORY_MIB=1000 MIN_FREE_MEMORY_MIB=3000 ./gpu_guard.sh run -- bash -c 'sleep 2'`. Expected: the reservation is released before `sleep`, the command exits 0, and the controller enters the post-command reservation loop.
- [ ] **Step 4: Validate default configuration without starting it.** Run `./gpu_guard.sh print-config`. Expected: it prints exactly four GPUs, 70,000 MiB target, 1,500 MiB margin, and 85% utilization target.
- [ ] **Step 5: Hand off usage.** Start with `conda activate CoT`, then use `./gpu_guard.sh run -- <training command>`. If a four-card set is unavailable, the controller waits instead of overloading or partially claiming cards.
