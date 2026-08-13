# LIBERO Post-Training Parallel Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reliably start standard LIBERO and LIBERO-plus evaluation in parallel after the active V3 training run successfully exports its final checkpoint.

**Architecture:** Add one small Bash orchestrator that waits for the known training PID, validates the 60k and final artifacts, then launches the two existing benchmark launchers as independent concurrent children. Add subprocess tests with fake training/evaluation processes so failure gating and concurrency are verified without GPUs or simulators.

**Tech Stack:** Bash, tmux, Python `unittest`, existing LIBERO evaluation launchers.

## Global Constraints

- Standard LIBERO uses GPUs `0,1,2,3`, 50 trials per task, and no videos.
- LIBERO-plus uses GPUs `0,1,2,3,4,5,6,7`, one trial per task, and no videos.
- Both evaluations start concurrently and use their existing distinct port ranges.
- Evaluation starts only after the monitored training PID exits and both required checkpoint files are non-empty.
- Each child is allowed to finish independently; the orchestrator reports both exit statuses.

---

### Task 1: Tested post-training orchestrator

**Files:**
- Create: `examples/modelExtensions/CoT/scripts/run_posttrain_libero_parallel_eval.sh`
- Create: `tests/test_posttrain_libero_parallel_eval.py`

**Interfaces:**
- Consumes environment variables `TRAIN_PID`, `MODEL_DIR`, `EXPECTED_STEP`, `LIBERO_EVAL_SCRIPT`, `LIBERO_PLUS_EVAL_SCRIPT`, `POLL_SECONDS`, and `GPU_RELEASE_WAIT_SECONDS`.
- Produces `<MODEL_DIR>/posttrain_parallel_eval.log`, both existing evaluation output trees, and an exit status of zero only when both evaluation launchers succeed.

- [ ] **Step 1: Write failure-gate and parallel-child tests**

Create subprocess tests that use a temporary model directory and fake Bash launchers. Cover: missing final artifacts launches nothing; valid artifacts launch both children concurrently; one child failure does not prevent the other child from completing and makes the orchestrator fail.

- [ ] **Step 2: Run tests and verify they fail before implementation**

Run: `python -m unittest tests.test_posttrain_libero_parallel_eval -v`

Expected: FAIL because `run_posttrain_libero_parallel_eval.sh` does not exist.

- [ ] **Step 3: Implement the minimal orchestrator**

Implement strict environment validation, PID polling, artifact gates for `checkpoints/steps_${EXPECTED_STEP}_pytorch_model.pt` and `final_model/pytorch_model.pt`, an optional post-training release delay, concurrent launcher startup, independent waits, timestamped status logging, and combined exit status.

- [ ] **Step 4: Run the focused tests**

Run: `python -m unittest tests.test_posttrain_libero_parallel_eval -v`

Expected: all tests PASS.

- [ ] **Step 5: Validate repository shell and Python syntax**

Run: `bash -n examples/modelExtensions/CoT/scripts/run_posttrain_libero_parallel_eval.sh`

Run: `python -m py_compile tests/test_posttrain_libero_parallel_eval.py`

Expected: both commands exit zero.

- [ ] **Step 6: Commit the orchestrator and tests**

```bash
git add examples/modelExtensions/CoT/scripts/run_posttrain_libero_parallel_eval.sh tests/test_posttrain_libero_parallel_eval.py docs/superpowers/plans/2026-08-14-libero-posttrain-parallel-eval.md
git commit -m "eval: automate parallel LIBERO post-training runs"
```

### Task 2: Arm and verify the live watcher

**Files:**
- No repository files changed.

**Interfaces:**
- Consumes active launcher PID `6562` and model directory `/root/data/yxz/outputs/qwen35_gr00t_libero_CoT_v3_q32_depthcond_8gpu_bs16`.
- Produces detached tmux session `libero_v3_posteval` and the watcher log under that model directory.

- [ ] **Step 1: Re-run both existing launchers in dry-run mode**

Use the latest complete intermediate checkpoint only to validate task/GPU planning. Standard LIBERO must print four suite jobs on GPUs 0-3; LIBERO-plus must print eight disjoint jobs on GPUs 0-7.

- [ ] **Step 2: Start the orchestrator in detached tmux**

Start `libero_v3_posteval` with `TRAIN_PID=6562`, `EXPECTED_STEP=60000`, the active model directory, and a 30-second GPU release delay.

- [ ] **Step 3: Verify the watcher is armed but evaluation is not running**

Run `tmux list-sessions`, inspect the watcher log, and search the process table. Expected: the watcher reports that it is waiting for PID 6562; no LIBERO evaluation server or worker exists yet.

- [ ] **Step 4: Report live status**

Provide the session name, watcher log path, current training step/ETA, and the exact two result roots.
