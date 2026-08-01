# RoboCasa Variable-GPU Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the RoboCasa-GR1 evaluator use 1–24 unique GPU workers while preserving the existing 24-task evaluation protocol and four-GPU behavior.

**Architecture:** The Python manifest builder remains the authoritative protocol validator, while the shell launcher performs the same checks before starting processes. Existing round-robin scheduling, one-server/one-simulator-per-GPU execution, task definitions, and aggregation remain unchanged.

**Tech Stack:** Python 3, pytest, Bash, Accelerate policy server, RoboCasa simulator workers.

## Global Constraints

- Accept 1–24 unique GPU identifiers.
- Keep exactly 24 RoboCasa tasks and the configured episode count per task.
- Keep `worker_id = task_index % num_workers` assignment.
- Keep one policy server and one sequential simulator worker per GPU entry.
- Preserve the existing four-GPU behavior.
- Reject empty, over-24, and duplicate GPU lists before launching a server.
- Do not change task definitions, seeds, action settings, result formats, aggregation, or metrics.

---

### Task 1: Generalize manifest GPU validation

**Files:**
- Create: `tests/test_robocasa_eval_protocol.py`
- Modify: `examples/simBenchmarks/Robocasa_tabletop/eval_files/robocasa_eval_protocol.py:31-48`

**Interfaces:**
- Consumes: `build_manifest(checkpoint, gpus, num_episodes, base_port, run_dir, env_names, save_video)`.
- Produces: manifests for 1–24 unique integer-valued GPU identifiers; invalid lists raise `ValueError` before task records are built.

- [ ] **Step 1: Write failing Python protocol tests**

Add tests that call the real `build_manifest` with 24 stable task names:

```python
from pathlib import Path

import pytest

from examples.simBenchmarks.Robocasa_tabletop.eval_files.robocasa_eval_protocol import (
    build_manifest,
)


def _manifest(gpus: list[str]):
    return build_manifest(
        checkpoint="checkpoint.pt",
        gpus=gpus,
        num_episodes=50,
        base_port=18000,
        run_dir=Path("/tmp/robocasa-eval"),
        env_names=[f"suite/task_{index:02d}" for index in range(24)],
        save_video=False,
    )


def test_build_manifest_supports_eight_unique_gpus():
    manifest = _manifest([str(index) for index in range(8)])
    assert manifest["gpus"] == list(range(8))
    assert len(manifest["tasks"]) == 24
    assert [task["worker_id"] for task in manifest["tasks"]] == [
        index % 8 for index in range(24)
    ]
    assert {worker: sum(task["worker_id"] == worker for task in manifest["tasks"])
            for worker in range(8)} == {worker: 3 for worker in range(8)}


@pytest.mark.parametrize(
    ("gpus", "message"),
    [
        ([], "between 1 and 24"),
        ([str(index) for index in range(25)], "between 1 and 24"),
        (["0", "1", "1"], "unique"),
    ],
)
def test_build_manifest_rejects_invalid_gpu_lists(gpus, message):
    with pytest.raises(ValueError, match=message):
        _manifest(gpus)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python -m pytest \
  tests/test_robocasa_eval_protocol.py -q
```

Expected: the eight-GPU case fails with the existing “requires exactly 4 GPUs” error.

- [ ] **Step 3: Implement minimal Python validation**

Replace the fixed four-GPU check with count and uniqueness checks after converting identifiers to integers:

```python
if not 1 <= len(gpus) <= 24:
    raise ValueError(f"RoboCasa evaluation requires between 1 and 24 GPUs, got {len(gpus)}")

gpu_values = [int(gpu) for gpu in gpus]
if len(set(gpu_values)) != len(gpu_values):
    raise ValueError("RoboCasa evaluation requires unique GPU identifiers")
```

Keep the 24-task and positive-episode checks unchanged.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all new protocol tests pass.

- [ ] **Step 5: Commit the Python protocol behavior**

```bash
git add tests/test_robocasa_eval_protocol.py \
  examples/simBenchmarks/Robocasa_tabletop/eval_files/robocasa_eval_protocol.py
git commit -m "feat(eval): support variable RoboCasa GPU manifests"
```

### Task 2: Generalize shell launcher validation

**Files:**
- Modify: `tests/test_robocasa_eval_protocol.py`
- Modify: `examples/simBenchmarks/Robocasa_tabletop/eval_files/run_multigpu_eval.sh:19-48`

**Interfaces:**
- Consumes: comma-separated `GPUS`, `CHECKPOINT`, `DRY_RUN=1`, and normal evaluation environment variables.
- Produces: one planned policy server/worker per unique GPU and 24 round-robin task records without launching processes in dry-run mode.

- [ ] **Step 1: Add a failing eight-GPU shell dry-run test**

Use `tmp_path` to create an empty checkpoint, launch the real shell script with `GPUS=0,1,2,3,4,5,6,7`, `DRY_RUN=1`, `NUM_EPISODES=1`, and Python executables set to `sys.executable`. Assert exit status zero, 24 `[robocasa] plan task=` lines, and a manifest whose worker IDs are `0..7`, each appearing three times. Add duplicate-GPU and empty-GPU subprocess cases that assert nonzero status and the corresponding validation message.

- [ ] **Step 2: Run the shell-focused tests and verify RED**

Run:

```bash
/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python -m pytest \
  tests/test_robocasa_eval_protocol.py -q
```

Expected: the eight-GPU dry run fails with “requires exactly 4 GPUs”.

- [ ] **Step 3: Implement minimal Bash validation**

Distinguish an unset GPU variable from an explicitly empty one:

```bash
GPUS="${GPUS-4,5,6,7}"
```

After parsing `GPU_LIST`, reject counts outside 1–24 and duplicate identifiers with an associative array before creating the run directory or invoking the manifest command. Leave server launch, worker loops, task assignment, cleanup, and aggregation unchanged.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: Python validation and shell dry-run tests all pass.

- [ ] **Step 5: Run the real eight-GPU dry run**

Run with the completed V2 checkpoint:

```bash
CHECKPOINT=/root/data/yxz/outputs/qwen35_gr00t_robocasa_fourier_CoT_v2_8gpu_bs16/checkpoints/steps_100000_pytorch_model.pt \
GPUS=0,1,2,3,4,5,6,7 \
BASE_PORT=18398 \
NUM_EPISODES=50 \
DRY_RUN=1 \
RUN_TIMESTAMP=robocasa_cot_v2_100k_eval_8gpu_dryrun \
bash examples/simBenchmarks/Robocasa_tabletop/eval_files/run_multigpu_eval.sh
```

Expected: eight GPU workers, 24 task plans, three tasks per worker, and no server or simulator process started.

- [ ] **Step 6: Run regression tests and syntax checks**

```bash
bash -n examples/simBenchmarks/Robocasa_tabletop/eval_files/run_multigpu_eval.sh
/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python -m pytest \
  tests/test_robocasa_eval_protocol.py -q
```

Expected: Bash syntax succeeds and all focused tests pass.

- [ ] **Step 7: Commit launcher support**

```bash
git add tests/test_robocasa_eval_protocol.py \
  examples/simBenchmarks/Robocasa_tabletop/eval_files/run_multigpu_eval.sh
git commit -m "feat(eval): run RoboCasa on variable GPU counts"
```
