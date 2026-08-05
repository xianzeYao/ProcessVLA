# RoboCasa CoT V2 Q0 Eight-GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible RoboCasa Fourier CoT V2 configuration that removes only the 32 action-side future/query tokens and launches on eight GPUs through the existing common runner.

**Architecture:** Copy the existing CoT V2 YAML and change only `run_id` and `framework.action_model.num_target_vision_tokens`. Reuse the existing common Accelerate/DeepSpeed launcher with environment variables; no new training code or shell wrapper is introduced.

**Tech Stack:** YAML, Bash, Accelerate, DeepSpeed ZeRO-2, PyTorch, pytest.

## Global Constraints

- The source configuration is `examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v2.yaml`.
- The new run ID is exactly `qwen35_gr00t_robocasa_fourier_CoT_v2_q0_8gpu_bs16`.
- `framework.action_model.num_target_vision_tokens` is exactly `0`.
- Every other YAML field remains identical to the source configuration.
- Training uses GPUs `0,1,2,3,4,5,6,7`, eight processes, per-device batch 16, global batch 128, and the existing ZeRO-2 launcher.

---

### Task 1: Add and verify the Q0 experiment configuration

**Files:**
- Create: `examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v2_q0.yaml`
- Modify: `tests/model/modules/action_model/test_gr00t_n16_n17_action_heads.py`

**Interfaces:**
- Consumes: `run_qwen35_gr00t_CoT_v2_common.sh`, which reads `CONFIG_YAML`, `RUN_ID`, `RUN_ROOT_DIR`, and `NUM_PROCESSES`.
- Produces: a Q0 YAML accepted by `train_starvla_cot_v2.py` and a verified eight-process dry-run command.

- [ ] **Step 1: Add a failing configuration-diff test**

Add a test that loads the source and Q0 YAML files, asserts the new run ID and zero target tokens, normalizes those two values back to the source values, and then asserts complete dictionary equality. Before the new YAML exists, the test must fail with `FileNotFoundError`.

- [ ] **Step 2: Add a zero-token action-head smoke test**

Construct the legacy `FlowmatchingActionHead` with a tiny two-layer DiT configuration and `num_target_vision_tokens=0`, replace the DiT with the existing capture model, run one forward pass, and assert that the captured action-side sequence has exactly `H` tokens and no extra query slots.

- [ ] **Step 3: Run the focused tests and confirm RED**

Run:

```bash
/tmp/cot-vla-gr00t-test-env/bin/python -m pytest \
  tests/model/modules/action_model/test_gr00t_n16_n17_action_heads.py -q
```

Expected: the configuration test fails because the Q0 YAML does not yet exist; the zero-token smoke test may already pass because the legacy head uses a valid zero-length embedding.

- [ ] **Step 4: Create the minimal Q0 YAML**

Copy all source YAML values exactly, changing only:

```yaml
run_id: qwen35_gr00t_robocasa_fourier_CoT_v2_q0_8gpu_bs16
framework:
  action_model:
    num_target_vision_tokens: 0
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the focused pytest command from Step 3. Expected: all tests pass.

- [ ] **Step 6: Verify the eight-GPU dry run**

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NUM_PROCESSES=8 \
CONFIG_YAML=examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v2_q0.yaml \
RUN_ID=qwen35_gr00t_robocasa_fourier_CoT_v2_q0_8gpu_bs16 \
RUN_ROOT_DIR=/root/data/yxz/outputs \
DRY_RUN=1 \
bash examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_CoT_v2_common.sh
```

Expected: the printed command contains the Q0 YAML, `--num_processes 8`, the Q0 run ID, and `train_starvla_cot_v2.py`.

- [ ] **Step 7: Run static and diff verification**

Run `git diff --check` and inspect the YAML comparison test to confirm no hidden configuration differences are allowed.

- [ ] **Step 8: Commit**

```bash
git add \
  examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v2_q0.yaml \
  tests/model/modules/action_model/test_gr00t_n16_n17_action_heads.py
git commit -m "exp: add RoboCasa CoT v2 Q0 config"
```
