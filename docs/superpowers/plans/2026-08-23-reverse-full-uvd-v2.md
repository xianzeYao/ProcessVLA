# Reverse Full-UVD V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reverse stride-4 full-episode UVD plan before the existing forward local UVD trace in the V2 no-query, depth-conditioned LIBERO architecture.

**Architecture:** Extend the optional V2 geometry layout with a full-UVD token group ordered before local UVD. Generate reverse full-episode labels in the existing LIBERO CoT dataset, share the numeric UVD head across both groups, and condition the q0 action model on depth, full-UVD, and local-UVD hidden tokens.

**Tech Stack:** Python 3.10, NumPy, PyTorch, OmegaConf YAML, pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-reverse-full-uvd-v2-design.md`

## Global Constraints

- Keep `num_target_vision_tokens: 0` and `include_depth_in_action_condition: true` in the new experiment.
- Keep the existing forward local UVD target and all existing V2 losses unchanged.
- Use real full-episode frames in reverse order with exactly `full_uvd_stride: 4`; always include the task-completion and current frames.
- Use `full_uvd_num_points: 128` for LIBERO and fail rather than truncate if a sample exceeds that capacity.
- Put reverse full-UVD tokens before forward local-UVD tokens so local tokens can attend to the complete global plan.
- Share the existing `uvd_head`; use a separate learned full-trajectory seed.
- Default the full-UVD feature to disabled so existing V2 and V3 configs/checkpoints remain compatible.

---

### Task 1: Reverse full-episode data labels

**Files:**
- Modify: `starVLA/dataloader/gr00t_lerobot/cot_geometry.py`
- Modify: `tests/test_cot_geometry.py`

**Interfaces:**
- Produces: `sample_reverse_uvd_indices(start: int, end: int, stride: int, max_points: int) -> np.ndarray`.
- Produces per-sample keys: `uvd_full`, `uvd_full_valid_mask`, `uvd_full_frame_indices`, `uvd_full_time`, and `uvd_full_endpoint_indices` when configured.

- [ ] **Step 1: Write failing reverse-sampling tests**

Add literal tests asserting:

```python
np.testing.assert_array_equal(
    sample_reverse_uvd_indices(2, 11, stride=4, max_points=4),
    [11, 7, 3, 2],
)
```

and that an interval requiring five points with `max_points=4` raises `ValueError` containing `capacity`.

- [ ] **Step 2: Run the tests and verify RED**

```bash
/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python -m pytest -q \
  tests/test_cot_geometry.py -k reverse
```

Expected: import or missing-function failure for `sample_reverse_uvd_indices`.

- [ ] **Step 3: Implement the sampler and optional dataset fields**

Implement validation for ordered endpoints, positive stride/capacity, generate `np.arange(end, start - 1, -stride)`, append `start` if needed, and raise before returning when the result exceeds `max_points`. In `_geometry_targets`, preserve the local target exactly and, only when `full_uvd_num_points > 0`, transform the reverse full indices through the same UVD normalization and validity logic.

- [ ] **Step 4: Run data tests and verify GREEN**

```bash
/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python -m pytest -q \
  tests/test_cot_geometry.py tests/test_cot_v3_lerobot_dataset.py
```

Expected: all selected tests pass.

### Task 2: Geometry layout, embedding, mask, and target packing

**Files:**
- Modify: `starVLA/model/modules/geometric_cot_v2.py`
- Modify: `tests/test_geometric_cot_v2.py`

**Interfaces:**
- Extend `GeometryTokenLayout` with `full_uvd_points_per_hand: int = 0`.
- Extend `GeometrySequenceSlices` with `uvd_full: slice` while retaining `uvd` as local.
- Produce `pack_full_uvd_targets_time_major(examples, layout, device=...) -> PackedUVDTargets`.

- [ ] **Step 1: Write failing layout and packing tests**

Use a layout with one depth query, two full points, and two local points. Assert exact sequence slices `[native, depth_current, depth_future, full, local]`, exact total token count, descending default full times `[1.0, 0.0]`, and packing from `uvd_full*` fields into fixed slots.

- [ ] **Step 2: Run the tests and verify RED**

```bash
/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python -m pytest -q \
  tests/test_geometric_cot_v2.py -k 'full_uvd or reverse_full'
```

Expected: constructor/signature failures because the full-UVD layout does not exist.

- [ ] **Step 3: Implement the optional full group**

Keep all old properties stable at `full_uvd_points_per_hand=0`. Create a full seed only when the count is positive, concatenate full embeddings before local embeddings, use descending physical times for full queries, and apply block-causal same-time attention independently within full and local multi-hand groups. Factor the packer internally so local uses `uvd*` keys and full uses `uvd_full*` keys without duplicating validation.

- [ ] **Step 4: Run geometry tests and verify GREEN**

```bash
/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python -m pytest -q \
  tests/test_geometric_cot_v2.py tests/test_geometric_cot_v3.py
```

Expected: all selected tests pass, including unchanged V3 tests.

### Task 3: V2 framework conditioning and objectives

**Files:**
- Modify: `starVLA/model/framework/VLM4A/QwenGR00TCoTV2.py`
- Modify: `tests/test_qwen_gr00t_cot_v2.py`

**Interfaces:**
- Extend `GeometryHiddenSplit` with optional `uvd_full` hidden tokens.
- Add `_prepare_full_uvd_targets(...)`, `_predict_full_uvd(...)`, and `_compute_full_uvd_losses(...)`.
- Add output keys `uvd_full_loss`, `uvd_full_absolute_loss`, and `uvd_full_relative_loss` when enabled.

- [ ] **Step 1: Write failing split, conditioning, and loss tests**

Assert that a configured split orders the q0-depth condition as native, current depth, future depth, full UVD, local UVD. Assert that local hidden queries can read full keys but full queries cannot read local keys. Assert the total objective adds `lambda_uvd_full * uvd_full_loss`, while a layout with zero full points retains the old output contract.

- [ ] **Step 2: Run the tests and verify RED**

```bash
/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python -m pytest -q \
  tests/test_qwen_gr00t_cot_v2.py -k full_uvd
```

Expected: missing layout, split, or output behavior failures.

- [ ] **Step 3: Implement the minimal optional framework path**

Parse `full_uvd_num_points`, `lambda_uvd_full`, and `lambda_uvd_full_relative`. Split full tokens before local tokens, add full hidden tokens before local tokens in the shared action condition, decode both groups through `uvd_head`, and add the masked full loss only when enabled. Preserve `_decode_geometry`'s three-value signature for V3 compatibility and expose full prediction through a separate helper.

- [ ] **Step 4: Preserve intervention semantics**

Whole-geometry and UVD interventions must carry, zero, or shuffle full-UVD together with local UVD when it exists. Direct `GeometryHiddenSplit` construction without `uvd_full` remains valid for old tests and V3.

- [ ] **Step 5: Run V2/V3 framework tests and verify GREEN**

```bash
/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python -m pytest -q \
  tests/test_qwen_gr00t_cot_v2.py tests/test_qwen_gr00t_cot_v3.py
```

Expected: all selected tests pass.

### Task 4: Trainer metrics and isolated q0-depth experiment

**Files:**
- Modify: `starVLA/training/train_starvla_cot_v1.py`
- Modify: `tests/test_cot_trainer_objective.py`
- Create: `examples/modelExtensions/CoT/configs/qwen35_gr00t_libero_CoT_v2_q0_depthcond_reverse_full_uvd_s4.yaml`
- Modify: `tests/test_cot_v2_entrypoints.py`

**Interfaces:**
- Log weighted and unweighted full-UVD aggregate, absolute, and relative losses.
- Produce a launchable LIBERO experiment differing from q0-depth only by run ID and full-UVD options.

- [ ] **Step 1: Write failing trainer/config tests**

Extend the trainer fixture with literal full losses and assert `weighted_uvd_full_loss == 0.2 * uvd_full_loss` and inclusion in `weighted_aux_loss`. Add a config-equivalence test that removes the new run ID and exactly these options before comparing to the existing q0-depth YAML:

```yaml
framework.geometry.full_uvd_num_points: 128
framework.geometry.lambda_uvd_full: 0.2
framework.geometry.lambda_uvd_full_relative: 0.1
datasets.vla_data.cot_geometry.full_uvd_stride: 4
datasets.vla_data.cot_geometry.full_uvd_num_points: 128
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python -m pytest -q \
  tests/test_cot_trainer_objective.py tests/test_cot_v2_entrypoints.py \
  -k 'full_uvd or reverse_full'
```

Expected: missing metrics and missing YAML failures.

- [ ] **Step 3: Implement metrics and create the exact config**

Add optional trainer metrics without changing V1/V3 output requirements. Copy the existing LIBERO q0-depth config, set a unique run ID, and add only the five full-UVD options listed above.

- [ ] **Step 4: Run trainer/config tests and verify GREEN**

```bash
/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python -m pytest -q \
  tests/test_cot_trainer_objective.py tests/test_cot_v2_entrypoints.py
```

Expected: all selected tests pass.

### Task 5: Compatibility and final verification

**Files:**
- Verify all files above; no new production behavior is added in this task.

**Interfaces:**
- Produces fresh regression evidence for data, V2, V3, trainer, and configuration contracts.

- [ ] **Step 1: Run focused regression suites**

```bash
/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python -m pytest -q \
  tests/test_cot_geometry.py \
  tests/test_geometric_cot_v2.py \
  tests/test_geometric_cot_v3.py \
  tests/test_qwen_gr00t_cot_v2.py \
  tests/test_qwen_gr00t_cot_v3.py \
  tests/test_cot_trainer_objective.py \
  tests/test_cot_v2_entrypoints.py
```

- [ ] **Step 2: Compile changed Python modules**

```bash
/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python -m py_compile \
  starVLA/dataloader/gr00t_lerobot/cot_geometry.py \
  starVLA/model/modules/geometric_cot_v2.py \
  starVLA/model/framework/VLM4A/QwenGR00TCoTV2.py \
  starVLA/training/train_starvla_cot_v1.py
```

- [ ] **Step 3: Verify scope and whitespace**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors and only planned files are modified.
