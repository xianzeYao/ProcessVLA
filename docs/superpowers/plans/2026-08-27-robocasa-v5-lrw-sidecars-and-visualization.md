# RoboCasa V5 LRW Sidecars and Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate validated bilateral thumb/index/wrist UVD sidecars for all 24 local RoboCasa GR1 tasks and render a fixed-seed audit visualization for ten randomly selected tasks.

**Architecture:** A small immutable sidecar contract owns paths, shapes, and validation. A replay CLI maps each rerender LeRobot episode to its original HDF5 demo, restores MuJoCo states without rerendering RGB-D, extracts six named bodies, projects through stored agentview matrices, and atomically writes one episode NPZ. A separate visualization CLI reads only rerender videos/parquet plus completed sidecars, records a fixed random selection before rendering, and produces per-task overlays, numeric records, and one contact sheet.

**Tech Stack:** Python 3.10, NumPy, h5py, pandas/pyarrow, MuJoCo/robosuite/RoboCasa, OpenCV or PIL, pytest, LeRobot episode layout.

**Spec:** `docs/superpowers/specs/2026-08-27-robocasa-cot-v5-lrw-hand-configuration-design.md`

## Global Constraints

- RoboCasa only; do not modify LIBERO V3 sidecars or any V2/V3/V4 dataset behavior.
- Landmark order is exactly `[L=thumb distal, R=index intermediate, W=wrist/hand base]`.
- Hand order is exactly `[left, right]`; sidecar axes are `[frame, hand, landmark, coordinate]`.
- Sidecars live below each rerender task root and are atomic, idempotent, resumable, and strictly validated before metadata is advertised.
- Projection uses the already stored `agentview_K` and `agentview_T_world_camera`; no RGB-D rerender is performed.
- Positive-depth projected points remain projection-valid; `agentview_in_frame` additionally applies the DIAL content-region mask.
- The audit selects ten distinct tasks with a fixed seed, writes the selection manifest before rendering, and never silently replaces failures.
- Preserve all unrelated dirty-worktree changes.

---

## File map

- Create `starVLA/robocasa_hand_lrw.py`: immutable LRW constants, sidecar path, typed payload, validation, loading, and projection wrapper.
- Create `starVLA/robocasa_hand_lrw_cli.py`: task/episode discovery, HDF5 alignment, simulator replay orchestration, metadata finalization, and report writing.
- Create `examples/modelExtensions/CoT/scripts/build_robocasa_hand_lrw_sidecars.py`: thin executable plus body extraction and atomic sidecar writer.
- Create `examples/modelExtensions/CoT/scripts/visualize_robocasa_hand_lrw_sidecars.py`: deterministic selection, video-frame loading, LRW trajectory drawing, records, and contact sheet.
- Create `tests/test_robocasa_hand_lrw_sidecar.py`: shape/dtype/path/projection contract.
- Create `tests/test_robocasa_hand_lrw_generator.py`: body order, replay order, atomic write, and metadata gating.
- Create `tests/test_robocasa_hand_lrw_generator_cli.py`: 24-task discovery, source mapping, validate-only behavior, and report contract.
- Create `tests/test_robocasa_hand_lrw_visualization.py`: deterministic selection, six-frame sampling, overlay, manifest, and no-resampling contract.
- Generate dataset files below `/root/data/yxz/datasets/robocasa_fourier_rerender/*/geometry/hand_lrw/`; these are data artifacts and are not committed.
- Generate audit artifacts below `artifacts/robocasa_hand_lrw_v5/`; commit only code, tests, and lightweight JSON summaries, not videos or large images.

### Task 1: Immutable bilateral LRW sidecar contract

**Files:**
- Create: `starVLA/robocasa_hand_lrw.py`
- Test: `tests/test_robocasa_hand_lrw_sidecar.py`

**Interfaces:**
- Produces: `HAND_NAMES`, `LANDMARK_NAMES`, `LANDMARK_BODY_NAMES`, `HandLRWSidecar`, `hand_lrw_path(dataset_root, episode_id)`, `validate_hand_lrw_payload(payload, frame_count, width, height)`, and `load_hand_lrw_sidecar(path, frame_count, width, height)`.
- Consumes: `project_world_to_agentview_uvd` from `starVLA.gripper_triangle` for arbitrary extra point axes.

- [ ] **Step 1: Write failing path and validation tests**

```python
from starVLA.robocasa_hand_lrw import (
    HAND_NAMES, LANDMARK_NAMES, hand_lrw_path, validate_hand_lrw_payload,
)


def test_bilateral_lrw_contract_has_explicit_axes(tmp_path):
    assert HAND_NAMES == ("left", "right")
    assert LANDMARK_NAMES == ("thumb", "index", "wrist")
    assert hand_lrw_path(tmp_path, 1001).relative_to(tmp_path).as_posix() == (
        "geometry/hand_lrw/chunk-001/episode_001001.npz"
    )
    payload = valid_payload(frame_count=2)  # arrays [2,2,3,3], masks [2,2,3]
    sidecar = validate_hand_lrw_payload(payload, frame_count=2, width=256, height=256)
    assert sidecar.world_xyz.shape == (2, 2, 3, 3)
    assert sidecar.agentview_in_frame.shape == (2, 2, 3)
```

Also assert exact keys/dtypes, finite world coordinates, positive depth for projection-valid points, `in_frame <= projection_valid`, image bounds, negative episode rejection, and frame-count mismatch rejection.

- [ ] **Step 2: Run the focused test and confirm it fails on the missing module**

Run: `PYTHONPATH=. pytest -q tests/test_robocasa_hand_lrw_sidecar.py`

Expected: collection failure with `ModuleNotFoundError: starVLA.robocasa_hand_lrw`.

- [ ] **Step 3: Implement the minimal immutable contract**

```python
HAND_NAMES = ("left", "right")
LANDMARK_NAMES = ("thumb", "index", "wrist")
LANDMARK_BODY_NAMES = (
    (
        "gripper0_left_L_thumb_distal_link",
        "gripper0_left_L_index_intermediate_link",
        "gripper0_left_left_hand",
    ),
    (
        "gripper0_right_R_thumb_distal_link",
        "gripper0_right_R_index_intermediate_link",
        "gripper0_right_right_hand",
    ),
)

@dataclass(frozen=True)
class HandLRWSidecar:
    world_xyz: np.ndarray                 # [T,2,3,3]
    agentview_uvd_pixels: np.ndarray      # [T,2,3,3]
    agentview_projection_valid: np.ndarray # [T,2,3]
    agentview_in_frame: np.ndarray         # [T,2,3]
```

Use exact-key validation and the same atomic-read semantics as `starVLA/gripper_triangle.py`; do not alias the LIBERO shape validator.

- [ ] **Step 4: Run contract tests**

Run: `PYTHONPATH=. pytest -q tests/test_robocasa_hand_lrw_sidecar.py tests/test_libero_gripper_triangle_sidecar.py`

Expected: both suites pass, proving RoboCasa's extra hand axis did not alter LIBERO.

- [ ] **Step 5: Commit the contract**

```bash
git add starVLA/robocasa_hand_lrw.py tests/test_robocasa_hand_lrw_sidecar.py
git commit -m "feat: add RoboCasa bilateral LRW sidecar contract"
```

### Task 2: Physical body extraction and atomic episode writes

**Files:**
- Create: `examples/modelExtensions/CoT/scripts/build_robocasa_hand_lrw_sidecars.py`
- Test: `tests/test_robocasa_hand_lrw_generator.py`

**Interfaces:**
- Consumes: `LANDMARK_BODY_NAMES`, `validate_hand_lrw_payload`, and `load_hand_lrw_sidecar` from Task 1.
- Produces: `extract_bilateral_lrw_world(env, states, reset_to) -> np.ndarray`, `write_sidecar_atomic(...) -> Literal["written","skipped"]`, `EpisodeSidecarSpec`, `finalize_geometry_metadata(...)`, and `summarize_sidecar(...)`.

- [ ] **Step 1: Write failing extraction and writer tests**

```python
def test_extracts_left_then_right_and_thumb_index_wrist_order():
    trajectory = extract_bilateral_lrw_world(fake_env, states, fake_reset_to)
    assert trajectory.shape == (len(states), 2, 3, 3)
    np.testing.assert_allclose(trajectory[0, 0], [left_thumb, left_index, left_wrist])
    np.testing.assert_allclose(trajectory[0, 1], [right_thumb, right_index, right_wrist])
    assert reset_states == list(states)


def test_metadata_is_advertised_only_after_every_sidecar_validates(tmp_path):
    with pytest.raises(Exception):
        finalize_geometry_metadata(task_root, specs_with_one_missing_episode)
    assert "hand_lrw" not in read_info(task_root).get("geometry_paths", {})
```

Also cover missing body names, non-finite body positions, skip-valid-existing behavior, corrupt-existing behavior, overwrite, temporary-file cleanup, and per-hand finger-distance/triangle-area summaries.

- [ ] **Step 2: Run the generator tests and confirm missing symbols fail**

Run: `PYTHONPATH=. pytest -q tests/test_robocasa_hand_lrw_generator.py`

Expected: import failure for the new builder functions.

- [ ] **Step 3: Implement body extraction and atomic writes**

```python
def extract_bilateral_lrw_world(env, states, reset_to):
    body_ids = np.asarray([
        [env.sim.model.body_name2id(name) for name in hand]
        for hand in LANDMARK_BODY_NAMES
    ], dtype=np.int64)
    output = np.empty((len(states), 2, 3, 3), dtype=np.float32)
    for frame_index, state in enumerate(states):
        reset_to(env, {"states": state})
        env._get_observations(force_update=True)
        output[frame_index] = np.asarray(env.sim.data.body_xpos[body_ids], dtype=np.float32)
    return output
```

Before this loop, the CLI initializes the demo with its `model_file`, `ep_meta`, and first state. Write NPZ keys exactly as Task 1 defines, fsync the temporary file, reload/validate it, then `os.replace` it.

- [ ] **Step 4: Run generator and regression tests**

Run: `PYTHONPATH=. pytest -q tests/test_robocasa_hand_lrw_generator.py tests/test_libero_gripper_triangle_generator.py`

Expected: pass.

- [ ] **Step 5: Commit physical extraction and writing**

```bash
git add examples/modelExtensions/CoT/scripts/build_robocasa_hand_lrw_sidecars.py tests/test_robocasa_hand_lrw_generator.py
git commit -m "feat: extract RoboCasa LRW sidecars from replay states"
```

### Task 3: Complete 24-task generator CLI and validation report

**Files:**
- Create: `starVLA/robocasa_hand_lrw_cli.py`
- Modify: `examples/modelExtensions/CoT/scripts/build_robocasa_hand_lrw_sidecars.py`
- Test: `tests/test_robocasa_hand_lrw_generator_cli.py`

**Interfaces:**
- Consumes: `FOURIER_TASKS`, Task 1 sidecar functions, Task 2 extraction/writer/finalizer, `project_world_to_agentview_uvd`, and `dial_content_region_mask`.
- Produces: `discover_task_roots(...)`, `resolve_episode_source(...)`, `parse_args(...)`, `run(...) -> dict`, and `main()`.

- [ ] **Step 1: Write failing discovery/alignment/validate-only tests**

```python
def test_discovers_canonical_24_tasks_in_fourier_order(tmp_path):
    roots = discover_task_roots(rerender_root, hdf5_root)
    assert [item.task.basename for item in roots] == [task.basename for task in FOURIER_TASKS]


def test_episode_source_requires_demo_and_frame_alignment(fake_task):
    source = resolve_episode_source(fake_task, episode_id=7)
    assert source.demo_id == "demo_7"
    np.testing.assert_array_equal(source.state_indices, np.arange(source.frame_count))
```

Build fixtures with `meta/info.json`, episode parquet columns `source.hdf5_demo_id` and `frame_index`, depth/camera NPZ paths, and a fake HDF5. Assert duplicate demo IDs, non-contiguous frame indices, missing camera matrices, HDF5 state-count mismatch, missing/corrupt sidecars, and partial metadata finalization all fail loudly and appear in the JSON report.

- [ ] **Step 2: Run the CLI test and confirm failure**

Run: `PYTHONPATH=. pytest -q tests/test_robocasa_hand_lrw_generator_cli.py`

Expected: import failure for `starVLA.robocasa_hand_lrw_cli`.

- [ ] **Step 3: Implement deterministic task/episode mapping and replay**

The CLI arguments are exact:

```text
--dataset-root /root/data/yxz/datasets/robocasa_fourier_rerender
--hdf5-root /root/data/yxz/datasets/robocasa_gr1_original_hdf5/HDF5
--robocasa-repo /root/data/yxz/benchmarks/robocasa-gr1-tabletop-tasks
--task <basename>                 # repeatable, optional
--episode-start 0
--episode-end <exclusive>
--max-episodes <optional>
--overwrite
--validate-only
--report-path artifacts/robocasa_hand_lrw_v5/sidecar_generation_report.json
```

For each episode: load stored camera matrices, open `data/<demo_id>/states`, initialize the environment with demo XML/meta, extract `[T,2,3,3]`, project, compute the DIAL content mask, write atomically, then validate. Aggregate exact selected/written/skipped/validated/missing/corrupt/error counts per task and globally. Call `finalize_geometry_metadata` only when every selected episode for a complete task root is valid; a partial `--episode-*` run must not advertise incomplete coverage.

- [ ] **Step 4: Run unit tests and one-episode local integration**

Run unit tests:

`PYTHONPATH=. pytest -q tests/test_robocasa_hand_lrw_sidecar.py tests/test_robocasa_hand_lrw_generator.py tests/test_robocasa_hand_lrw_generator_cli.py`

Run one episode in the RoboCasa environment:

```bash
PYTHONPATH=. /root/data/yxz/miniforge3/envs/robocasa/bin/python \
  examples/modelExtensions/CoT/scripts/build_robocasa_hand_lrw_sidecars.py \
  --dataset-root /root/data/yxz/datasets/robocasa_fourier_rerender \
  --hdf5-root /root/data/yxz/datasets/robocasa_gr1_original_hdf5/HDF5 \
  --robocasa-repo /root/data/yxz/benchmarks/robocasa-gr1-tabletop-tasks \
  --task PnPBottleToCabinetClose --max-episodes 1 \
  --report-path artifacts/robocasa_hand_lrw_v5/smoke_report.json
```

Expected: one valid NPZ with shape `[T,2,3,3]`; the report is successful but task metadata is not advertised because only a subset was selected.

- [ ] **Step 5: Commit the CLI**

```bash
git add starVLA/robocasa_hand_lrw_cli.py \
  examples/modelExtensions/CoT/scripts/build_robocasa_hand_lrw_sidecars.py \
  tests/test_robocasa_hand_lrw_generator_cli.py
git commit -m "feat: orchestrate RoboCasa LRW sidecar generation"
```

### Task 4: Fixed-seed ten-task UVD trajectory audit

**Files:**
- Create: `examples/modelExtensions/CoT/scripts/visualize_robocasa_hand_lrw_sidecars.py`
- Test: `tests/test_robocasa_hand_lrw_visualization.py`

**Interfaces:**
- Consumes: Task 1 loader/path, canonical `FOURIER_TASKS`, and `sample_real_uvd_indices(start, min(start + 16, T - 1), 6)`.
- Produces: `select_audit_windows(task_roots, seed, task_count)`, `draw_lrw_trajectory_overlay(...)`, `write_contact_sheet(...)`, `run(...)`, and `main()`.

- [ ] **Step 1: Write failing deterministic-selection and rendering tests**

```python
def test_selection_is_fixed_seed_distinct_and_not_resampled():
    first = select_audit_windows(task_roots, seed=42, task_count=10)
    second = select_audit_windows(task_roots, seed=42, task_count=10)
    assert first == second
    assert len({item.task for item in first}) == 10


def test_overlay_and_record_preserve_bilateral_lrw_axes(tmp_path):
    result = render_selection(selection, output_root=tmp_path)
    assert result.record["hand_order"] == ["left", "right"]
    assert result.record["landmark_order"] == ["thumb", "index", "wrist"]
    assert np.asarray(result.record["uvd_pixels"]).shape == (6, 2, 3, 3)
```

Also assert six unique indices for chosen full windows, time/hand/landmark colors are distinguishable, marker size/opacity changes with finite depth, invalid/out-of-frame points are counted, the manifest exists before the first renderer callback, and a renderer failure leaves the original selection plus an error rather than choosing another task.

- [ ] **Step 2: Run the visualization test and confirm failure**

Run: `PYTHONPATH=. pytest -q tests/test_robocasa_hand_lrw_visualization.py`

Expected: import failure for the new visualization module.

- [ ] **Step 3: Implement the audit renderer**

Use seed 42 by default. Select 10 task indices without replacement, then one episode and one base frame with at least 17 frames when available. Record `selection_manifest.json` atomically before opening videos. Draw the six time samples on the base agentview frame: solid lines for left hand, dashed/outlined lines for right; fixed colors for thumb/index/wrist; arrowheads or increasing index labels for time; marker size and legend for metric depth. Write `task/demo/window.json`, `task/demo/window.jpg`, and `contact_sheet.jpg`.

- [ ] **Step 4: Run visualization and sidecar regression tests**

Run: `PYTHONPATH=. pytest -q tests/test_robocasa_hand_lrw_visualization.py tests/test_libero_gripper_triangle_preview.py`

Expected: pass.

- [ ] **Step 5: Commit visualization code**

```bash
git add examples/modelExtensions/CoT/scripts/visualize_robocasa_hand_lrw_sidecars.py \
  tests/test_robocasa_hand_lrw_visualization.py
git commit -m "feat: visualize RoboCasa bilateral LRW trajectories"
```

### Task 5: Build all sidecars and produce the ten-task audit

**Files:**
- Generate: `/root/data/yxz/datasets/robocasa_fourier_rerender/*/geometry/hand_lrw/**/*.npz`
- Generate: `artifacts/robocasa_hand_lrw_v5/sidecar_generation_report.json`
- Generate: `artifacts/robocasa_hand_lrw_v5/audit_seed42/**`

**Interfaces:**
- Consumes: completed Tasks 1-4.
- Produces: validated dataset sidecars and the user-facing ten-task visual audit.

- [ ] **Step 1: Run the complete resumable generator**

```bash
PYTHONPATH=. /root/data/yxz/miniforge3/envs/robocasa/bin/python \
  examples/modelExtensions/CoT/scripts/build_robocasa_hand_lrw_sidecars.py \
  --dataset-root /root/data/yxz/datasets/robocasa_fourier_rerender \
  --hdf5-root /root/data/yxz/datasets/robocasa_gr1_original_hdf5/HDF5 \
  --robocasa-repo /root/data/yxz/benchmarks/robocasa-gr1-tabletop-tasks \
  --report-path artifacts/robocasa_hand_lrw_v5/sidecar_generation_report.json
```

Expected: status `success`, all 24 task roots finalized, zero missing/corrupt/generation errors. If interrupted, rerun the same command; valid existing episodes are skipped.

- [ ] **Step 2: Re-run in validate-only mode**

```bash
PYTHONPATH=. /root/data/yxz/miniforge3/envs/robocasa/bin/python \
  examples/modelExtensions/CoT/scripts/build_robocasa_hand_lrw_sidecars.py \
  --dataset-root /root/data/yxz/datasets/robocasa_fourier_rerender \
  --hdf5-root /root/data/yxz/datasets/robocasa_gr1_original_hdf5/HDF5 \
  --robocasa-repo /root/data/yxz/benchmarks/robocasa-gr1-tabletop-tasks \
  --validate-only \
  --report-path artifacts/robocasa_hand_lrw_v5/sidecar_validation_report.json
```

Expected: every episode validates without simulator replay.

- [ ] **Step 3: Generate the fixed-seed audit**

```bash
PYTHONPATH=. /root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python \
  examples/modelExtensions/CoT/scripts/visualize_robocasa_hand_lrw_sidecars.py \
  --dataset-root /root/data/yxz/datasets/robocasa_fourier_rerender \
  --output-root artifacts/robocasa_hand_lrw_v5/audit_seed42 \
  --seed 42 --task-count 10 --action-horizon 16 --uvd-num-points 6
```

Expected: ten distinct tasks, ten overlays and numeric records, one manifest, one contact sheet, and reported invalid/out-of-frame counts.

- [ ] **Step 4: Inspect every selected overlay and numeric record**

Check that thumb/index tracks follow the visible fingers, wrist tracks attach to the correct hand base, left/right are not swapped, time arrows move continuously, depth encoding changes consistently, and no points are systematically mirrored or shifted into DIAL padding. Record findings in `artifacts/robocasa_hand_lrw_v5/audit_seed42/review.json` with `status`, `reviewed_count=10`, and per-example notes.

- [ ] **Step 5: Commit only lightweight reports and final verification state**

```bash
git add -f artifacts/robocasa_hand_lrw_v5/sidecar_generation_report.json \
  artifacts/robocasa_hand_lrw_v5/sidecar_validation_report.json \
  artifacts/robocasa_hand_lrw_v5/audit_seed42/selection_manifest.json \
  artifacts/robocasa_hand_lrw_v5/audit_seed42/review.json
git commit -m "data: validate RoboCasa LRW sidecars and audit selection"
```

Do not commit NPZ sidecars, videos, or image sheets.
