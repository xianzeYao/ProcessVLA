# LIBERO Gripper-Triangle CoT V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add validated LIBERO gripper-triangle sidecars and a separately registered, time-major `QwenGR00TCoTV3` training path with three landmark trajectories, complete tests, YAML, and a one-click launcher.

**Architecture:** A small shared sidecar module owns the immutable `[time, landmark, coordinate]` schema and deterministic episode paths. A resumable LIBERO replay CLI produces the sidecars, while a V3-only dataset adapter reads them and reuses the existing depth/temporal sampling path. A V3 geometry module and framework subclass preserve V2 action/depth behavior but replace hand semantics with landmark identities and add a disabled-by-default triangle-shape loss interface.

**Tech Stack:** Python 3.10, NumPy, pandas/pyarrow, h5py, LIBERO/robosuite MuJoCo, PyTorch, OmegaConf, Bash, pytest/unittest.

## Global Constraints

- Preserve every existing tracked and untracked user change; do not edit the current V2 model/config/test modifications.
- Do not modify RoboCasa data or training.
- Do not modify LIBERO Parquet files; add deterministic sidecars and optional `meta/info.json` keys only.
- Landmark order is exactly `left_finger_tip`, `right_finger_tip`, `wrist_hand_base`.
- Token order is exactly time-major: `[L0,R0,W0,L1,R1,W1,...]`.
- Occlusion does not invalidate a point; V3 training validity additionally requires an in-frame projection.
- Keep absolute and adjacent-time temporal UVD losses; expose shape loss but set `lambda_uvd_shape: 0.0` in the first YAML.
- Existing V1/V2 registry, configs, launchers, sample fields, checkpoints, and tests must remain unchanged.
- Use test-driven development: observe every new test fail for the intended missing behavior before adding production code.
- Commit only files belonging to the current task; never stage unrelated dirty files.

---

### Task 1: Define and validate the gripper-triangle sidecar contract

**Files:**
- Create: `starVLA/dataloader/gr00t_lerobot/gripper_triangle.py`
- Create: `tests/test_libero_gripper_triangle_sidecar.py`

**Interfaces:**
- Produces: `LANDMARK_NAMES`, `LANDMARK_BODY_NAMES`, `GripperTriangleSidecar`, `gripper_triangle_path(dataset_root, episode_id)`, `project_world_to_agentview_uvd(...)`, `validate_gripper_triangle_payload(payload, frame_count, width, height)`, `load_gripper_triangle_sidecar(path, frame_count, width, height)`.
- Consumers: Tasks 2 and 3.

- [ ] **Step 1: Write failing path and schema tests**

Create fixtures with two frames and literal arrays. Assert episode `1007` resolves to
`geometry/gripper_triangle/chunk-001/episode_001007.npz`; exact keys, float32/bool dtypes,
`[T,3,3]`/`[T,3]` shapes, positive depth, and `in_frame <= projection_valid` are enforced.
Also assert landmark/body constants have the approved order. Cover the shared projection
helper with an identity/exact camera fixture: depth is camera Z, projection validity
requires finite positive depth, and in-frame validity is tracked separately.

- [ ] **Step 2: Run Task 1 tests and verify RED**

Run:

```bash
/root/data/yxz/miniforge3/envs/CoT/bin/python -m pytest tests/test_libero_gripper_triangle_sidecar.py -v
```

Expected: import failure because `gripper_triangle.py` does not exist.

- [ ] **Step 3: Implement the immutable contract**

Define:

```python
LANDMARK_NAMES = ("left_finger_tip", "right_finger_tip", "wrist_hand_base")
LANDMARK_BODY_NAMES = (
    "gripper0_finger_joint1_tip",
    "gripper0_finger_joint2_tip",
    "gripper0_right_gripper",
)

@dataclass(frozen=True)
class GripperTriangleSidecar:
    world_xyz: np.ndarray
    agentview_uvd_pixels: np.ndarray
    agentview_projection_valid: np.ndarray
    agentview_in_frame: np.ndarray
```

Validation must reject unexpected/missing keys, wrong dtype/shape/frame count, non-finite
world/valid UVD values, non-positive valid depth, impossible in-frame masks, and U/V marked
in-frame outside `[0,width) x [0,height)`.

- [ ] **Step 4: Run Task 1 tests and verify GREEN**

Run the Task 1 pytest command. Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add starVLA/dataloader/gr00t_lerobot/gripper_triangle.py tests/test_libero_gripper_triangle_sidecar.py
git commit -m "feat: define LIBERO gripper triangle sidecars"
```

### Task 2: Build resumable LIBERO sidecar generation and validation

**Files:**
- Create: `examples/modelExtensions/CoT/scripts/build_libero_gripper_triangle_sidecars.py`
- Modify: `examples/modelExtensions/CoT/scripts/visualize_libero_gripper_triangle.py`
- Modify: `tests/test_libero_gripper_triangle_sidecar.py`

**Interfaces:**
- Consumes: Task 1 contract and existing `env_kwargs_from_hdf5_attrs` / camera sidecars.
- Produces: `extract_gripper_triangle_world`, `write_sidecar_atomic`, `update_geometry_metadata_atomic`, CLI generation/validation report. Imports projection from the shared Task 1 module.

- [ ] **Step 1: Write failing pure generation tests**

Extend the projection edge cases from Task 1 and assert the approved point order. Use a
small fake MuJoCo model/data object to assert body-name lookup order independently of IDs.

- [ ] **Step 2: Write failing atomic/resume tests**

In a temporary dataset, assert atomic writing produces a valid NPZ without leaving a
temporary file; a valid existing sidecar is skipped without `--overwrite`; a corrupt file
fails in validate-only mode; and metadata is updated with the exact template/order only
after the requested outputs validate.

- [ ] **Step 3: Run focused tests and verify RED**

Run the Task 1 pytest command. Expected: missing generator functions/CLI behavior.

- [ ] **Step 4: Implement the generator**

Reuse source columns `source.hdf5_path`, `source.hdf5_demo_id`, and
`source.hdf5_index` from each episode Parquet. Group episodes by HDF5 so one environment
serves every selected episode from the same task. Restore each indexed state, read body
positions, load stored agentview matrices, project, validate, atomically write the sidecar,
then atomically update `meta/info.json` after the selection completes.

CLI arguments must be:

```text
--dataset-root
--suite (repeatable)
--episode-start
--episode-end
--max-episodes
--overwrite
--validate-only
--report-path
```

The report must include counts, missing/corrupt files, minimum finger distance/triangle
area, projection/in-frame ratios, and per-suite summaries.

- [ ] **Step 5: Make the preview script import shared constants/projection**

Replace its duplicate point constants and projection math with imports from Task 1/the
generator while preserving its CLI and existing 40 preview artifacts. Do not change its
drawing or video semantics.

- [ ] **Step 6: Run tests and compile**

```bash
/root/data/yxz/miniforge3/envs/CoT/bin/python -m pytest tests/test_libero_gripper_triangle_sidecar.py tests/test_libero_gripper_triangle_preview.py -v
/root/data/yxz/miniforge3/envs/CoT/bin/python -m py_compile \
  starVLA/dataloader/gr00t_lerobot/gripper_triangle.py \
  examples/modelExtensions/CoT/scripts/build_libero_gripper_triangle_sidecars.py \
  examples/modelExtensions/CoT/scripts/visualize_libero_gripper_triangle.py
```

Expected: exit zero.

- [ ] **Step 7: Run one real-episode smoke generation**

Use `libero_goal --max-episodes 1` with an isolated temporary dataset metadata/output
fixture or a CLI output override. Verify the sidecar has 112 frames and matches the
already validated preview episode's nondegenerate geometry.

- [ ] **Step 8: Commit Task 2**

Commit only the generator, preview refactor, sidecar module changes, and focused tests.

### Task 3: Add the V3-only LIBERO dataset adapter

**Files:**
- Create: `starVLA/dataloader/cot_v3_lerobot_datasets.py`
- Modify: `starVLA/dataloader/__init__.py`
- Create: `tests/test_cot_v3_lerobot_dataset.py`

**Interfaces:**
- Consumes: `load_gripper_triangle_sidecar` and existing `CoTLeRobotSingleDataset`.
- Produces: `LiberoGripperTriangleCoTLeRobotSingleDataset`, V3 `get_vla_dataset`, unchanged list-based `collate_fn`.

- [ ] **Step 1: Write failing sidecar loader tests**

Construct a temporary episode DataFrame, depth/camera files, and a literal sidecar.
Instantiate an uninitialized dataset fixture with its cache/current trajectory fields and
assert `_load_episode_geometry` returns depth, `[T,3,3]` UVD, `[T,3]` in-frame validity,
and unchanged state. Assert missing sidecars and frame mismatches fail loudly.

- [ ] **Step 2: Write failing sampled-target test**

Call inherited `_geometry_targets` for a short literal trajectory and assert:

```text
uvd.shape == (K, 3, 3)
uvd_valid_mask.shape == (K, 3)
uvd_frame_indices are real unique episode indices
uvd_time.shape == (K,)
```

Assert off-screen points are invalid rather than boundary-clamped valid targets.
Also assert `uvd_landmark_ids.shape == (K, 3)` and that every sampled row is exactly
`[0, 1, 2]` in the approved landmark order.

- [ ] **Step 3: Write failing factory-registration test**

Patch dataset construction and assert `dataset_py: cot_v3_lerobot_datasets` selects only
the new adapter for the same four ordered LIBERO roots. Assert the existing
`cot_lerobot_datasets` branch remains unchanged.

- [ ] **Step 4: Run Task 3 tests and verify RED**

```bash
/root/data/yxz/miniforge3/envs/CoT/bin/python -m pytest tests/test_cot_v3_lerobot_dataset.py -v
```

Expected: missing V3 dataset module/registry branch.

- [ ] **Step 5: Implement the V3 adapter and factory**

Override `_load_episode_geometry` and the narrow `_geometry_targets` result augmentation.
Resolve the deterministic path from episode ID, derive width/height from depth, validate
the sidecar, and cache the tuple expected by the inherited sampler. Add
`uvd_landmark_ids` without changing inherited sampling semantics. Copy the established
LIBERO modality config/four-suite mixture boundary without changing the V2 factory.

- [ ] **Step 6: Run Task 3 plus V2 dataset regression tests**

```bash
/root/data/yxz/miniforge3/envs/CoT/bin/python -m pytest \
  tests/test_cot_v3_lerobot_dataset.py tests/test_cot_geometry.py tests/test_paired_geometry_probe.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit Task 3**

Commit the V3 dataset module, dataloader dispatch addition, and its test only.

### Task 4: Implement time-major landmark layout, packing, and losses

**Files:**
- Create: `starVLA/model/modules/geometric_cot_v3.py`
- Modify: `starVLA/model/modules/cot_losses.py`
- Create: `tests/test_geometric_cot_v3.py`
- Modify: `tests/test_cot_losses.py`

**Interfaces:**
- Produces: `LandmarkGeometryTokenLayout`, `LandmarkGeometryTokenEmbedding`, `PackedLandmarkUVDTargets`, `build_time_major_landmark_ids`, `pack_landmark_uvd_targets_time_major`, `build_landmark_geometry_full_attention_mask`, `uvd_triangle_shape_loss`.
- Consumers: Task 5.

- [ ] **Step 1: Write failing layout/packing tests**

For `K=2`, `landmark_count=3`, assert six UVD slots and flattened literal target:

```text
[L0,R0,W0,L1,R1,W1]
```

Assert landmark IDs are `[0,1,2,0,1,2]` and times are
`[t0,t0,t0,t1,t1,t1]`. Reject any input not shaped `[T,3,3]` / `[T,3]` or with a
landmark count different from the fixed layout.

- [ ] **Step 2: Write failing embedding/attention tests**

Zero trajectory/time parameters, assign literal landmark embedding rows, and assert
equal-time L/R/W queries remain distinct. For the full attention mask, assert L/R/W at
the same time read one another, later time reads earlier groups, and earlier time cannot
read later groups.

- [ ] **Step 3: Write failing temporal and shape-loss tests**

Retain the existing temporal loss but test it with three landmarks and a mutation where
only `R1` moves; only the R trajectory delta must contribute. For shape loss, assert:

- identical triangles and a common translation give zero;
- changing finger separation gives positive loss;
- changing wrist offset gives positive loss;
- any time lacking one valid landmark is excluded;
- no valid triangle returns a differentiable zero.

- [ ] **Step 4: Run focused tests and verify RED**

```bash
/root/data/yxz/miniforge3/envs/CoT/bin/python -m pytest tests/test_geometric_cot_v3.py tests/test_cot_losses.py -v
```

Expected: missing V3 module and shape loss.

- [ ] **Step 5: Implement V3 geometry primitives**

Use explicit fields `uvd_time_points` and `landmark_count`; do not expose a public
`hand_count` name. Reuse V2 depth pooling/slot append helpers where their contract is
landmark-agnostic. Implement the same-time attention grouping with integer division by
`landmark_count`.

- [ ] **Step 6: Implement the shape loss**

Reshape flattened `[B,K*3,3]` tensors to `[B,K,3,3]`, form grasp/wrist axes, require all
three masks at a time, apply coordinate-weighted Smooth-L1, and return a differentiable
zero through `pred.sum() * 0.0` if no triangle is valid.

- [ ] **Step 7: Run focused and V2 geometry/loss regressions**

```bash
/root/data/yxz/miniforge3/envs/CoT/bin/python -m pytest \
  tests/test_geometric_cot_v3.py tests/test_cot_losses.py tests/test_geometric_cot_v2.py -v
```

Expected: all pass.

- [ ] **Step 8: Commit Task 4**

Commit the V3 geometry module, shape-loss addition, and focused tests only.

### Task 5: Register the independent QwenGR00TCoTV3 framework

**Files:**
- Create: `starVLA/model/framework/VLM4A/QwenGR00TCoTV3.py`
- Create: `tests/test_qwen_gr00t_cot_v3.py`

**Interfaces:**
- Consumes: Task 4 V3 layout/packing/embedding/mask and shape loss; inherits stable V2 native-Qwen/action/depth methods.
- Produces: registry key `QwenGR00TCoTV3`, class `Qwen_GR00T_CoT_V3`, V3 loss dictionary keys `uvd_absolute_loss`, `uvd_temporal_loss`, `uvd_shape_loss`.

- [ ] **Step 1: Write failing registry/config tests**

Assert independent registry identity, required `landmark_count == 3`, four times produce
12 UVD slots, `uvd_token_order == "time_major"`, and non-boolean action-condition flags
retain the V2 validation behavior.

- [ ] **Step 2: Write failing target/loss aggregation tests**

Use an uninitialized V3 model fixture. Assert target packing calls the V3 packer and the
loss formula exactly equals:

```python
absolute + lambda_uvd_temporal * temporal + lambda_uvd_shape * shape
```

With `lambda_uvd_shape=0.0`, mutate only the returned shape loss and assert UVD total is
unchanged. With a positive weight, assert it changes by the exact weighted amount.

- [ ] **Step 3: Write failing V3 train-output test**

Patch expensive backbone/depth/action operations with complete real-shaped tensors and
assert the training result logs all three V3 UVD components while retaining the existing
action/depth/total fields. Do not assert mock call existence; assert returned tensor
values and shapes.

- [ ] **Step 4: Run Task 5 tests and verify RED**

```bash
/root/data/yxz/miniforge3/envs/CoT/bin/python -m pytest tests/test_qwen_gr00t_cot_v3.py -v
```

Expected: missing V3 framework.

- [ ] **Step 5: Implement V3 as a narrow V2 subclass**

Call the V2 initializer for Qwen/action/depth setup, then replace only the geometry
layout/embedding with V3 equivalents and set V3 weights. Override target packing,
geometry full-mask construction, checkpoint validation, UVD loss computation, and the
training result assembly required to expose temporal/shape logs. Do not edit
`QwenGR00TCoTV2.py`.

- [ ] **Step 6: Run V3 and V2 framework regressions**

```bash
/root/data/yxz/miniforge3/envs/CoT/bin/python -m pytest \
  tests/test_qwen_gr00t_cot_v3.py tests/test_qwen_gr00t_cot_v2.py \
  tests/test_geometric_cot_v3.py tests/test_geometric_cot_v2.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit Task 5**

Commit only the new V3 framework/test and any Task 4 files adjusted by real integration.

### Task 6: Add trainer, YAML, and one-click launcher

**Files:**
- Create: `starVLA/training/train_starvla_cot_v3.py`
- Create: `examples/modelExtensions/CoT/configs/qwen35_gr00t_libero_CoT_v3_q0_depthcond.yaml`
- Create: `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_CoT_v3_common.sh`
- Create: `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_libero_CoT_v3.sh`
- Create: `tests/test_cot_v3_entrypoints.py`

**Interfaces:**
- Produces: `CotV3Trainer`, executable trainer CLI, exact initial V3 config, dry-runnable launcher.

- [ ] **Step 1: Write failing config tests**

Load the YAML and assert framework `QwenGR00TCoTV3`, dataset
`cot_v3_lerobot_datasets`, `uvd_num_points=4`, `landmark_count=3`,
`lambda_uvd_temporal=0.1`, `lambda_uvd_shape=0.0`, depth action conditioning enabled,
LIBERO action/state dimensions unchanged, and no RoboCasa settings.

- [ ] **Step 2: Write failing CLI/launcher tests**

Assert trainer `--help` exits zero. Run the bench launcher with `DRY_RUN=1`, a literal
port/process override, and an extra dotlist argument; assert output names the V3 trainer,
V3 YAML, exact run ID, overrides, and forwarded argument.

- [ ] **Step 3: Run Task 6 tests and verify RED**

```bash
/root/data/yxz/miniforge3/envs/CoT/bin/python -m pytest tests/test_cot_v3_entrypoints.py -v
```

Expected: missing trainer/YAML/scripts.

- [ ] **Step 4: Implement trainer and launchers**

Mirror the established V2 lifecycle and distributed command, changing only V3 names,
paths, and required config message. Make both shell scripts executable. Base the YAML on
the current LIBERO q0 depth-condition YAML but change only the fields enumerated in the
approved spec plus a new run ID.

- [ ] **Step 5: Run entrypoint and existing V2 entrypoint tests**

```bash
/root/data/yxz/miniforge3/envs/CoT/bin/python -m pytest \
  tests/test_cot_v3_entrypoints.py tests/test_cot_v2_entrypoints.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit Task 6**

Commit the V3 trainer/config/scripts/test only.

### Task 7: Add an end-to-end CPU contract test and compatibility audit

**Files:**
- Create: `tests/test_libero_cot_v3_contract.py`
- Modify only if a verified integration defect requires it: V3 files from Tasks 1-6.

**Interfaces:**
- Consumes: all V3 data/model interfaces.
- Produces: one fixture proving loader output, packing, embeddings, temporal loss, and disabled shape-weight semantics agree end to end.

- [ ] **Step 1: Write the end-to-end fixture test**

Use a temporary sidecar with two times and literal L/R/W UVD values. Feed the loader's
sample dictionary to the V3 packer and an uninitialized V3 loss fixture. Assert exact
time-major flattened values, IDs/times, valid masks, output slot count, and total UVD
formula with zero shape weight.

- [ ] **Step 2: Run the contract test and observe any integration failure**

```bash
/root/data/yxz/miniforge3/envs/CoT/bin/python -m pytest tests/test_libero_cot_v3_contract.py -v
```

Expected before final integration adjustments: fail only on a real interface mismatch;
if it passes immediately, retain it as a cross-module contract test because each lower
layer already completed a RED cycle.

- [ ] **Step 3: Make only required integration adjustments**

Correct mismatched names/shapes at the narrowest V3 boundary. Do not relax assertions or
modify V2 behavior to make the contract pass.

- [ ] **Step 4: Run the complete focused suite**

```bash
/root/data/yxz/miniforge3/envs/CoT/bin/python -m pytest \
  tests/test_libero_gripper_triangle_sidecar.py \
  tests/test_cot_v3_lerobot_dataset.py \
  tests/test_geometric_cot_v3.py \
  tests/test_qwen_gr00t_cot_v3.py \
  tests/test_cot_v3_entrypoints.py \
  tests/test_libero_cot_v3_contract.py \
  tests/test_cot_geometry.py tests/test_cot_losses.py \
  tests/test_geometric_cot_v2.py tests/test_qwen_gr00t_cot_v2.py \
  tests/test_cot_v2_entrypoints.py -v
```

Expected: zero failures.

- [ ] **Step 5: Compile every new production module and shell-check launchers**

Run `py_compile` on all V3 Python files, `bash -n` on both launchers, and the real
`DRY_RUN=1` launcher command. Expected: exit zero and V3-only command paths.

- [ ] **Step 6: Audit old sample compatibility**

Materialize fixed V2 samples from one real episode before adding metadata/sidecars and
again afterward; compare keys, shapes, dtypes, and array values exactly. Record the
result in the generation report.

- [ ] **Step 7: Commit Task 7**

Commit the contract test and only verified V3 integration adjustments.

### Task 8: Generate and validate all four LIBERO suites

**Files/data:**
- Create in existing dataset roots: `geometry/gripper_triangle/chunk-*/episode_*.npz`
- Modify atomically: four `meta/info.json` files
- Create: `artifacts/libero_gripper_triangle_v3/sidecar_generation_report.json`
- Create: `artifacts/libero_gripper_triangle_v3/recovery_validation_report.json`

**Interfaces:**
- Consumes: Task 2 generator and all existing rerender HDF5 mappings/cameras.
- Produces: 1,693 validated episode sidecars covering 273,465 LIBERO frames.

- [ ] **Step 1: Generate one suite shard and validate immediately**

Run a ten-episode `libero_goal` shard, then `--validate-only` for the same range. Inspect
one NPZ and compare its projected L/R/W points with the prior overlay manifest/video.

- [ ] **Step 2: Run one real V3 dataset sample smoke test**

Instantiate the V3 loader against the generated suite, request fixed episode/frame
indices, and assert `[4,3,3]` UVD plus correct time-major packing. Also materialize the
same indices with the V2 loader and confirm its single-track output remains unchanged.

- [ ] **Step 3: Generate all suites resumably**

Use the LIBERO simulator environment with local checkout on `PYTHONPATH`. Process all
four suite roots, skip the already valid shard, and write the generation report under
the repository artifact directory.

- [ ] **Step 4: Validate complete coverage**

Run `--validate-only` over all suites. Require exactly 1,693 sidecars and 273,465 frames,
no missing/corrupt/schema-invalid files, positive minimum finger distance, positive
minimum triangle area, and finite positive-depth projections. Report out-of-frame points
without treating them as corrupt.

- [ ] **Step 5: Compute aperture and orientation recovery evidence**

For every or a documented deterministic large sample of frames:

- fit the monotonic/linear relationship between finger distance and the stored two-value
  gripper qpos;
- construct a triangle frame from `R-L` and midpoint-to-wrist;
- fit one constant triangle-to-EEF rotation offset on a calibration split;
- report held-out geodesic rotation error and derived RPY error distributions.

Do not make these empirical metrics generation pass conditions unless geometry is
degenerate; record them as evidence for the representation claim.

- [ ] **Step 6: Run final tests and launcher dry-run fresh**

Repeat Task 7's full focused suite, compile checks, shell checks, complete sidecar
validation, and V3 launcher dry-run. Capture exit-zero summaries.

- [ ] **Step 7: Final workspace audit**

Use `git status`, targeted diffs, and commit file lists to prove existing V2/user changes
were not staged or overwritten. Report dataset files modified, artifact reports, exact
commands, test totals, and recovery metrics.
