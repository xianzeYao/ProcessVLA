# Online Evaluation Depth and Point-Cloud Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** During real LIBERO and RoboCasa policy rollouts, compare predicted current/future depth with simulator metric depth and render predicted UVD in an online real RGB-D point cloud.

**Architecture:** Put delayed future-depth accounting and camera-space point-cloud math in dependency-light pure modules. Add optional geometry-return paths to existing evaluation clients without changing default evaluation behavior. Benchmark-specific adapters capture simulator depth and calibration, while a shared renderer creates per-checkpoint videos; v1/v1.5 and v2 rollouts remain separate after their actions diverge.

**Tech Stack:** Python 3.10, NumPy, Matplotlib, MuJoCo/robosuite camera utilities, LIBERO, RoboCasa Gym, websocket policy server, PyAV/H.264, unittest.

## Global Constraints

- Online metrics cover masked `depth_current` and `depth_future`; UVD is qualitative only.
- LIBERO future depth is observed at anchor `t + 8`.
- RoboCasa future depth is observed at anchor `t + 16`, after normal replanning at `t + 12`.
- Incomplete future targets caused by early termination are excluded from future aggregates but retain current metrics.
- Point clouds use only online simulator RGB-D and matching intrinsics; no training point cloud is loaded.
- Point clouds are rebuilt per frame and never fused across time.
- v1/v1.5 and v2 run from the same task/seed but produce separate rollout artifacts and GT.
- Existing standard evaluation behavior is unchanged unless geometry-probe flags are explicitly enabled.
- Do not create a Git commit.

---

### Task 1: Delayed online depth accounting

**Files:**
- Create: `examples/simBenchmarks/CoT/geometry_probe/online_depth.py`
- Create: `tests/test_online_geometry_depth.py`

**Interfaces:**
- Produces: `PendingDepthAnchor` with frame, target frame, predicted current/future depth, current GT/validity, metadata, and optional completed future GT.
- Produces: `OnlineDepthRecorder(action_horizon: int)`.
- Produces methods: `add_anchor(frame_index: int, prediction: Mapping[str, np.ndarray], gt_current: np.ndarray, valid_current: np.ndarray, metadata: Mapping[str, Any] | None = None) -> PendingDepthAnchor`, `observe(frame_index: int, depth: np.ndarray, valid: np.ndarray) -> None`, `finalize(end_frame: int) -> None`, `records() -> tuple[PendingDepthAnchor, ...]`, and `summary() -> dict[str, Any]`.

- [ ] **Step 1: Write failing immediate-current and delayed-future tests**

```python
def test_future_depth_attaches_only_at_target_frame():
    recorder = OnlineDepthRecorder(action_horizon=8)
    recorder.add_anchor(frame_index=4, prediction=prediction, gt_current=gt4, valid_current=mask4)
    recorder.observe(11, gt11, mask11)
    assert recorder.records()[0].future_complete is False
    recorder.observe(12, gt12, mask12)
    assert recorder.records()[0].future_complete is True
    np.testing.assert_array_equal(recorder.records()[0].gt_future, gt12)
```

Also assert early termination excludes future metrics and that current/future valid counts are distinct.

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `python -m unittest tests.test_online_geometry_depth -v`

Expected: import failure for `online_depth`.

- [ ] **Step 3: Implement immutable records and metric reuse**

Use existing `masked_depth_metrics` for both targets. Store arrays as copied float32/bool values so later simulator mutation cannot change records. Reject duplicate anchors and non-monotonic observations.

- [ ] **Step 4: Run depth-accounting tests**

Run: `python -m unittest tests.test_online_geometry_depth -v`

Expected: all tests pass.

### Task 2: Camera-space point-cloud math

**Files:**
- Create: `examples/simBenchmarks/CoT/geometry_probe/pointcloud.py`
- Create: `tests/test_online_geometry_pointcloud.py`

**Interfaces:**
- Produces: `backproject_rgbd(rgb, depth, valid, intrinsic, *, max_points, seed) -> tuple[xyz, colors]`.
- Produces: `model_uvd_to_camera_xyz(uvd, intrinsic, *, image_size, depth_scale=1.0) -> np.ndarray`.
- Produces: `PointCloudLimits` and `pointcloud_limits(frames) -> PointCloudLimits`.

- [ ] **Step 1: Write failing synthetic pinhole tests**

Use `K=[[2,0,1],[0,2,1],[0,0,1]]`, known depths, and expected `x=(u-cx)z/fx`, `y=(v-cy)z/fy`. Assert normalized model UVD at `(0.5, 0.5, 2)` maps to the optical axis for a matching resized intrinsic.

- [ ] **Step 2: Write invalid-depth and deterministic-subsampling tests**

Assert NaN, zero, negative, and masked pixels are removed; the same seed yields identical points; output never exceeds `max_points`.

- [ ] **Step 3: Run tests and verify missing functions**

Run: `python -m unittest tests.test_online_geometry_pointcloud -v`

Expected: import failure.

- [ ] **Step 4: Implement vectorized back-projection without Open3D**

Use NumPy meshgrid, inverse pinhole equations, RGB float colors in `[0,1]`, and seeded index selection. Validate shapes, finite nonsingular intrinsics, and positive `max_points`.

- [ ] **Step 5: Run point-cloud tests**

Run: `python -m unittest tests.test_online_geometry_pointcloud -v`

Expected: all tests pass.

### Task 3: Shared online geometry artifact renderer

**Files:**
- Create: `examples/simBenchmarks/CoT/geometry_probe/online_visualization.py`
- Create: `tests/test_online_geometry_visualization.py`

**Interfaces:**
- Consumes: per-frame RGB-D/K, the active `PendingDepthAnchor`, and predicted UVD.
- Produces: `render_online_geometry_frame(frame: Mapping[str, Any], anchor: PendingDepthAnchor, intrinsic: np.ndarray, xyz_limits: PointCloudLimits) -> np.ndarray` and `write_online_rollout_artifacts(frames: Sequence[Mapping[str, Any]], recorder: OnlineDepthRecorder, output_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]`.

- [ ] **Step 1: Write failing fixed-layout and RGB-frame tests**

Assert even-sized uint8 RGB output, one 3D axis, fixed XYZ/depth/error limits across two frames, and moving point-cloud/UVD artists without subplot movement.

- [ ] **Step 2: Run visualization tests and verify missing renderer**

Run: `python -m unittest tests.test_online_geometry_visualization -v`

Expected: import failure.

- [ ] **Step 3: Implement the online layout**

Include current RGB, real current depth, predicted current depth/error, delayed real future depth, predicted future depth/error, and an RGB point cloud with predicted UVD. Show `future pending` until the target frame is recorded. Use `view_init(elev=24, azim=-58)` and episode-fixed limits.

- [ ] **Step 4: Implement JSONL/summary/MP4 writing**

Use existing `write_video_frames` for H.264. Write per-anchor status and metrics plus config metadata: checkpoint, task, seed, action horizon, execution horizon, frame count, and depth GT mode.

- [ ] **Step 5: Run visualization and H.264 tests**

Run: `python -m unittest tests.test_online_geometry_visualization tests.test_episode_geometry_h264 tests.test_episode_geometry_h264_fractional_fps -v`

Expected: all tests pass.

### Task 4: Optional geometry return in evaluation clients

**Files:**
- Modify: `examples/simBenchmarks/LIBERO/eval_files/model2libero_interface.py`
- Modify: `examples/simBenchmarks/Robocasa_tabletop/eval_files/model2robocasa_interface.py`
- Create: `tests/test_eval_geometry_clients.py`

**Interfaces:**
- LIBERO: `ModelClient` adds keyword `return_geometry: bool = False`; `step()` includes `geometry_anchor` only when a chunk is refreshed.
- RoboCasa: `PolicyWarper` adds keyword `return_geometry: bool = False`; `step()` includes `geometry_anchor` alongside actions for probe callers.
- Default `False` preserves existing response dictionaries and server requests.

- [ ] **Step 1: Write failing mock-websocket tests**

Assert geometry-enabled clients add `return_geometry=True`, expose geometry only on inference anchors, and cache actions exactly as before. Assert disabled clients make the original request and return the original schema.

- [ ] **Step 2: Run client tests and verify missing option**

Run: `python -m unittest tests.test_eval_geometry_clients -v`

Expected: constructor/signature failure.

- [ ] **Step 3: Implement opt-in request and canonical geometry squeezing**

Keep raw geometry arrays in model space and include server timing. Do not rerun the websocket request on intermediate LIBERO action steps.

- [ ] **Step 4: Run client and existing entry-point tests**

Run: `python -m unittest tests.test_eval_geometry_clients tests.test_cot_v2_entrypoints -v`

Expected: all tests pass.

### Task 5: LIBERO real-evaluation adapter

**Files:**
- Create: `examples/simBenchmarks/CoT/geometry_probe/run_libero_online_depth_probe.py`
- Create: `tests/test_libero_online_depth_probe.py`

**Interfaces:**
- CLI accepts host, port, suite, task id, episode index, seed, unnorm key, output directory, and max steps.
- Produces one checkpoint-controlled rollout with per-step online RGB-D/K, anchor geometry, metrics, JSONL, summary, and MP4.

- [ ] **Step 1: Write failing observation conversion tests**

Mock normalized and metric `agentview_depth`; assert conversion to meters, vertical orientation matching RGB, resize to model space, validity mask, and resized intrinsic consistency.

- [ ] **Step 2: Write a failing mocked rollout cadence test**

Assert one geometry request at steps 0 and 8, real depth capture at every step, future target completion at step 8, and no UVD metric key in the summary.

- [ ] **Step 3: Run tests and verify missing runner**

Run: `python -m unittest tests.test_libero_online_depth_probe -v`

Expected: import failure.

- [ ] **Step 4: Implement the probe using official initial states**

Construct `OffScreenRenderEnv` with agentview/wrist RGB and `camera_depths=True`, use `ModelClient(return_geometry=True)`, execute normal cached actions, and query current camera calibration at capture time. Do not use training HDF5 states.

- [ ] **Step 5: Finalize delayed targets and artifacts on success/failure/timeout**

Close client/environment in `finally`; incomplete targets remain explicit rather than being filled with the last frame.

- [ ] **Step 6: Run LIBERO adapter tests**

Run: `python -m unittest tests.test_libero_online_depth_probe tests.test_online_geometry_depth tests.test_online_geometry_pointcloud tests.test_online_geometry_visualization -v`

Expected: all tests pass.

### Task 6: RoboCasa real-evaluation adapter with low-level depth capture

**Files:**
- Modify: `examples/simBenchmarks/Robocasa_tabletop/eval_files/wrappers/multistep_wrapper.py`
- Modify: `examples/simBenchmarks/Robocasa_tabletop/eval_files/simulation_env.py`
- Create: `examples/simBenchmarks/CoT/geometry_probe/run_robocasa_online_depth_probe.py`
- Create: `tests/test_robocasa_online_depth_probe.py`

**Interfaces:**
- Adds an opt-in low-level observation hook to `MultiStepWrapper`; default `None` has no overhead/schema change.
- The probe uses `n_action_steps=12`, captures each of the 12 internal simulator states, and can therefore attach a `t+16` target four low-level steps into the next action chunk.

- [ ] **Step 1: Write failing multistep hook tests**

With a fake base env and three actions, assert the hook receives base env/observation at low-level frames 1, 2, and 3 in order. Assert no calls and unchanged wrapper result when the hook is `None`.

- [ ] **Step 2: Write failing RoboCasa RGB-D/K conversion tests**

Assert raw 1280x800 egoview RGB/depth receives the existing DIAL transform, metric depth conversion, and `transformed_agentview_intrinsic`; assert output RGB/depth/K share 224x224 coordinates.

- [ ] **Step 3: Write a mocked 12/16 cadence test**

Assert inference anchors at low-level frames 0, 12, and 24, while the first future-depth GT attaches at frame 16 rather than 12 or 24.

- [ ] **Step 4: Run tests and verify missing hook/runner failures**

Run: `python -m unittest tests.test_robocasa_online_depth_probe -v`

Expected: missing hook or adapter failure.

- [ ] **Step 5: Implement the opt-in low-level hook**

Invoke it immediately after each underlying `super().step(act)`, before the observation is replaced by the next internal step. Keep standard vector-evaluation return values unchanged.

- [ ] **Step 6: Implement the one-environment probe**

Use the existing RoboCasa environment/task naming, `PolicyWarper(return_geometry=True, n_action_steps=12)`, transformed egoview RGB-D/K helpers, and shared recorder/renderer. Access the base simulator only in the opt-in probe path.

- [ ] **Step 7: Run RoboCasa adapter and standard wrapper tests**

Run: `python -m unittest tests.test_robocasa_online_depth_probe tests.test_eval_geometry_clients -v`

Expected: all tests pass.

### Task 7: One-episode online runtime validation

**Files:**
- No source changes expected.
- New artifacts in separate `online_depth_probe/{benchmark}/{checkpoint_label}/{task}_{seed}` directories.

**Interfaces:**
- Produces one LIBERO and one RoboCasa online rollout per available checkpoint label, without overwriting standard benchmark videos.

- [ ] **Step 1: Start the geometry-enabled policy server for the selected LIBERO checkpoint**

Use `GeometryProbePolicyWrapper`; verify metadata reports `supports_geometry=true` and action chunk size 8.

- [ ] **Step 2: Run one LIBERO task/seed and inspect depth units**

Assert median valid simulator depth and model depth are finite and in meters, current/future target indices are 8 apart, and the video opens as H.264/yuv420p.

- [ ] **Step 3: Repeat for the second LIBERO checkpoint if both servers are available**

Use the same suite/task/episode/seed and a separate output directory.

- [ ] **Step 4: Start the geometry-enabled server and RoboCasa environment for one selected task**

Verify action horizon 16 and execution horizon 12 in run metadata.

- [ ] **Step 5: Run one RoboCasa task/seed and inspect t+16 attachment**

Confirm low-level frames are captured continuously across the t+12 replan and that future depth for anchor 0 is sourced from frame 16.

- [ ] **Step 6: Inspect point-cloud video frames**

Check first, inference-boundary, future-target, middle, and final frames for fixed axes, correctly oriented RGB point clouds, and predicted UVD placement.

- [ ] **Step 7: Run the complete combined regression suite**

Run: `python -m unittest tests.test_online_geometry_depth tests.test_online_geometry_pointcloud tests.test_online_geometry_visualization tests.test_eval_geometry_clients tests.test_libero_online_depth_probe tests.test_robocasa_online_depth_probe tests.test_cot_v2_entrypoints tests.test_qwen_gr00t_cot_v2 -v`

Expected: all tests pass. If an external simulator package or EGL runtime is unavailable, unit tests must still pass and the exact runtime dependency failure is recorded without claiming the corresponding rollout succeeded.
