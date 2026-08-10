# Cadence-Aligned Training Geometry Videos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render every training episode frame while running v1/v2 geometry only at real control anchors, and add fixed episode-wide 3D UVD visualization.

**Architecture:** Separate render samples from inference samples with a pure cadence planner. Materialize lightweight display frames independently from complete-horizon anchor samples, cache one prediction per anchor, and bind every rendered frame to its active anchor. Replace horizon-zero episode prediction curves with timestamped sparse chunk segments and reuse the same immutable context for every frame.

**Tech Stack:** Python 3.10, NumPy, Matplotlib, PyAV/H.264, unittest, existing LeRobot rerender stores and CoT checkpoint loader.

## Global Constraints

- LIBERO action horizon and execution horizon are 8; render cadence is 1.
- RoboCasa action horizon is 16, execution horizon is 12; render cadence is 1.
- LIBERO UVD offsets are `[0, 3, 5, 8]`; RoboCasa offsets are `[0, 3, 6, 10, 13, 16]`.
- RoboCasa prediction segments after chunk step 12 are dashed.
- Subplot positions, axis limits, 3D view, legends, and color normalization remain fixed for the full episode.
- Existing dense preview videos and cached artifacts are not overwritten.
- No training-set point cloud is generated.
- Do not create a Git commit.

---

### Task 1: Pure render/inference cadence plan

**Files:**
- Modify: `examples/simBenchmarks/CoT/geometry_probe/episode_video.py`
- Create: `tests/test_episode_geometry_cadence.py`

**Interfaces:**
- Produces: `EpisodeCadencePlan`, containing `inference_samples`, `render_samples`, `anchor_index_by_render`, and `chunk_step_by_render`.
- Produces: `build_episode_cadence_plan(episodes, *, action_horizon, inference_stride, render_stride) -> EpisodeCadencePlan`.
- Consumes: existing `EpisodeRef` and `SampleRef`.

- [ ] **Step 1: Write failing LIBERO and RoboCasa cadence tests**

```python
def test_libero_renders_every_frame_but_infers_every_eight():
    plan = build_episode_cadence_plan(
        [EpisodeRef("libero_spatial", 7, 20)],
        action_horizon=8,
        inference_stride=8,
        render_stride=1,
    )
    assert [x.frame_index for x in plan.inference_samples] == [0, 8]
    assert [x.frame_index for x in plan.render_samples] == list(range(20))
    assert plan.anchor_index_by_render == (0,) * 8 + (1,) * 12
    assert plan.chunk_step_by_render[:10] == tuple(range(8)) + (0, 1)

def test_robocasa_replans_at_twelve_with_sixteen_step_horizon():
    plan = build_episode_cadence_plan(
        [EpisodeRef("task", 9, 30)],
        action_horizon=16,
        inference_stride=12,
        render_stride=1,
    )
    assert [x.frame_index for x in plan.inference_samples] == [0, 12]
    assert plan.anchor_index_by_render[11:14] == (0, 1, 1)
```

- [ ] **Step 2: Run the cadence tests and verify the missing-interface failure**

Run: `python -m unittest tests.test_episode_geometry_cadence -v`

Expected: import failure for `EpisodeCadencePlan` or `build_episode_cadence_plan`.

- [ ] **Step 3: Implement immutable cadence types and validation**

```python
@dataclass(frozen=True)
class EpisodeCadencePlan:
    inference_samples: tuple[SampleRef, ...]
    render_samples: tuple[SampleRef, ...]
    anchor_index_by_render: tuple[int, ...]
    chunk_step_by_render: tuple[int, ...]

def build_episode_cadence_plan(
    episodes: Sequence[EpisodeRef],
    *,
    action_horizon: int,
    inference_stride: int,
    render_stride: int,
) -> EpisodeCadencePlan:
    if min(action_horizon, inference_stride, render_stride) < 1:
        raise ValueError("horizon and strides must be positive")
    if inference_stride > action_horizon:
        raise ValueError("inference_stride cannot exceed action_horizon")
    inference, render, bindings, chunk_steps = [], [], [], []
    for episode in episodes:
        anchors = list(range(0, episode.episode_length - action_horizon, inference_stride))
        if not anchors:
            raise ValueError(f"episode has no complete anchor: {episode}")
        base = len(inference)
        inference.extend(
            SampleRef(episode.suite, episode.episode_id, episode.episode_length, frame)
            for frame in anchors
        )
        for frame in range(0, episode.episode_length, render_stride):
            local_anchor = max(index for index, anchor in enumerate(anchors) if anchor <= frame)
            render.append(SampleRef(episode.suite, episode.episode_id, episode.episode_length, frame))
            bindings.append(base + local_anchor)
            chunk_steps.append(frame - anchors[local_anchor])
    return EpisodeCadencePlan(tuple(inference), tuple(render), tuple(bindings), tuple(chunk_steps))
```

Validate positive values, require `inference_stride <= action_horizon`, create anchors only where `anchor + action_horizon < episode_length`, render the complete episode, and bind the tail to the most recent complete anchor.

- [ ] **Step 4: Run cadence and existing episode-plan tests**

Run: `python -m unittest tests.test_episode_geometry_cadence tests.test_episode_geometry_video -v`

Expected: all tests pass.

### Task 2: Lightweight complete-episode display frames

**Files:**
- Modify: `examples/simBenchmarks/CoT/geometry_probe/dataset_probe.py`
- Modify: `examples/simBenchmarks/CoT/geometry_probe/paired_probe.py`
- Create: `tests/test_episode_geometry_display_frames.py`

**Interfaces:**
- Produces: `LiberoRerenderStore.load_display_frame(ref) -> dict[str, Any]`.
- Produces: `RoboCasaRerenderStore.load_display_frame(ref) -> dict[str, Any]`.
- Produces: `materialize_display_frames(store, plan, output_dir) -> list[Path]` and `load_materialized_display_frame(path) -> dict[str, Any]`.
- The display bundle contains images, current depth/valid mask, horizon-zero GT UVD/valid mask, timestamp, frame index, and metadata; it contains no future target.

- [ ] **Step 1: Write failing tail-frame and schema tests**

Use synthetic store fixtures to assert that frame `episode_length - 1` materializes without requiring `frame + horizon`, and that loading returns exactly current-frame geometry.

- [ ] **Step 2: Run the new tests and verify `load_display_frame` is missing**

Run: `python -m unittest tests.test_episode_geometry_display_frames -v`

Expected: attribute/import failure.

- [ ] **Step 3: Factor shared episode-frame extraction and implement both stores**

Reuse each store's episode cache and video decoder. For GT UVD, select the exact current episode point rather than calling the horizon sampler. Resize only current depth. Preserve dual-hand layout as `[2, 3]` for RoboCasa and single-hand layout as `[3]` for LIBERO.

- [ ] **Step 4: Implement display materialization without changing anchor bundles**

```python
def materialize_display_frames(store, plan, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for sample_index, ref in enumerate(plan):
        path = output_dir / f"frame_{sample_index:06d}.npz"
        if not path.exists():
            sample = store.load_display_frame(ref)
            images = sample["images"]
            payload = {
                "image_count": np.asarray(len(images), dtype=np.int64),
                "depth_current": np.asarray(sample["depth_current"], dtype=np.float32),
                "depth_current_valid": np.asarray(sample["depth_current_valid"], dtype=np.bool_),
                "uvd_current": np.asarray(sample["uvd_current"], dtype=np.float32),
                "uvd_current_valid": np.asarray(sample["uvd_current_valid"], dtype=np.bool_),
                "metadata_json": np.asarray(json.dumps(sample["metadata"])),
            }
            payload.update({f"image_{i}": np.asarray(image, dtype=np.uint8) for i, image in enumerate(images)})
            np.savez_compressed(path, **payload)
        paths.append(path)
    return paths
```

- [ ] **Step 5: Run display, paired-probe, and dataset-reader tests**

Run: `python -m unittest tests.test_episode_geometry_display_frames tests.test_paired_geometry_probe -v`

Expected: all tests pass.

### Task 3: Sparse anchored episode UVD context

**Files:**
- Modify: `examples/simBenchmarks/CoT/geometry_probe/episode_curves.py`
- Replace expectations in: `tests/test_episode_geometry_curves.py`

**Interfaces:**
- Produces: immutable `UvdPredictionSegment(anchor_frame, frame_indices, values, valid, executed)`.
- Extends: `EpisodeCurveContext` with complete per-frame GT and `segments: dict[str, tuple[UvdPredictionSegment, ...]]`.
- Produces: `build_cadence_episode_curve_context(display_samples, anchor_samples, predictions_by_label, labels, *, execution_horizon) -> EpisodeCurveContext`.

- [ ] **Step 1: Replace horizon-zero stitching tests with sparse segment tests**

Assert that LIBERO creates segment frame indices `[0, 3, 5, 8]`, that RoboCasa marks offsets 13 and 16 as unexecuted for execution horizon 12, and that invalid GT points remain NaN gaps.

- [ ] **Step 2: Run curve tests and verify failure against the old horizon-zero context**

Run: `python -m unittest tests.test_episode_geometry_curves -v`

Expected: missing `segments` or wrong segment shapes.

- [ ] **Step 3: Implement segment canonicalization and episode-wide limits**

Use `uvd_frame_indices` from anchor samples as the source of absolute segment time. Never infer offsets from array length. Compute limits from complete GT plus all valid prediction segment points. Keep depth image/error limits based on anchor targets and predictions.

- [ ] **Step 4: Run curve tests**

Run: `python -m unittest tests.test_episode_geometry_curves -v`

Expected: all tests pass.

### Task 4: Fixed 3D UVD rendering

**Files:**
- Modify: `examples/simBenchmarks/CoT/geometry_probe/visualization.py`
- Modify: `tests/test_episode_geometry_fixed_layout.py`
- Modify: `tests/test_episode_geometry_video_render.py`

**Interfaces:**
- Consumes: `EpisodeCurveContext.segments`, full GT, active anchor, and chunk step.
- Produces: a fixed-position Matplotlib 3D UVD panel within `render_paired_episode_frame`.

- [ ] **Step 1: Write failing fixed-3D-axis tests**

Assert identical `get_position().bounds`, `get_xlim3d`, `get_ylim3d`, `get_zlim3d`, `elev`, and `azim` at the first and last video frames. Assert RoboCasa's unexecuted tail uses `--` and that the execution marker changes without changing line data.

- [ ] **Step 2: Run rendering tests and verify the missing 3D panel failure**

Run: `python -m unittest tests.test_episode_geometry_fixed_layout tests.test_episode_geometry_video_render -v`

Expected: no 3D axis or wrong fixed layout.

- [ ] **Step 3: Add the 3D axis to the existing deterministic GridSpec**

Plot `(u, v, d)` with fixed `view_init(elev=24, azim=-58)`. Keep checkpoint colors consistent with 2D panels. Use hand-specific marker/line style for RoboCasa and split every segment at the execution boundary into solid and dashed parts.

- [ ] **Step 4: Pass active anchor and chunk step through frame rendering**

Update `render_paired_episode_frame` and `_build_paired_episode_figure` signatures so the moving markers are explicit and do not derive inference cadence from video index.

- [ ] **Step 5: Run fixed-layout and H.264 render tests**

Run: `python -m unittest tests.test_episode_geometry_fixed_layout tests.test_episode_geometry_video_render tests.test_episode_geometry_h264 tests.test_episode_geometry_h264_fractional_fps -v`

Expected: all tests pass and frames remain even-sized RGB uint8.

### Task 5: Cadence-aligned runner and CLI

**Files:**
- Modify: `examples/simBenchmarks/CoT/geometry_probe/run_episode_geometry_videos.py`
- Modify: `tests/test_episode_geometry_video_cli.py`

**Interfaces:**
- New CLI: `--inference-stride`, `--render-stride`.
- Legacy CLI: `--stride N` sets both values, emits `DeprecationWarning`, and conflicts with either new option.
- Output metadata records action horizon, inference stride, render stride, UVD offsets, GT mode, and anchor/render source frames.

- [ ] **Step 1: Write failing CLI defaults and legacy-conflict tests**

Assert defaults `(8, 1)` for LIBERO and `(12, 1)` for RoboCasa after benchmark resolution. Assert `--stride 4` resolves to `(4, 4)` with a warning and that mixing it with `--render-stride` exits with an error.

- [ ] **Step 2: Write a failing runner binding test**

Mock two anchors and ten display frames. Assert checkpoint prediction is loaded twice, the renderer is called ten times, frames 0–7 receive anchor 0, and frames 8–9 receive anchor 1.

- [ ] **Step 3: Run CLI tests and verify failure**

Run: `python -m unittest tests.test_episode_geometry_video_cli -v`

Expected: missing new arguments and old one-to-one sample assumptions.

- [ ] **Step 4: Refactor main and `_render_episode_videos` around `EpisodeCadencePlan`**

Materialize/predict only `inference_samples`; materialize all `render_samples` with the lightweight path. Build anchor metrics once. Use bindings to select the active anchor prediction and metric for each display frame.

- [ ] **Step 5: Write explicit cadence-aligned output metadata**

Use a new default directory name such as `episode_cadence_aligned_i8_r1` or `episode_cadence_aligned_i12_r1`; never reuse `episode_stride1`.

- [ ] **Step 6: Run all geometry-video unit tests**

Run: `python -m unittest tests.test_episode_geometry_cadence tests.test_episode_geometry_display_frames tests.test_episode_geometry_curves tests.test_episode_geometry_fixed_layout tests.test_episode_geometry_video_cli tests.test_episode_geometry_video tests.test_episode_geometry_video_render tests.test_episode_geometry_h264 tests.test_episode_geometry_h264_fractional_fps tests.test_paired_geometry_probe -v`

Expected: all tests pass.

### Task 6: Regenerate and inspect the two agreed training videos

**Files:**
- No source changes expected.
- New artifacts under the existing 240-probe roots in new `episode_cadence_aligned_*` directories.

**Interfaces:**
- Consumes the previously selected LIBERO episode 256 and RoboCasa episode 990 manifests/checkpoints.
- Produces one H.264/yuv420p video for each benchmark plus run metadata.

- [ ] **Step 1: Run one LIBERO cadence-aligned preview**

Run the episode-video command with `--bench libero --inference-stride 8 --render-stride 1`, the existing fixed source manifest, and a new output directory.

- [ ] **Step 2: Run one RoboCasa cadence-aligned preview**

Run the same command with `--bench robocasa --inference-stride 12 --render-stride 1` and the selected task/episode scope.

- [ ] **Step 3: Inspect first, boundary, middle, and final frames**

Create contact sheets in `/tmp`, visually verify fixed axes/layout, prediction reuse within chunks, marker transitions at 8/12, and RoboCasa's dashed 12–16 tail.

- [ ] **Step 4: Probe both videos**

Verify H.264, yuv420p, even dimensions, expected full episode frame counts, and source-derived FPS with OpenCV/PyAV.

- [ ] **Step 5: Run the complete existing geometry regression suite**

Run: `python -m unittest tests.test_geometry_only_checkpoint tests.test_episode_geometry_cadence tests.test_episode_geometry_display_frames tests.test_episode_geometry_curves tests.test_episode_geometry_fixed_layout tests.test_episode_geometry_h264 tests.test_episode_geometry_h264_fractional_fps tests.test_episode_geometry_video_cli tests.test_episode_geometry_video tests.test_episode_geometry_video_render tests.test_paired_geometry_probe -v`

Expected: all tests pass.
