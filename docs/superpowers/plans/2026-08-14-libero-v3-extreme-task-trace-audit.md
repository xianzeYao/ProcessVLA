# LIBERO V3 Extreme-Task Trace Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Select two best and two worst tasks from every completed standard LIBERO V3 suite, rerun five deterministic seeds per selected task, and generate synchronized action/LRW-UVD/depth-error videos and summaries.

**Architecture:** Extend the existing action inference path with an opt-in geometry payload that reuses the same V3 backbone forward. Keep task selection, trace alignment/metrics, simulator collection, and rendering in focused modules. Cache one raw rollout record before rendering so all 80 artifacts are resumable and re-renderable without checkpoint inference.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, MuJoCo/robosuite/LIBERO, matplotlib, imageio/ffmpeg, pytest, Bash.

## Global Constraints

- Do not change training, loss weights, checkpoint weights, or the default action-only response.
- Rank from the completed standard LIBERO V3 50-trial worker logs.
- Select two best and two worst tasks per suite with task-id tie breaking.
- Use seeds 7,8,9,10,11 on one fixed representative initial-state index per task.
- Preserve every rerun, including outcome mismatches; never search for a more attractive episode.
- Treat simulator-realized LRW as rollout self-consistency ground truth, not expert demonstration.
- Use time-major UVD with left/right/wrist landmark order.
- Do not stop or mutate the active LIBERO-Plus evaluation.
- Write exactly 80 rollout MP4s, 80 per-rollout PNGs, 16 five-seed summaries, and four suite contact sheets.
- Commit each independently testable task separately and never include existing untracked artifacts.

---

### Task 1: Opt-in one-forward action and V3 geometry response

**Files:**
- Modify: starVLA/model/framework/VLM4A/QwenGR00TCoTV2.py
- Modify: deployment/model_server/policy_wrapper.py
- Modify: tests/test_qwen_gr00t_cot_v3.py
- Create: tests/test_policy_wrapper_geometry_response.py

**Interfaces:**
- Consumes: Qwen_GR00T_CoT_V2.predict_action(examples, return_geometry=False, timing_callback=None)
- Produces: optional result["geometry"] with depth_current, depth_future, uvd, uvd_time, and uvd_landmark_ids
- Produces: PolicyServerWrapper.predict_action(..., inference_seed=None) with scoped deterministic Torch RNG

- [ ] **Step 1: Write failing framework tests**

Add a mocked V3 instance test that counts _run_geometry_backbone calls and requests return_geometry=True:

~~~python
result = model.predict_action([example], return_geometry=True)
assert backbone_calls == 1
assert set(result["geometry"]) == {
    "depth_current", "depth_future", "uvd", "uvd_time", "uvd_landmark_ids"
}
assert result["geometry"]["uvd"].shape == (1, 12, 3)
assert result["geometry"]["uvd_time"].shape == (1, 12)
assert result["geometry"]["uvd_landmark_ids"].tolist()[0] == [0, 1, 2] * 4
~~~

Also call without return_geometry and assert the result keys remain exactly normalized_actions.

- [ ] **Step 2: Run tests and verify failure**

Run:

~~~bash
pytest -q tests/test_qwen_gr00t_cot_v3.py -k return_geometry
~~~

Expected: FAIL because predict_action does not return geometry.

- [ ] **Step 3: Implement the shared-forward payload**

In QwenGR00TCoTV2.predict_action, pop return_geometry before invoking the action expert. Reuse split and qwen_inputs already computed for the action. When requested, call _decode_geometry once and construct time/landmark arrays from geometry_layout for V3. Do not decode geometry on action-only requests.

- [ ] **Step 4: Add deterministic wrapper tests**

Create a minimal fake framework whose predict_action records torch.rand output. Assert identical inference_seed values return identical actions, different seeds differ, and omitting inference_seed preserves the existing path. Assert geometry arrays are forwarded through PolicyServerWrapper._to_numpy.

- [ ] **Step 5: Implement scoped inference seeding**

Pop inference_seed in PolicyServerWrapper.predict_action. When provided, enter torch.random.fork_rng for the framework device, call torch.manual_seed with the request seed, then invoke the framework. Do not alter global RNG state outside the request.

- [ ] **Step 6: Run focused regression tests**

Run:

~~~bash
pytest -q tests/test_qwen_gr00t_cot_v2.py tests/test_qwen_gr00t_cot_v3.py tests/test_policy_wrapper_geometry_response.py
~~~

Expected: all pass.

- [ ] **Step 7: Commit**

~~~bash
git add starVLA/model/framework/VLM4A/QwenGR00TCoTV2.py deployment/model_server/policy_wrapper.py tests/test_qwen_gr00t_cot_v3.py tests/test_policy_wrapper_geometry_response.py
git commit -m "feat: expose CoT geometry with policy actions"
~~~

### Task 2: Parse standard evaluation logs and select extreme tasks

**Files:**
- Create: examples/simBenchmarks/CoT/geometry_probe/libero_trace_audit_selection.py
- Create: tests/test_libero_trace_audit_selection.py

**Interfaces:**
- Produces: TaskEvaluation(suite, task_id, language, outcomes)
- Produces: AuditCase(suite, task_id, language, rank_group, initial_state_index, seed, original_success)
- Produces: parse_worker_log(path, suite) -> list[TaskEvaluation]
- Produces: select_extreme_tasks(rows, count=2) -> dict[str, list[TaskEvaluation]]
- Produces: build_audit_cases(rows, seeds=(7,8,9,10,11), extremes=2) -> list[AuditCase]

- [ ] **Step 1: Write parser and selection tests**

Use a synthetic log containing repeated Task lines, Success booleans, and Current task success rate records. Assert task ids follow completed block order, all 50-style outcomes remain ordered, and malformed/incomplete blocks raise ValueError.

Test deterministic ties:

~~~python
selected = select_extreme_tasks(rows, count=2)
assert [row.task_id for row in selected["best"]] == [0, 1]
assert [row.task_id for row in selected["worst"]] == [4, 3]
~~~

Assert a best task chooses its first successful index, a worst task chooses its first failure, and five seeds create five cases with one shared initial-state index.

- [ ] **Step 2: Run tests and verify failure**

Run:

~~~bash
pytest -q tests/test_libero_trace_audit_selection.py
~~~

Expected: import failure for the new module.

- [ ] **Step 3: Implement immutable selection dataclasses and parser**

Parse only explicit Task and Success records. Validate that each task has one language and at least one outcome. Sort best by descending rate then ascending task id; sort worst by ascending rate then ascending task id. Reject overlap when a suite has fewer than four distinct tasks.

- [ ] **Step 4: Add real-log contract test**

Point a test helper at an injected fixture path rather than the absolute output path. Add a CLI-level validation function that requires four suites, four selected tasks per suite, and 80 cases.

- [ ] **Step 5: Run focused tests**

Run:

~~~bash
pytest -q tests/test_libero_trace_audit_selection.py
~~~

Expected: all pass.

- [ ] **Step 6: Commit**

~~~bash
git add examples/simBenchmarks/CoT/geometry_probe/libero_trace_audit_selection.py tests/test_libero_trace_audit_selection.py
git commit -m "feat: select LIBERO trace audit tasks"
~~~

### Task 3: Canonicalize, align, and score three-point traces

**Files:**
- Create: examples/simBenchmarks/CoT/geometry_probe/libero_trace_audit_metrics.py
- Modify: examples/simBenchmarks/CoT/geometry_probe/probe_utils.py
- Create: tests/test_libero_trace_audit_metrics.py

**Interfaces:**
- Produces: canonicalize_v3_uvd(flat, time_points=4, landmarks=3) -> ndarray[T,L,3]
- Produces: align_realized_trace(step_uvd, anchor, offsets) -> (target[T,L,3], valid[T,L])
- Produces: compute_anchor_metrics(pred, target, valid, image_size, depth_dead_zone_m) -> dict
- Produces: backproject_uvd(uvd, camera_k, image_size) -> camera XYZ

- [ ] **Step 1: Write canonicalization and early-termination tests**

Use a flattened time-major 12x3 array with unique token ids and assert reshape order is t0-left/right/wrist followed by t1-left/right/wrist. Test offsets that extend past episode end produce false validity without repeated tail values.

- [ ] **Step 2: Write exact metric tests**

Construct two time points with known pixel and depth offsets. Assert per-landmark UV ADE in pixels, d MAE in millimetres, adjacent delta-d MAE, FDE, direction accuracy with 2 mm dead zone, and persistence values. Include one invalid landmark and assert it creates NaN/gaps rather than zeros.

- [ ] **Step 3: Run tests and verify failure**

Run:

~~~bash
pytest -q tests/test_libero_trace_audit_metrics.py
~~~

Expected: import failure for the new module.

- [ ] **Step 4: Implement metrics from existing probe utilities**

Reuse canonicalize_uvd_prediction and uvd_trajectory_metrics where their contracts match. Add explicit per-anchor/per-landmark outputs:

~~~python
{
    "left": {"uv_ade_px": ..., "d_mae_mm": ..., "delta_d_mae_mm": ...},
    "right": {...},
    "wrist": {...},
    "aggregate": {...},
    "persistence": {...}
}
~~~

Compute center/span/approach in back-projected camera XYZ and keep projection-valid and in-frame counts separate.

- [ ] **Step 5: Run focused and existing metric tests**

Run:

~~~bash
pytest -q tests/test_libero_trace_audit_metrics.py tests/test_cot_diagnostics.py tests/test_paired_geometry_probe.py
~~~

Expected: all pass.

- [ ] **Step 6: Commit**

~~~bash
git add examples/simBenchmarks/CoT/geometry_probe/libero_trace_audit_metrics.py examples/simBenchmarks/CoT/geometry_probe/probe_utils.py tests/test_libero_trace_audit_metrics.py
git commit -m "feat: score landmark UVD rollout traces"
~~~

### Task 4: Collect resumable selected-task rollouts

**Files:**
- Create: examples/simBenchmarks/CoT/geometry_probe/libero_trace_audit_rollout.py
- Create: tests/test_libero_trace_audit_rollout.py

**Interfaces:**
- Consumes: AuditCase and websocket response from Task 1
- Produces: collect_rollout(case, client, args) -> RolloutRecord
- Produces: save_rollout_record(path, record) and load_rollout_record(path)
- Raw record contains RGB/wrist frames, metric agentview depth, executed actions, anchor steps, predicted UVD/dense depth, realized per-step UVD/XYZ/valid masks, dense-depth targets, latency, metadata, and outcome

- [ ] **Step 1: Write pure cadence and request tests**

Assert dummy stabilization steps are excluded, a request occurs at anchors 0,8,16, and inference_seed is derived deterministically from audit seed and anchor index. Assert the request sets return_geometry=True and uses both images in primary/wrist order.

- [ ] **Step 2: Write raw-record round-trip tests**

Create a two-anchor synthetic RolloutRecord, save compressed NPZ plus JSON metadata, reload it, and assert exact shapes/dtypes. Corrupt a required field and assert validation fails before resume accepts it.

- [ ] **Step 3: Run tests and verify failure**

Run:

~~~bash
pytest -q tests/test_libero_trace_audit_rollout.py
~~~

Expected: import failure for the new module.

- [ ] **Step 4: Implement simulator collection**

Use OffScreenRenderEnv with agentview and robot0_eye_in_hand RGB plus agentview depth. Reuse standard evaluation image flipping, gripper post-processing, task max-step limits, and ten dummy steps. Read LANDMARK_BODY_NAMES from starVLA.gripper_triangle and project body_xpos with robosuite camera calibration, applying the same 180-degree image flip.

- [ ] **Step 5: Complete anchors after rollout**

Cache predictions at each request. After the episode, align each anchor with realized per-step LRW using response uvd_time/landmark ids and mark incomplete horizons invalid. Compute anchor metrics through Task 3 before writing the finalized raw record.

- [ ] **Step 6: Run focused tests**

Run:

~~~bash
pytest -q tests/test_libero_trace_audit_rollout.py tests/test_libero_gripper_triangle_sidecar.py
~~~

Expected: all pass.

- [ ] **Step 7: Commit**

~~~bash
git add examples/simBenchmarks/CoT/geometry_probe/libero_trace_audit_rollout.py tests/test_libero_trace_audit_rollout.py
git commit -m "feat: collect LIBERO landmark trace rollouts"
~~~

### Task 5: Render per-rollout videos and five-seed summaries

**Files:**
- Create: examples/simBenchmarks/CoT/geometry_probe/libero_trace_audit_visualization.py
- Create: tests/test_libero_trace_audit_visualization.py

**Interfaces:**
- Consumes: validated RolloutRecord
- Produces: render_dashboard_frame(record, frame_index) -> uint8 RGB array
- Produces: render_rollout_video(record, path, fps=10) -> Path
- Produces: render_rollout_summary(record, path) -> Path
- Produces: render_task_seed_summary(records, path) -> Path
- Produces: render_suite_contact_sheet(task_summaries, path) -> Path

- [ ] **Step 1: Write fixed-layout frame tests**

Build a synthetic record with three landmarks and two anchors. Assert the renderer returns identical 2560x1440x3 uint8 shapes at early/late frames, fixed axis positions/limits, and different moving-cursor locations.

- [ ] **Step 2: Write curve-content tests**

Inspect plotted line data or the prepared view model. Assert each of the UV ADE, d MAE, and delta-d MAE panels contains exactly three left/right/wrist curves, uses anchor steps on x, preserves NaN gaps, and uses px/mm/mm units.

- [ ] **Step 3: Write video and summary smoke tests**

Write a two-frame H.264/yuv420p MP4 and decode it to verify frame count. Render one per-rollout PNG, one five-seed summary, and one suite contact sheet with nonzero dimensions.

- [ ] **Step 4: Run tests and verify failure**

Run:

~~~bash
pytest -q tests/test_libero_trace_audit_visualization.py
~~~

Expected: import failure for the new module.

- [ ] **Step 5: Implement the stable dashboard**

Render:
- agentview with predicted solid and simulator-realized dashed LRW triangles;
- wrist inset and metadata;
- fixed full-episode translation, rotation-vector delta, and gripper action curves;
- active-chunk predicted/realized/persistence depth for left/right/wrist;
- three full-episode error panels, each with separate left/right/wrist curves and a moving anchor cursor;
- a two-second final summary card.

Use landmark colors consistently across overlay and plots. Label realized traces explicitly as simulator-realized, not expert GT.

- [ ] **Step 6: Run visualization regression tests**

Run:

~~~bash
pytest -q tests/test_libero_trace_audit_visualization.py tests/test_episode_geometry_fixed_layout.py tests/test_episode_geometry_h264.py
~~~

Expected: all pass.

- [ ] **Step 7: Commit**

~~~bash
git add examples/simBenchmarks/CoT/geometry_probe/libero_trace_audit_visualization.py tests/test_libero_trace_audit_visualization.py
git commit -m "feat: visualize LIBERO landmark trace audits"
~~~

### Task 6: Add manifest CLI and four-GPU launcher

**Files:**
- Create: examples/simBenchmarks/CoT/geometry_probe/run_libero_v3_trace_audit.py
- Create: examples/simBenchmarks/CoT/geometry_probe/run_libero_v3_trace_audit.sh
- Create: tests/test_libero_trace_audit_cli.py

**Interfaces:**
- CLI inputs: checkpoint, source-log-dir, output-dir, suites, seeds, host/port, task filter, max cases, render-only, dry-run
- CLI outputs: selection.json, run_manifest.json, episodes.jsonl, summary.json, summary.csv, raw/, videos/, summaries/, contact_sheets/
- Launcher inputs: MODEL_DIR, CKPT_NAME, SOURCE_LOG_DIR, OUTPUT_DIR, GPUS, BASE_PORT, SEEDS

- [ ] **Step 1: Write dry-run manifest tests**

Provide four synthetic worker logs. Assert dry-run emits 16 selected tasks, seeds 7-11, 80 cases, stable artifact ids, and no simulator/server side effects. Test max-cases=1 for smoke use.

- [ ] **Step 2: Run tests and verify failure**

Run:

~~~bash
pytest -q tests/test_libero_trace_audit_cli.py
~~~

Expected: import failure for the new CLI.

- [ ] **Step 3: Implement resumable CLI**

Write selection and run manifests atomically. Skip a case only when its raw record validates. Append one finalized episodes.jsonl row per case without duplicates. Aggregate per landmark, task, rank group, suite, and seed into JSON/CSV.

- [ ] **Step 4: Implement four-GPU launcher**

Start one normal policy server per suite/GPU, wait for all ports, run one worker per suite, clean up only child processes, and aggregate/render after workers finish. Add DRY_RUN=1 and MAX_CASES=1 smoke controls. Refuse to start if output manifest points to a different checkpoint or selection.

- [ ] **Step 5: Run CLI and shell checks**

Run:

~~~bash
pytest -q tests/test_libero_trace_audit_cli.py
bash -n examples/simBenchmarks/CoT/geometry_probe/run_libero_v3_trace_audit.sh
DRY_RUN=1 bash examples/simBenchmarks/CoT/geometry_probe/run_libero_v3_trace_audit.sh
~~~

Expected: tests pass, shell syntax passes, dry run lists four suites and 80 cases.

- [ ] **Step 6: Commit**

~~~bash
git add examples/simBenchmarks/CoT/geometry_probe/run_libero_v3_trace_audit.py examples/simBenchmarks/CoT/geometry_probe/run_libero_v3_trace_audit.sh tests/test_libero_trace_audit_cli.py
git commit -m "feat: launch LIBERO V3 trace audits"
~~~

### Task 7: Real smoke test, full generation, and verification

**Files:**
- Create during run: output audit directory only
- Modify only if a verified defect is found: files owned by Tasks 1-6

**Interfaces:**
- Consumes: V3 final_model checkpoint and completed standard LIBERO worker logs
- Produces: 80 validated rollout bundles/videos and all summaries from the approved spec

- [ ] **Step 1: Run the complete CPU test suite for touched areas**

Run:

~~~bash
pytest -q tests/test_qwen_gr00t_cot_v2.py tests/test_qwen_gr00t_cot_v3.py tests/test_policy_wrapper_geometry_response.py tests/test_libero_trace_audit_selection.py tests/test_libero_trace_audit_metrics.py tests/test_libero_trace_audit_rollout.py tests/test_libero_trace_audit_visualization.py tests/test_libero_trace_audit_cli.py
~~~

Expected: all pass.

- [ ] **Step 2: Check GPU/process state without changing Plus**

Run nvidia-smi and process inspection. If Plus still owns required GPU memory, keep implementation complete and wait; do not terminate its servers or workers.

- [ ] **Step 3: Run one real smoke case**

Launch with MAX_CASES=1 and a free GPU. Verify one geometry response has shape 1x12x3, one raw record validates, the MP4 decodes, all three landmark curves exist, and the action-only standard client remains compatible.

- [ ] **Step 4: Inspect the smoke visualization**

Check image flip, LRW body-name order, predicted/realized line styles, point locations, fixed action axes, per-landmark UV/d/delta-d curves, units, and summary labels. Correct only evidence-backed defects and rerun focused tests.

- [ ] **Step 5: Run all 80 rollouts**

Use four GPUs after Plus releases them. Preserve raw records after every case and render incrementally. Do not retry or replace valid outcome mismatches.

- [ ] **Step 6: Validate artifact contract**

Assert:
- 4 suites, 16 tasks, 80 cases;
- 80 valid raw records, MP4s, and rollout PNGs;
- 16 task summaries and 4 contact sheets;
- every MP4 has nonzero frames and H.264/yuv420p;
- summary counts equal episodes.jsonl counts;
- no NaN is silently converted to zero;
- checkpoint, seeds, initial-state indices, and source logs are recorded.

- [ ] **Step 7: Analyze results**

Report best/worst and success/failure contrasts for per-landmark UV ADE, d MAE, delta-d MAE, persistence improvement, action phases, and task-level five-seed variance. Keep self-consistency conclusions separate from expert optimality.

- [ ] **Step 8: Commit any final verification-only fixes separately**

If verification required no code change, skip this step. Otherwise stage only
the named implementation and test files touched by the evidence-backed fix:

~~~bash
git add starVLA/model/framework/VLM4A/QwenGR00TCoTV2.py deployment/model_server/policy_wrapper.py examples/simBenchmarks/CoT/geometry_probe/libero_trace_audit_selection.py examples/simBenchmarks/CoT/geometry_probe/libero_trace_audit_metrics.py examples/simBenchmarks/CoT/geometry_probe/libero_trace_audit_rollout.py examples/simBenchmarks/CoT/geometry_probe/libero_trace_audit_visualization.py examples/simBenchmarks/CoT/geometry_probe/run_libero_v3_trace_audit.py examples/simBenchmarks/CoT/geometry_probe/run_libero_v3_trace_audit.sh tests/test_qwen_gr00t_cot_v3.py tests/test_policy_wrapper_geometry_response.py tests/test_libero_trace_audit_selection.py tests/test_libero_trace_audit_metrics.py tests/test_libero_trace_audit_rollout.py tests/test_libero_trace_audit_visualization.py tests/test_libero_trace_audit_cli.py
git commit -m "fix: validate LIBERO V3 trace audit artifacts"
~~~
