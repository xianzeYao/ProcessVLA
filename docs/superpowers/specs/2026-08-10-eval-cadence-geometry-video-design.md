# Evaluation-Cadence Geometry Video Design

## Goal

Correct the existing LIBERO and RoboCasa episode geometry videos so model geometry is inferred at the same cadence as benchmark control, while every source frame is still rendered. Add an episode-wide 3D UVD visualization for training-set diagnostics. Add a small online-evaluation probe that compares predicted current and future depth against simulator depth and renders predicted UVD over an online RGB-D point cloud.

The first validation scope is one previously selected LIBERO training episode, one previously selected RoboCasa training episode, one LIBERO online evaluation episode, and one RoboCasa online evaluation episode. Existing artifacts are not overwritten.

## Terminology

- **Action horizon:** Number of actions and geometric future span produced by the checkpoint. LIBERO uses 8; RoboCasa uses 16.
- **Execution horizon / inference cadence:** Number of environment actions executed before the next model inference. LIBERO uses 8. The current RoboCasa batch evaluation uses 12.
- **Render cadence:** Number of source or simulator frames between video frames. It is 1 for these videos.
- **Anchor:** An environment/source frame at which a new model inference is performed.

These values must be represented separately in code and artifact metadata.

## Training-Set Episode Videos

### Data flow

The existing rerender datasets remain the source of RGB, dense metric depth, GT UVD, timestamps, and language. No training-set point clouds are materialized.

Each episode has two plans:

1. A render plan containing every eligible source frame.
2. An inference plan containing only chunk anchors.

LIBERO anchors are `0, 8, 16, ...`. RoboCasa anchors are `0, 12, 24, ...`. A prediction is cached at its anchor and reused for all rendered frames until the next anchor. Reuse means the visualization retains the same prediction; it does not claim that the model was rerun at an intermediate frame.

### UVD timing

The complete sparse UVD trajectory from every anchor is retained.

- LIBERO's four points correspond to offsets `[0, 3, 5, 8]` within its 8-step horizon.
- RoboCasa's six points correspond to offsets `[0, 3, 6, 10, 13, 16]` within its 16-step horizon.

For RoboCasa, offsets after the 12-step execution boundary are drawn with a dashed line to show that the checkpoint predicted them but the standard evaluator replans before executing them.

Episode-wide predicted curves are built from anchored sparse segments, not horizon-zero predictions made at every frame. GT UVD remains the complete dataset trajectory. The video displays a moving episode-time marker and a chunk-local execution marker.

### 3D UVD panel

Add a fixed-view 3D axis with coordinates `(normalized u, normalized v, camera-z depth)`. It plots the complete episode GT trajectory plus the anchored v1/v2 sparse prediction segments. LIBERO has one hand trajectory. RoboCasa distinguishes left and right hands by line style or separate markers while retaining checkpoint colors.

Axis limits, camera elevation/azimuth, subplot positions, legends, and color normalization are computed once per episode and remain fixed across frames. This panel is a UVD-coordinate visualization, not a metric XYZ point cloud, so it does not require camera intrinsics.

### Depth semantics

Depth predictions belong to the current anchor. During intermediate render frames, the panels continue to show:

- anchor `depth_current` prediction against anchor current-depth GT;
- anchor `depth_future` prediction against GT at `anchor + action_horizon`.

Titles include the anchor and target frame indices. Intermediate RGB may advance, but depth targets do not silently change. Metrics are computed once per anchor and reused through that chunk.

## Online Evaluation Probe

### Rollout semantics

Run each checkpoint in its normal benchmark environment and let it execute its own actions. v1/v1.5 and v2 use the same task and initial seed, but their rollouts and future GT are kept separate after their trajectories diverge.

At every inference anchor, request actions and geometry in the same server call. Record geometry, anchor RGB, anchor simulator depth, camera calibration, and the target future frame index. Record simulator RGB and metric depth at every executed environment step.

When the rollout reaches `anchor + action_horizon`, attach that observed depth as the anchor's future-depth GT. Thus LIBERO evaluates at `t+8`. RoboCasa evaluates at `t+16`; this is the actual policy future after the normal replan at `t+12`, rather than a counterfactual execution of the discarded final four actions from the original chunk.

Anchors whose target time is never reached because the episode terminates are marked incomplete and excluded from future-depth aggregates. Their current-depth metrics remain valid.

### Benchmark adapters

LIBERO uses the simulator's `agentview_depth`, converts normalized MuJoCo depth to meters when needed, and queries the current agentview intrinsic/extrinsic matrices.

RoboCasa renders egoview depth from the underlying simulator, converts it to meters, and applies the same DIAL flip/crop/pad/resize transform and transformed intrinsic used to create the training data. The existing thumb-index extraction helper remains available for visualization validation, but the first evaluation metric scope does not require UVD GT.

### Metrics

The first online evaluation reports masked current- and future-depth metrics using the same definitions as the training-set probe. It does not report a numeric UVD error metric.

Per-anchor records include task, seed, checkpoint label, anchor frame, target frame, completion state, latency, and depth metrics. Aggregate results keep current and future valid counts separate.

### Point-cloud visualization

At video time, back-project the online simulator's real current RGB-D frame with its matching transformed camera intrinsic. Spatially subsample valid pixels to keep rendering bounded. The point cloud uses RGB colors and fixed episode-level XYZ limits and view angles.

Predicted UVD waypoints are converted from model-normalized pixels and camera-z depth to camera-frame XYZ with the same intrinsic, then overlaid on the real point cloud. The point cloud is rebuilt for each rendered simulator frame; frames are not fused across time. This avoids camera-registration and accumulated-occlusion artifacts.

Predicted and GT depth images and depth-error maps remain the quantitative comparison. The point cloud is a qualitative context for predicted UVD only.

## Interfaces and Artifacts

The episode-video command gains independent parameters for inference cadence and render cadence. Defaults come from the selected benchmark, while explicit overrides remain available for experiments. For backward compatibility, legacy `--stride N` sets both cadences to `N` and emits a deprecation warning; combining it with either new cadence argument is rejected. New cadence-aligned runs use explicit inference and render values.

Training and online-evaluation outputs go into new directories whose names state `cadence_aligned`. Every run writes its effective action horizon, execution horizon, render cadence, UVD offsets, checkpoint labels, task/episode identity, and GT mode into JSON metadata.

The two existing dense, horizon-zero preview videos remain untouched and retain their diagnostic interpretation.

## Testing and Validation

Unit tests cover:

- separate render and inference plans;
- LIBERO and RoboCasa anchor schedules;
- exact UVD frame offsets;
- RoboCasa solid/dashed execution-boundary splitting;
- chunk prediction reuse without intermediate inference;
- full-episode curve construction from sparse anchored segments;
- fixed 3D UVD axis/view/layout;
- UVD-to-XYZ back-projection with synthetic intrinsics;
- delayed future-depth target attachment and early termination;
- point-cloud subsampling and invalid-depth masking;
- CLI validation and artifact metadata.

After unit tests, regenerate the selected LIBERO and RoboCasa training videos and inspect their first, middle, chunk-boundary, and final frames. Then run one online LIBERO episode and one online RoboCasa episode, verify depth units and alignment numerically, inspect their videos, and probe the resulting H.264 files for frame count, FPS, pixel format, and dimensions.

## Non-Goals

- No training-set point-cloud materialization.
- No cross-frame point-cloud fusion.
- No numeric online UVD metric in the first version.
- No attempt to overlay v1 and v2 after their online rollouts diverge.
- No overwrite of existing artifacts and no Git commit.
