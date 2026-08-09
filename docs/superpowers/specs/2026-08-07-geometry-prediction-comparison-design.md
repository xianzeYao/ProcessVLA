# V1.5/V2 Geometry Prediction Comparison

## Goal

Separate two possible causes of policy-score differences:

1. depth/UVD prediction quality differs between model versions;
2. decoded geometry is similar, but the geometry latent affects the action expert differently.

The first stage is offline and uses fixed training samples. Only after its coordinate transforms and metrics are verified do we add a small LIBERO rollout diagnostic.

## Models

### LIBERO

- V1.5: `/root/data/yxz/outputs/qwen35_gr00t_libero_CoT_v1_5_60k_4gpu/final_model/pytorch_model.pt`
- V2: `/root/data/yxz/outputs/qwen35_gr00t_libero_CoT_v2_8gpu_bs16/final_model/pytorch_model.pt`

The corresponding 60k checkpoints may be used instead of `final_model` only after verifying they are byte-equivalent or loading the same training step.

### RoboCasa

- V1: `/root/data/yxz/outputs/qwen35_gr00t_robocasa_fourier_CoT_v1/final_model/pytorch_model.pt`
- V2: `/root/data/yxz/outputs/qwen35_gr00t_robocasa_fourier_CoT_v2_8gpu_bs16/final_model/pytorch_model.pt`

These are the normal 100k runs. Dataset statistics and the saved run configuration must be loaded from the same run directory as each checkpoint.

## Stage 1: Fixed Training-Sample Probe

Create stable JSON manifests containing dataset name, trajectory ID, and base-frame index. Both versions must consume exactly the same manifest.

- LIBERO: 240 samples, 60 from each of spatial, object, goal, and LIBERO-10.
- RoboCasa: 240 samples, exactly 10 from each of the 24 training tasks.
- Prefer different episodes over adjacent frames from the same episode.
- Use seed 42 only to create the manifest; subsequent runs read the saved manifest and perform no sampling.

For each model, call the existing `predict_geometry()` interface and save:

- predicted and GT current depth;
- predicted and GT future depth;
- predicted and GT UVD;
- valid/out-of-frame masks and sample identity metadata.

RoboCasa outputs must be converted to a common `[time, hand, 3]` representation before comparison: V1 is hand-major internally and V2 is time-major internally.

## Metrics and Visuals

Report paired aggregate and per-task metrics.

Depth metrics, calculated only on valid GT pixels:

- masked MAE;
- RMSE;
- AbsRel;
- delta-1 accuracy;
- the same masked SmoothL1 objective used during training.

UVD metrics:

- UV average displacement error in pixels;
- UV final displacement error in pixels;
- depth/Z MAE in meters;
- valid and out-of-frame results separately;
- per-hand metrics for RoboCasa.

Render a deterministic subset of representative and worst-error samples. Each panel contains RGB, GT, V1.5/V1 prediction, V2 prediction, and absolute-error maps. UVD trajectories are overlaid on RGB with consistent time colors and separate hand styles.

The probe writes a summary JSON/CSV, per-sample records, raw prediction arrays, and PNG panels under one timestamped diagnostic output directory.

## Stage 2: Small LIBERO Rollout Probe

Proceed only after Stage 1 demonstrates correct image rotation, resize, depth units, UVD ordering, and future-frame alignment.

Use the LIBERO V1.5 and V2 checkpoints on the same fixed episodes:

- 32 nominal episodes;
- 32 camera-perturbed episodes;
- 32 robot-initial-state episodes;
- balance each slice across the four LIBERO suites where possible.

Collect metric depth, camera matrices, and EEF world position from the simulator without feeding GT data to the policy. Store predictions at environment timestep `t` and align them offline to realized frames at the same temporal offsets used by training, including the future endpoint at `t + 8`.

Report geometry metrics by perturbation, model, suite, and episode success. This distinguishes:

- geometry failure under camera/robot shift;
- good geometry with failed control;
- errors concentrated in already-failed rollouts.

## Interpretation

- If V1/V1.5 geometry is worse than V2 on the same training samples, investigate the external two-layer QFormer capacity and optimization.
- If geometry is similar but task success differs, investigate the high-dimensional UVD latent and its action conditioning; decoded UVD alone is not the action input.
- If training geometry is similar but rollout geometry diverges, investigate OOD generalization and camera/robot-state sensitivity.
- The shared bounding-box CoT prompt is not a cause of relative differences in these runs because baseline, V1/V1.5, and V2 use the same prompt; prompt ablation is a separate experiment.

## Acceptance Criteria

- Both versions evaluate identical manifest entries with identical GT tensors.
- Coordinate conventions and depth units pass explicit sanity checks.
- Aggregate metrics can be recomputed from saved per-sample records.
- At least one compact comparison panel is produced for every dataset/task group.
- Stage 2 is not started until Stage 1 outputs have been visually reviewed.
