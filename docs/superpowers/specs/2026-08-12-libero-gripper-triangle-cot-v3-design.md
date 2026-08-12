# LIBERO Gripper-Triangle CoT V3 Design

## Goal

Add a three-landmark LIBERO geometry representation and a separate
`QwenGR00TCoTV3` training path without changing the behavior, inputs, checkpoints,
or launchers of existing V1/V2 training.

The first V3 experiment represents each sampled future end-effector state with:

1. `left_finger_tip`
2. `right_finger_tip`
3. `wrist_hand_base`

The representation is intended to retain end-effector trajectory information while
making gripper aperture observable through finger-tip separation. This experiment
does not initially claim exact 7-DoF invertibility. A later ablation may replace these
moving landmarks with three fixed points on one rigid body to isolate pure 6D pose
recovery.

RoboCasa is explicitly out of scope.

## Dataset sidecar

Generate one episode-level file:

```text
geometry/gripper_triangle/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npz
```

with:

```text
world_xyz              float32 [T, 3, 3]
agentview_uvd_pixels   float32 [T, 3, 3]
agentview_projection_valid bool [T, 3]
agentview_in_frame     bool    [T, 3]
```

Axis order is `[time, landmark, coordinate]`; landmark order is always
`[left_finger_tip, right_finger_tip, wrist_hand_base]`. UVD depth is camera-frame
positive Z in meters and U/V use the stored agentview image coordinates.

The generator restores the original HDF5 MuJoCo states and reads the physical bodies:

```text
gripper0_finger_joint1_tip
gripper0_finger_joint2_tip
gripper0_right_gripper
```

It projects with the existing per-frame `agentview_K` and
`agentview_T_world_camera` sidecar. Occluded points remain valid geometric targets;
`projection_valid` means finite positive camera depth and does not test visual surface
visibility. `in_frame` additionally requires U/V to lie inside the stored image.
Training uses `in_frame` as its regression-valid mask so an off-screen projection is
not silently clamped into a false image-boundary target. Geometry validation and
visualization retain every positive-depth projection.

To preserve the exact old Parquet read path, do not add a repeated path column to
every frame. Add only top-level metadata to `meta/info.json`:

```json
{
  "geometry_paths": {
    "gripper_triangle": "geometry/gripper_triangle/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npz"
  },
  "gripper_triangle": {
    "landmark_order": ["left_finger_tip", "right_finger_tip", "wrist_hand_base"],
    "frame": "world",
    "uvd_camera": "agentview"
  }
}
```

Existing loaders ignore these optional keys, read the same Parquet bytes as before,
and produce byte-for-byte equivalent sample fields. The V3 loader resolves the
deterministic template by episode index and fails loudly if a sidecar is absent or
has a frame-count/schema mismatch.

The generator is resumable and idempotent. It writes each NPZ to a temporary sibling,
validates it, then atomically replaces the episode target. `meta/info.json` is updated
only after every requested episode succeeds. It supports `--episode-start`,
`--episode-end`, `--overwrite`, and `--validate-only`.

## V3 target and token ordering

For `K` sampled times, the loader emits:

```text
uvd                 float32 [K, 3, 3]
uvd_valid_mask      bool    [K, 3]
uvd_time            float32 [K]
uvd_landmark_ids    int64   [K, 3]
```

The model packs tokens in time-major order:

```text
[L0, R0, W0, L1, R1, W1, ..., L(K-1), R(K-1), W(K-1)]
```

V3 replaces the V2 hand terminology with `landmark_count=3`. Each UVD query is:

```text
trajectory_seed + time_embedding(t) + landmark_embedding(landmark_id)
```

There is no effector or dataset embedding because the LIBERO run contains one
end-effector and is not mixed with RoboCasa. At full-attention layers, all three
landmarks at the same time may read one another bidirectionally; temporal ordering
remains causal across time groups.

With the initial LIBERO configuration `K=4`, V3 has 12 UVD tokens instead of V2's 4.
The UVD head remains a shared 3-value `(u,v,d)` regressor.

## Losses

### Absolute UVD

Keep masked Smooth-L1 on every valid landmark coordinate.

### Temporal delta

Keep the existing adjacent-time displacement objective independently for each
landmark:

```text
pred[t+1, landmark] - pred[t, landmark]
```

versus the corresponding ground-truth delta. Rename the V3-facing metric to
`uvd_temporal_loss`; its configured relative weight remains `0.1` inside the UVD
objective.

### Triangle shape interface

Provide a masked same-time shape loss over two independent vectors:

```text
grasp_axis  = right_finger_tip - left_finger_tip
wrist_axis  = wrist_hand_base - 0.5 * (left_finger_tip + right_finger_tip)
```

Do not optimize three redundant triangle edges. The initial YAML sets
`lambda_uvd_shape: 0.0`, so this interface does not affect the first experiment.
It remains separately logged and can be enabled later for a controlled ablation.
The loss is evaluated in normalized UVD model space only when all three landmarks at
that time are valid; it is translation-invariant but is not presented as a direct SO(3)
rotation loss.

The V3 UVD objective is:

```text
uvd_total = absolute
          + lambda_uvd_temporal * temporal
          + lambda_uvd_shape * shape
```

The outer `lambda_uvd` remains unchanged.

## Framework and compatibility boundary

Add a separately registered `QwenGR00TCoTV3` framework and V3 geometry module rather
than changing V2 configuration semantics. Reuse stable V2 action/depth/native-Qwen
behavior where possible, while keeping V3 layout, target packing, landmark embedding,
loss names, and checkpoint validation explicit.

- V1/V2 YAMLs and launchers remain unchanged.
- The V2 data loader continues deriving one EEF UVD track from
  `observation.state[:, :3]`.
- V3 requires the gripper-triangle sidecar and cannot silently fall back to the V2
  point.
- A V2 CoT checkpoint is not advertised as strict-load compatible with V3 because V3
  adds landmark embeddings and changes the number of fixed geometry slots.
- Training V3 from the configured base VLM/action initialization remains supported.

## Training entry points

Add:

```text
starVLA/training/train_starvla_cot_v3.py
examples/modelExtensions/CoT/configs/qwen35_gr00t_libero_CoT_v3_q0_depthcond.yaml
examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_libero_CoT_v3.sh
examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_CoT_v3_common.sh
```

The initial YAML follows the current LIBERO V2 q0 depth-condition experiment except:

- framework name becomes `QwenGR00TCoTV3`;
- `landmark_count: 3`;
- `uvd_num_points: 4` continues to mean four temporal samples;
- `lambda_uvd_temporal: 0.1`;
- `lambda_uvd_shape: 0.0`;
- dataset entry selects the V3 sidecar-aware loader.

The one-click launcher supports the existing distributed environment overrides and
forwards additional CLI arguments in the same manner as the V2 launcher.

## Validation and tests

### Data generation tests

- Physical body extraction uses the fixed landmark order.
- Projection matches hand-derived camera fixtures.
- Sidecars have exact keys, dtypes, shapes, and frame counts.
- Resuming skips valid sidecars; overwrite replaces only the requested episodes.
- Invalid/corrupt/partial sidecars fail validation.
- Old dataset samples before and after adding sidecars are field- and value-equivalent.

### Loader and ordering tests

- A sampled episode produces `[K,3,3]` targets.
- Packing is exactly `[L0,R0,W0,L1,R1,W1,...]`.
- Short horizons preserve real frame indices and fixed-slot masking.
- Missing sidecar, landmark-order mismatch, non-finite/non-positive depth, and frame-count
  mismatch fail loudly.

### Model and loss tests

- `QwenGR00TCoTV3` registers independently of V2.
- Four times and three landmarks create exactly 12 UVD slots.
- Equal-time landmark queries have distinct embeddings and mutual full-attention.
- Temporal loss compares adjacent time values of the same landmark, never neighboring
  flattened tokens.
- Shape loss is zero for matching translated triangles, positive for incorrect
  aperture/orientation, and mask-safe.
- `lambda_uvd_shape=0.0` makes total training loss independent of the reported shape
  loss while retaining its diagnostics.
- V2 registry, token counts, configs, tests, and sample outputs remain unchanged.

### Entrypoint tests

- YAML fields and run ID are checked exactly.
- Training entry `--help` imports and exits successfully.
- One-click launcher selects the V3 trainer/YAML and forwards arguments.
- A one-batch CPU fixture exercises loader, time-major packing, V3 forward target
  preparation, and all three UVD loss interfaces without requiring a full VLM load.

### Full LIBERO validation

After generation, validate all four suites:

- expected episode and frame counts;
- no missing sidecars;
- positive minimum finger separation and triangle area;
- finite positive camera depth;
- correlation/error between finger separation and stored gripper qpos;
- triangle-frame versus official EEF rotation after fitting one fixed frame offset.

Write a machine-readable report. Rotation/aperture results are evaluation evidence,
not a generation pass condition unless the basic geometry is degenerate.

## Non-goals

- Do not modify RoboCasa data or training.
- Do not replace V2 or migrate existing V2 checkpoints.
- Do not enable triangle shape loss in the first V3 run.
- Do not claim exact RPY/gripper inversion before reporting the recovery validation.
- Do not implement the fixed-rigid-three-point ablation in this change.
