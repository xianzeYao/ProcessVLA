# RoboCasa CoT V5 LRW Hand-Configuration Token Design

## Status and decision

V5 is a RoboCasa-only experiment built from the validated V2 q0+depth configuration.
It preserves V2's six sampled times and two-hand token layout, but replaces each
hand's single pinch-midpoint target with a three-landmark hand configuration:

```text
[L, R, W] = [thumb distal, index intermediate, wrist/hand base]
```

The semantic unit of a Qwen trace token is therefore one `(time, hand)` configuration:

```text
V2: 12 Qwen tokens -> 12 pinch-midpoint UVD points
V5: 12 Qwen tokens -> 12 hand configurations -> 36 LRW UVD points
```

V5 deliberately does not allocate one Qwen token per physical landmark. Landmark
expansion happens only in the UVD decoder. The first run keeps V2's action horizon,
UVD temporal sampling, depth prediction, action model, outer loss weights, data
mixture, optimizer, and training schedule unchanged.

## Motivation

The current RoboCasa V2 trace contains six sampled times and two hands in time-major
order:

```text
[left@t0, right@t0, left@t1, right@t1, ..., left@t5, right@t5]
```

Each target is currently the midpoint of the corresponding thumb-distal and
index-intermediate bodies. That point describes hand translation but discards the
relative geometry of the fingers and wrist. A direct V3-style extension would place
all 36 landmark-time pairs in Qwen, tripling trace token count even though the three
landmarks describe a single hand configuration.

V5 decouples reasoning-token granularity from geometric-output granularity. Qwen
retains one state for each hand at each sampled time; a structured decoder expands
that state into thumb, index, and wrist coordinates. This tests whether richer
end-effector geometry helps action prediction without increasing Qwen trace tokens.

## Fixed temporal and training contract

The first V5 experiment preserves the RoboCasa V2 q0+depth settings:

- action horizon remains 16;
- `uvd_num_points` remains six sampled times per hand;
- `uvd_hand_count` remains two, ordered `[left, right]`;
- Qwen UVD token order remains time-major
  `[left@t0, right@t0, ..., left@t5, right@t5]`;
- UVD frame indices, normalized times, terminal clamping, and temporal masks retain
  their V2 definitions;
- current/future depth queries and losses remain unchanged;
- the Fourier GR1 24-task mixture remains unchanged;
- `num_target_vision_tokens=0` and
  `include_depth_in_action_condition=true` remain fixed.

V5 adds no coarse trace, trace history, flow matching, or new action queries.

## RoboCasa LRW sidecar

The existing rerender dataset contains only the thumb-index midpoint, so V5 cannot
recover separate LRW landmarks from its current Parquet columns. It must build an
episode-level sidecar by replaying the original RoboCasa MuJoCo states, following the
proven LIBERO V3 sidecar pattern.

For each hand, physical body positions are:

| hand | L: thumb | R: index | W: wrist/hand base |
| --- | --- | --- | --- |
| left | `gripper0_left_L_thumb_distal_link` | `gripper0_left_L_index_intermediate_link` | `gripper0_left_left_hand` |
| right | `gripper0_right_R_thumb_distal_link` | `gripper0_right_R_index_intermediate_link` | `gripper0_right_right_hand` |

Here `L/R/W` are landmark names, not left/right-hand identifiers. The sidecar uses
separate hand and landmark axes so the distinction is explicit.

For each original HDF5 demo, the builder restores every saved simulator state,
extracts the six body positions, and projects them with the rerender episode's stored
agentview intrinsics/extrinsics. It writes one atomic, idempotent sidecar per episode:

```text
world_xyz                  float32 [frames, 2 hands, 3 landmarks, 3]
agentview_uvd_pixels       float32 [frames, 2 hands, 3 landmarks, 3]
agentview_projection_valid bool    [frames, 2 hands, 3 landmarks]
agentview_in_frame         bool    [frames, 2 hands, 3 landmarks]
```

Axes are ordered as:

```text
hand:     [left, right]
landmark: [L=thumb, R=index, W=wrist]
coord:    [u, v, depth]
```

Positive-depth points remain geometrically valid even when occluded. In-frame masks
use the same agentview content region and image convention as the current RoboCasa
loader. Source demo IDs and frame counts must match the rerender episode before a
sidecar is accepted. The builder is resumable; it writes a temporary file and
atomically renames only after shape, finiteness, and alignment validation. Dataset
metadata advertises the sidecar only after every episode in that task root validates.

## Dataset adapter and target packing

V5 owns a RoboCasa dataset adapter rather than changing the V2 loader. It loads the
episode LRW sidecar and samples the same six future frame indices already selected by
V2. Its canonical target is:

```text
[batch, time=6, hand=2, landmark=3, coord=3]
```

The public flattened order is:

```text
[L_left_t0, R_left_t0, W_left_t0,
 L_right_t0, R_right_t0, W_right_t0,
 ...,
 L_right_t5, R_right_t5, W_right_t5]
```

The adapter emits aligned time, hand, landmark, validity, in-frame, and boundary
metadata of length 36. It never derives LRW points from the old pinch midpoint, and
V2 continues to load its original labels unchanged.

## Qwen token layout and attention

With `depth_query_count=8`, `uvd_num_points=6`, and two hands, the appended geometry
sequence remains identical in length to RoboCasa V2:

```text
[depth_current x 8]
[depth_future  x 8]
[hand_configuration x 12]
```

Each hand-configuration seed combines a learned trace seed with the existing time and
hand identities. Qwen therefore receives 28 appended geometry tokens, including 12
UVD tokens, rather than 36 landmark tokens.

Full-attention-layer masks retain V2's depth-group behavior and its causal order over
the time-major hand tokens. Native Qwen tokens cannot read appended geometry tokens;
linear-attention layers retain their native causal path. V5 does not merge left and
right hands into one temporal token.

## Structured LRW decoder

Let `z[t,h]` be the Qwen state for time `t` and hand `h`. The decoder owns three
learned landmark embeddings `e_L`, `e_R`, and `e_W`, which never enter Qwen:

```text
q[t,h,k]   = z[t,h] + landmark_embedding[k]
raw[t,h,k] = shared_MLP(q[t,h,k])
uv[t,h,k]  = sigmoid(raw[..., 0:2])
d[t,h,k]   = softplus(raw[..., 2:3])
```

The shared MLP follows the V2 UVD head's hidden-width/GELU/three-value structure. Its
structured output is `[B,6,2,3,3]`; flattening to `[B,36,3]` happens only at the
loss/diagnostic interface.

The action model consumes the 12 actual Qwen hand-configuration states. It never
consumes the 36 decoder-expanded landmark features or ground-truth LRW coordinates:

```text
[native, depth_current, depth_future, hand_configuration]
```

This keeps the intervention interpretable: richer physical supervision is attached
to the same number of Qwen trace states used by V2.

## Losses

The absolute UVD loss supervises every valid LRW landmark. The temporal relative loss
compares consecutive sampled times only for the same `(hand, landmark)`:

```text
L_uvd = L_absolute_LRW + 0.1 * L_temporal_relative_LRW

L_total = 1.0  * L_action
        + 0.14 * L_depth_current
        + 0.15 * L_depth_future
        + 0.62 * L_uvd
```

Flattened adjacency must not accidentally compare different landmarks or hands. The
V3 geometric-shape loss interface may be reused, but the first controlled run fixes
`lambda_uvd_shape=0.0`; adding a triangle/shape regularizer is a later ablation, not
part of the initial V2 comparison.

## Diagnostics and visualization

The prediction field `uvd` is flattened `[B,36,3]` with explicit `uvd_times`,
`uvd_hand_ids`, and `uvd_landmark_ids`. The diagnostic field `uvd_tokens` contains
only the 12 actual Qwen states, so token cosine/effective-rank metrics continue to
measure hand-time reasoning tokens rather than decoder-expanded outputs.

After all sidecars pass validation, a reproducible visualization job uses a fixed
random seed to select ten distinct tasks from the 24-task mixture and one valid
episode/window from each task. For every selected window it saves:

- an agentview overlay showing the six-step trajectories of L/thumb, R/index, and
  W/wrist for both hands, with hand, landmark, and temporal direction visually
  distinguishable;
- a companion record containing task, source demo ID, episode index, six frame
  indices, UVD values, validity/in-frame masks, and the random seed;
- a ten-task contact sheet for rapid inspection.

Selection is recorded in a manifest before rendering so failed or unattractive
examples are not silently resampled. Overlays use 2D pixel location for placement
and visibly encode UVD depth (for example by marker size or opacity plus a legend),
so the figure checks all three UVD coordinates rather than only UV. The job also
reports invalid/out-of-frame counts for each selected trajectory.

## Version isolation and intended files

V2, V3, and V4 code, YAMLs, checkpoint contracts, and launchers remain unchanged. V5
may reuse stable projection, depth decoding, loss, and Qwen-forward utilities, but
owns its sidecar contract, data adapter, token/target packing, LRW decoder, framework
registration, tests, YAML, trainer entrypoint, and launcher.

The intended V5 surface is:

- `starVLA/robocasa_hand_lrw.py`
- `examples/modelExtensions/CoT/scripts/build_robocasa_hand_lrw_sidecars.py`
- `examples/modelExtensions/CoT/scripts/visualize_robocasa_hand_lrw_sidecars.py`
- `starVLA/dataloader/robocasa_v5_lerobot_datasets.py`
- `starVLA/model/modules/geometric_cot_v5.py`
- `starVLA/model/framework/VLM4A/QwenGR00TCoTV5.py`
- `starVLA/training/train_starvla_cot_v5.py`
- `examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v5_q0_depthcond.yaml`
- `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_CoT_v5_common.sh`
- `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_robocasa_fourier_CoT_v5.sh`
- focused V5 sidecar/data/model/config/entrypoint tests under existing test locations.

No LIBERO V5 adapter or configuration is added.

## Training baseline

The initial V5 YAML copies the RoboCasa V2 q0+depth training contract:

- Qwen3.5-4B base initialization;
- DiT-B, action dimension 29, state dimension 58;
- action horizon 16 and zero target-vision/action query tokens;
- eight current-depth and eight future-depth queries;
- six UVD times, two hands, and three LRW landmarks per hand-time token;
- per-device batch size 16 and the existing eight-GPU-compatible launcher;
- 100k steps, 5k warmup, identical learning rates and outer loss weights;
- seed 42 and the same Fourier GR1 unified mixture.

The first experiment starts from the same base initialization as V2, not a trained V2
checkpoint, so the comparison changes only the trace target/decoder architecture and
the information represented by each trace token.

## Validation

Tests and data checks must cover:

- sidecar extraction resolves all six named bodies for representative left/right
  episodes and fails loudly when a body is missing;
- original-demo/rerender frame counts and source IDs align before writing;
- LRW sidecars have exact shape/dtype/axis order, finite values, and atomic/resumable
  behavior;
- sampled targets are `[B,6,2,3,3]`, flatten to 36 points in the documented order,
  and preserve V2's six temporal indices and terminal handling;
- six times and two hands create 12 Qwen UVD tokens, not six or 36;
- geometry sequence length is `native + 8 + 8 + 12`;
- each `(time,hand)` hidden state produces three predictions through distinct LRW
  embeddings and the shared head;
- U/V and depth activations remain sigmoid/softplus;
- action conditioning contains 12 hand-configuration states and no decoder-expanded
  landmark states;
- relative losses preserve `(hand,landmark)` temporal adjacency;
- inference emits 36 predictions with aligned time/hand/landmark metadata;
- V2 registry/config/token counts and representative tests remain unchanged;
- V5 YAML is q0+depth with the exact RoboCasa horizon, mixture, and 100k schedule;
- the V5 trainer imports with `--help`, and its launcher selects only V5 components;
- a CPU fixture exercises layout, attention, decoder, packing, and loss contracts;
- a one-batch model/data smoke test runs when local model and data are available;
- the fixed-seed ten-task visualization manifest, per-task overlays/records, and
  contact sheet are produced and contain no silent resampling.

## Non-goals

- Do not add LIBERO V5 in this change.
- Do not change action horizon, temporal sample count, terminal behavior, or depth
  supervision.
- Do not add trace history, coarse/full trajectories, flow matching, or token
  orthogonality losses.
- Do not enable shape loss in the first run.
- Do not modify V2 behavior or migrate V2 checkpoints.
- Do not claim denser LRW supervision guarantees better policy success; this is the
  experiment to be tested.
