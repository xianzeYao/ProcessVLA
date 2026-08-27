# RoboCasa CoT V5 Temporal-Configuration Token Design

## Status and decision

V5 is a RoboCasa-only experiment built from the validated V2 q0+depth configuration.
It changes only the representation of the existing dual-hand UVD trace:

```text
V2: one Qwen token per (time, hand) -> one UVD point
V5: one Qwen token per time        -> both hands' UVD points
```

The first V5 run keeps the existing action horizon, UVD sampling, depth prediction,
loss weights, action model, data mixture, and optimizer schedule. It does not add a
coarse trace, trace history, flow matching, or new geometry labels.

## Motivation

The current RoboCasa V2 trace contains six sampled times and two hands, so Qwen
processes twelve UVD tokens in time-major order:

```text
[L0, R0, L1, R1, ..., L5, R5]
```

The two hand points at one time are a single robot configuration, not two sequential
reasoning steps. Treating them as separate Qwen tokens couples the representation cost
to the number of effectors and introduces an arbitrary left/right order in layers that
retain native causal attention. V5 instead makes temporal resolution determine the
number of reasoning tokens while a small structured decoder determines geometric
output granularity.

This is not horizon compression. All twelve V2 UVD targets remain supervised and the
six temporal samples remain unchanged.

## Fixed RoboCasa data contract

V5 reuses the existing `robocasa_lerobot_datasets` labels and episode geometry without
regeneration. For every sample:

- action horizon remains 16;
- `uvd_num_points` remains six sampled times;
- `uvd_hand_count` remains two;
- UVD targets remain `[T=6, H=2, C=3]` before packing;
- hand order remains `[left, right]`;
- flattened prediction/target order remains
  `[L0, R0, L1, R1, ..., L5, R5]`;
- UVD frame indices, normalized times, validity masks, out-of-frame masks, and
  boundary-clamp masks retain their V2 definitions;
- depth current/future targets and the Fourier GR1 24-task mixture remain unchanged.

The existing loader already emits the required structured dual-hand labels. V5 adds
no dataset subclass; only model-side target packing and token layout distinguish
reasoning tokens from output points.

## Token layout and attention

With `depth_query_count=8` and `uvd_num_points=6`, the appended V5 geometry sequence is:

```text
[depth_current x 8]
[depth_future  x 8]
[temporal_configuration x 6]
```

V5 therefore appends 22 geometry tokens rather than V2's 28. Each temporal token is:

```text
trajectory_seed + time_embedding(normalized_time)
```

Hand identity is not placed in the Qwen sequence. It is introduced only in the
structured UVD decoder.

The full-attention-layer mask preserves V2's group behavior for current and future
depth tokens. Temporal-configuration tokens are causal by time: token `t` may read
native/depth tokens and configuration tokens at times `<= t`, never future times.
Native Qwen tokens cannot read appended geometry tokens. Linear-attention layers keep
their native causal path.

## Structured multi-hand decoder

Let the Qwen output for temporal slot `t` be `z_t` and let `e_left`, `e_right` be two
learned decoder-only hand embeddings. The shared head predicts:

```text
h[t, hand]   = z_t + hand_embedding[hand]
raw[t, hand] = MLP(h[t, hand])
uv[t, hand]  = sigmoid(raw[..., 0:2])
d[t, hand]   = softplus(raw[..., 2:3])
```

The MLP has the same hidden-width/GELU/three-value structure as the V2 UVD head. Its
output is `[B, 6, 2, 3]`, then flattened time-major to `[B, 12, 3]` for the existing
losses, diagnostics, response fields, and visualizers.

The action model consumes only the six Qwen temporal-configuration hidden states, not
the twelve decoder-expanded hand states and never ground-truth UVD coordinates. Its
condition remains:

```text
[native, depth_current, depth_future, temporal_configuration]
```

`num_target_vision_tokens=0` and `include_depth_in_action_condition=true` are fixed.

## Targets and losses

V5 retains the V2 flattened target contract and losses:

```text
L_uvd = L_absolute + 0.1 * L_adjacent_relative_per_hand

L_total = 1.0  * L_action
        + 0.14 * L_depth_current
        + 0.15 * L_depth_future
        + 0.62 * L_uvd
```

The relative loss compares consecutive temporal samples for the same hand. It must
never compare `L_t` to `R_t`. Existing masks continue to exclude invalid projections.
There is no explicit left/right coordination loss in the first run; sharing `z_t` is
the controlled architectural intervention.

## Output and diagnostics contract

The public prediction field `uvd` remains flattened `[B,12,3]` in V2-compatible
time-major order. `uvd_times` and `uvd_hand_ids` also remain output-point metadata of
length twelve.

The diagnostic field `uvd_tokens` contains the six actual Qwen temporal states. Token
cosine/effective-rank metrics therefore measure six temporal reasoning tokens rather
than twelve hand-time output slots. V5 exposes an explicit token-semantics marker so
diagnostics cannot infer that one hidden token corresponds to one output point.

Depth diagnostics, action loss metrics, per-time/per-hand UVD errors, and inference
response validation remain enabled. V5 checkpoint validation must reject a strict V2
load rather than silently treating V2's hand-specific geometry tokens as temporal
configuration tokens.

## Version isolation and implementation boundary

V2, V3, and V4 code, YAMLs, checkpoint contracts, and launchers remain unchanged. V5
may reuse stable projection, data loading, depth decoding, loss, and Qwen-forward
utilities, but owns its token layout, embeddings, attention mask, structured decoder,
framework registration, tests, YAML, trainer entrypoint, and launcher.

The intended V5 files are:

- `starVLA/model/modules/geometric_cot_v5.py`
- `starVLA/model/framework/VLM4A/QwenGR00TCoTV5.py`
- `starVLA/training/train_starvla_cot_v5.py`
- `examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v5_q0_depthcond.yaml`
- `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_CoT_v5_common.sh`
- `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_robocasa_fourier_CoT_v5.sh`
- focused V5 unit/config/entrypoint tests under the existing CoT test locations.

No V5 LIBERO adapter or configuration is added in this implementation.

## Training baseline

The initial V5 YAML copies the RoboCasa V2 q0+depth training contract:

- Qwen3.5-4B base initialization;
- DiT-B, action dimension 29, state dimension 58;
- action horizon 16 and zero target-vision/action query tokens;
- eight current-depth and eight future-depth queries;
- six UVD temporal samples and two hands;
- per-device batch size 16, eight-GPU-compatible launcher;
- 100k steps, 5k warmup, identical learning rates and loss weights;
- seed 42 and the same Fourier GR1 unified mixture.

The first experiment starts from the same base initialization as V2, not from the V2
trained checkpoint, so the V2/V5 comparison changes only the trace-token architecture.

## Validation

Unit and contract tests must cover:

- six sampled times create six Qwen UVD tokens and twelve output points;
- geometry sequence length is `native + 8 + 8 + 6`;
- hand expansion order is exactly `[L0,R0,L1,R1,...]`;
- one temporal hidden state produces two hand predictions through distinct hand
  embeddings and a shared head;
- U/V and depth output activations remain sigmoid/softplus;
- full-layer attention is causal across temporal slots and retains both depth groups;
- action condition contains six temporal trace states and no expanded decoder states;
- target packing and relative loss preserve per-hand temporal adjacency;
- inference returns V2-compatible flattened UVD values and metadata;
- V2 registry/config/token counts and representative tests remain unchanged;
- V5 YAML is q0+depth with the exact RoboCasa horizon, hand count, data mixture, and
  100k schedule;
- the V5 trainer imports with `--help`, and the launcher selects the V5 YAML/trainer
  while forwarding overrides;
- a CPU module fixture exercises layout, mask, decoder, packing, and loss contracts;
- a one-batch model/data smoke test runs when the local model and dataset are available.

## Non-goals

- Do not add LIBERO V5 in this change.
- Do not change UVD sample count, action horizon, or terminal behavior.
- Do not add trace history, coarse/full trajectories, flow matching, or token
  orthogonality losses.
- Do not modify V2 behavior or migrate V2 checkpoints.
- Do not claim reduced token count alone guarantees faster wall-clock training; the
  experiment tests representation quality and action success first.
