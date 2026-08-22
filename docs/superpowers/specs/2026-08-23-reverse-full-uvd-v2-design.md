# Reverse Full-UVD V2 Design

## Goal

Extend the existing QwenGR00TCoTV2 no-query, depth-conditioned LIBERO experiment with two causally ordered trajectory representations:

1. a reverse, stride-4 full-episode EEF UVD trace from the task-completion frame back to the current frame; and
2. the unchanged forward local UVD trace over the action horizon.

The action model consumes current/future depth, the reverse full trace, and the forward local trace. Existing V2 and V3 configurations remain unchanged unless the new full-trace options are present.

## Data Contract

For a sample at frame `t` in an episode ending at frame `T`, the local V2 target remains unchanged. The new full target uses real frames only:

```text
[T, T-4, T-8, ..., t]
```

The current frame `t` is appended when the stride does not land on it. The target exposes `uvd_full`, `uvd_full_valid_mask`, `uvd_full_frame_indices`, `uvd_full_time`, and `uvd_full_endpoint_indices`. `uvd_full_time` records chronological normalized time and therefore descends from `1.0` at the goal to `0.0` at the current frame.

LIBERO's longest documented training demonstration is 505 steps. A fixed capacity of 128 points covers 506 frames at stride 4, including a non-aligned current endpoint. Samples shorter than the capacity use the existing fixed-slot padding and validity-mask convention. Samples exceeding the configured capacity fail loudly instead of silently truncating the full trajectory.

## Model Architecture

The V2 geometry sequence becomes:

```text
[native image/language]
[current-depth queries]
[future-depth queries]
[reverse full-UVD queries]
[forward local-UVD queries]
```

The causal order lets every local-UVD query read the complete full-UVD prefix, while full-UVD queries cannot read local planning tokens. Full and local traces use separate learned trajectory seeds but share the existing UVD regression head, so the additional capacity is attributable to temporal representation rather than a second decoder.

The q0 experiment continues to use zero learned action/future vision queries. Its action condition is:

```text
[native, current depth, future depth, full UVD, local UVD]
```

## Objectives

The original V2 objective is unchanged. When full-UVD is enabled, add:

```text
L_full = L_full_absolute + lambda_full_relative * L_full_relative
L_total = L_V2 + lambda_full_uvd * L_full
```

The experiment starts with `lambda_full_uvd=0.2` and `lambda_full_uvd_relative=0.1`. Both losses average only valid full-trace points or segments.

## Compatibility

- `full_uvd_num_points` defaults to zero, which preserves the existing V2 layout, state dict, outputs, and action condition.
- V3 replaces the V2 layout and does not enable full-UVD; its behavior and checkpoint contract remain unchanged.
- Existing `uvd` output and metrics continue to mean the forward local trace.
- Full-trace predictions are exposed as `uvd_full` only when enabled.

## Validation

Tests cover reverse real-frame sampling, capacity failures, fixed packing, token order and attention direction, q0-depth action conditioning, the additional objective and trainer metrics, V2/V3 backward compatibility, and a config-equivalence check against the current LIBERO q0-depth experiment.
