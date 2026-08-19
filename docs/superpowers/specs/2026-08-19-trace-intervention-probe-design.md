# Trace Intervention Probe Design

## Goal

Determine whether the current QwenGR00TCoTV2 action velocity field causally
depends on its geometry hidden-token channel. Hold the observation, native
tokens, initial Gaussian action noise, solver schedule, checkpoint, dtype, and
batch ordering fixed while changing only the geometry condition.

This stage measures geometry-latent dependence. It does not claim that the
generated action follows the decoded numerical UVD trace, because the current
policy consumes high-dimensional hidden tokens and no validated action-to-trace
projector exists yet.

## Scope

Stage A adds three opt-in diagnostic layers:

1. The legacy flow-matching action head accepts common initial actions, exposes
   its velocity evaluation, accepts a per-Euler-step condition schedule, and can
   return detached step diagnostics.
2. QwenGR00TCoTV2 constructs matched geometry interventions from one backbone
   result and evaluates all requested variants without recomputing native
   tokens.
3. An offline runner reuses materialized paired-probe samples, records raw
   trajectories, and summarizes local and roll-forward effects.

Training, checkpoint formats, the default action condition, and the default
`predict_action()` return value are unchanged. Stage B numerical bottlenecks,
Stage C execution-consistency losses, and simulator rollout are out of scope.

## Action-head interfaces

`FlowmatchingActionHead.predict_action()` gains keyword-only diagnostics:

```python
def predict_action(
    vl_embs: torch.Tensor,
    state: torch.Tensor | None = None,
    encoder_attention_mask: torch.Tensor | None = None,
    *,
    initial_actions: torch.Tensor | None = None,
    condition_schedule: Sequence[tuple[torch.Tensor, torch.Tensor | None]] | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[FlowStepDiagnostics, ...]]:
```

`initial_actions=None`, `condition_schedule=None`, and
`return_diagnostics=False` execute the existing numerical path and return only
the action tensor. Supplied initial actions must exactly match
`[B, action_horizon, action_dim]`, device, and dtype. The head clones the input
so the caller-owned tensor is not mutated.

`FlowStepDiagnostics` contains `step_index`, `t_cont`, `t_discretized`,
`x_before`, `pred_velocity`, and `x_after`. Tensor fields are detached clones.

The integrator and local-effect computation share one public diagnostic
velocity method:

```python
def predict_velocity(
    actions: torch.Tensor,
    *,
    t_cont: float,
    vl_embs: torch.Tensor,
    state: torch.Tensor | None = None,
    encoder_attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
```

This prevents the probe from duplicating the DiT embedding and slicing logic.

## Geometry interventions

`GeometryActionCondition` records the condition tensor, mask, optional donor
permutation, and whether the condition is an out-of-training-distribution
counterfactual.

| Variant | Semantics |
|---|---|
| `correct` | Existing trained condition. |
| `native_only` | Native sequence and native mask only. |
| `zero_geometry` | Zero all trained geometry tokens while retaining their positions and mask. |
| `within_task_shuffle` | Apply one non-identity within-task donor permutation to the entire depth-current/depth-future/UVD bundle. |
| `cross_task_swap` | Apply one donor permutation for which every donor task differs from its recipient task. |
| `uvd_only` | Keep UVD and zero trained depth slots; for Q0 this equals `correct`. |
| `depth_only` | Keep depth and zero UVD for depth-conditioned models; for Q0 append depth and mark the result as a diagnostic counterfactual. |

Within-task shuffling requires at least two samples for every task present in a
batch. Cross-task swapping requires at least two task labels and is rejected
when any task occupies more than half the batch, because a cross-task
derangement is then impossible. Every permutation is recorded and validated as
non-identity. Offline batching therefore packs within-task sample pairs and may
place multiple pairs from one task in a larger batch, up to the half-batch
cross-task limit.

## Diagnostic execution

`Qwen_GR00T_CoT_V2.predict_action_interventions()` performs the following:

1. preprocess and run the Qwen geometry backbone once;
2. construct `correct` plus each requested alternative condition;
3. sample or accept one initial action tensor and reuse it for every rollout;
4. run the correct trajectory twice for the exact-repeat numerical floor and
   reject the probe if its maximum absolute action error exceeds the configured
   tolerance;
5. at every correct-path `x_k`, evaluate correct and alternative velocity to
   measure the local effect;
6. run the alternative for `all` steps and for each individual Euler step,
   returning final actions and step diagnostics.

For an intervention only at step `k`, all earlier and later steps use the
correct condition. Local effects always use the correct-path `x_k`, never a
state from an already-diverged rollout.

## Offline runner and artifacts

The runner consumes one V2 checkpoint and the materialized NPZ samples created
by the paired geometry probe. Batches contain at least two samples from each of
at least two tasks so both shuffle types are defined.

The output directory contains:

```text
config.json
summary.json
per_sample.jsonl
trajectories.npz
```

`config.json` records commit, checkpoint, sample paths and identities, seed,
batch construction, permutations, inference steps, dtype/device, action groups,
normalized-space status, the repeat tolerance, and the exact input-example
keys. The current RoboCasa training configuration uses `include_state: false`;
the recorded keys, checkpoint `include_state`/`state_dim`, and
`proprioceptive_state_present` flag make that no-state input contract explicit
even though the action-head configuration retains a nonzero compatibility
`state_dim`. A known checkpoint `include_state` value that disagrees with the
materialized example keys is rejected. `trajectories.npz` stores initial noise,
both correct-repeat actions, correct states/velocities, local alternative
velocities, and final rollout actions.

Effects are reported as both L2 norm and dimension-normalized RMS. RoboCasa
groups are left arm `[0:7]`, right arm `[7:14]`, left hand `[14:20]`, right hand
`[20:26]`, and waist `[26:29]`. Uncertainty uses episode-cluster bootstrap;
samples from the same episode are not treated as independent.

## Acceptance criteria

- Default action-head output type, shape, and numerical order remain unchanged.
- Same condition and same initial actions produce an exact repeat or a recorded
  numerical floor within an explicit tolerance.
- Different supplied initial actions normally produce different actions.
- Zero geometry preserves the trained condition shape and mask.
- Native-only mask length matches its condition length.
- Shuffle permutations are non-identity and move the whole geometry bundle.
- A single-step schedule selects the alternative only at that step.
- Local effects compare velocities at identical correct-path states.
- Batch-size-one shuffles fail explicitly.
- Diagnostic inputs, velocities, and actions are checked for NaN/Inf.
- CPU unit tests pass without a checkpoint or simulator.
