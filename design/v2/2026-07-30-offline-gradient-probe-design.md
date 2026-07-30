# CoT V2 Offline Gradient Probe Design

## Goal

Measure how each CoT V2 objective acts on shared representation parameters, so
loss weights are selected from gradient magnitude and direction rather than raw
loss scale. The probe must not alter the normal training backward path or slow
every training step.

## Decision

Use a hybrid diagnostic design:

1. Keep low-cost online logging for raw and weighted losses, the combined module
   gradient norm, and prediction metrics.
2. Add a standalone offline probe that loads a checkpoint and evaluates fixed
   dataloader batches without optimizer updates.
3. Add only cheap clipping observability to the online trainer: the pre-clipping
   norm already returned by `clip_grad_norm_`, plus a clipping-trigger flag and
   scale.

Rejected alternatives:

- Per-loss gradients every training step: requires repeated backward traversals
  through Qwen and the Action Expert, materially increasing time, memory, and
  distributed communication.
- Raw-loss-only calibration: loss values do not determine parameter-update
  magnitude or reveal direction conflicts.
- Offline-only diagnostics: checkpoint probes explain attribution but do not
  show continuous loss, clipping, or prediction trends.

## Probe Inputs and Reproducibility

The command accepts:

- `--config_yaml`: the complete V2 experiment YAML used to construct the model
  and benchmark dataloader;
- `--checkpoint`: an exact `.pt` or `.safetensors` model checkpoint;
- `--output`: JSON output path;
- `--num_batches`, default `8`;
- `--batch_size`, default `1`;
- `--seed`, default `42`;
- `--qwen_tail_layers`, default `2`;
- `--sample_indices`, an optional comma-separated index list that overrides
  seeded sampling.

The probe sets dataset workers to zero. By default it samples dataset indices
uniformly without replacement using `seed`, instead of taking only the first
episodes. It records the selected indices so the exact sample set can be reused
with `--sample_indices` at every checkpoint. This is fixed and reproducible, but
is not called a validation set: it diagnoses the training objective on
training-distribution samples and never updates parameters.

Provenance records the resolved checkpoint path, file size and modification
time, config SHA-256, sample count, seed, parameter names/counts, dtype, device,
and Git revision. It does not hash the full multi-gigabyte checkpoint. LIBERO,
CALVIN, and RoboCasa use the same command and their existing V2 YAML.

The model remains in training mode to preserve the training computation path.
Each batch uses one forward pass, so action diffusion noise and dropout
realizations are shared by all per-loss gradient measurements from that batch.
No optimizer or scheduler is constructed and no parameter update occurs.
Before each batch forward, the random generators are reset from `seed` and the
batch ordinal so repeated checkpoint probes use the same stochastic realization.

## Objectives and Effective Weights

The probe reads these scalar outputs from `QwenGR00TCoTV2.forward`:

| Name | Raw objective | Effective weight |
|---|---|---:|
| `action` | `action_loss` | `lambda_action` |
| `depth_current` | `depth_current_loss` | `lambda_depth_current` |
| `depth_future` | `depth_future_loss` | `lambda_depth_future` |
| `uvd_absolute` | `uvd_absolute_loss` | `lambda_uvd` |
| `uvd_relative` | `uvd_relative_loss` | `lambda_uvd * lambda_uvd_relative` |

`uvd_loss` is not measured as an additional objective because it already equals
`uvd_absolute + lambda_uvd_relative * uvd_relative`; including both would double
count UVD supervision.

## Shared Parameter Scope

The default scope contains only parameters where action and auxiliary objectives
can interact:

- all trainable parameters in `geometry_tokens`;
- trainable parameters in the final two Qwen language-model transformer layers.

Depth decoder, UVD numeric head, and Action Expert private parameters are
excluded from cosine computation. Their gradients measure head fitting rather
than competition over shared representation. The number of Qwen tail layers is
configurable; `0` means geometry tokens only. A later full-Qwen mode is deferred
because retaining several full-model gradient vectors is unnecessarily costly
for the first diagnostic version.

The script fails clearly if the requested Qwen layer path does not exist, the
selected scope is empty, or the action gradient is zero over the entire scope.

## Gradient Computation

For a single forward result, use `torch.autograd.grad` on the selected shared
parameters. It does not populate `.grad` and therefore cannot accidentally
perform an optimization step.

For each parameter group and objective, accumulate in FP32:

\[
\lVert g_i\rVert_2,
\qquad
\lVert \lambda_i g_i\rVert_2,
\qquad
\cos(g_i,g_a)=
\frac{g_i^\top g_a}{\lVert g_i\rVert_2\lVert g_a\rVert_2}.
\]

Unused gradients are treated as zero for the same selected parameter vector.
The implementation retains the action gradient and an accumulated weighted
auxiliary gradient, processes each auxiliary objective sequentially, and does
not retain five complete gradient copies simultaneously.

It also reports:

\[
g_{aux}=\sum_i\lambda_i g_i,
\quad
\frac{\lVert g_{aux}\rVert_2}{\lVert\lambda_a g_a\rVert_2},
\quad
\cos(g_{aux},g_a).
\]

Statistics are calculated separately for `geometry_tokens`, `qwen_tail`, and
their union `shared_total`.

## Outputs

The JSON contains:

- one record per batch with raw losses, weighted losses, raw/weighted gradient
  norms, auxiliary-to-action norm ratios, and cosine with action;
- combined auxiliary norm, ratio, and cosine;
- mean, standard deviation, median, minimum, and maximum over batches;
- explicit finite/nonzero flags and count of valid measurements;
- run provenance and selected-parameter metadata.

The probe prints a compact terminal table but JSON is the source of truth for
later comparison across checkpoints.

## Online Additions

Normal training keeps its current raw/weighted loss and periodic combined-module
gradient logs. When gradient clipping is configured, add:

- `train/grad_norm_pre_clip`;
- `train/grad_clip_threshold`;
- `train/grad_clip_triggered`;
- `train/grad_clip_scale = min(1, threshold / max(pre_clip_norm, eps))`.

These fields reuse the norm already returned by `clip_grad_norm_`; they do not
install extra hooks or run another backward pass.

## Failure Handling

The standalone command exits nonzero for missing checkpoints, non-V2 objective
keys, empty parameter scopes, non-finite losses/gradients, and zero action
gradient across all measured batches. A zero gradient for one auxiliary on one
batch is recorded rather than treated as fatal because validity masks can make
an objective inactive.

No model, optimizer, scheduler, dataset, or checkpoint file is modified. Output
is written only to the requested probe JSON path.

## Verification

Unit tests use a small differentiable toy model to verify:

1. exact L2 norms and cosine signs for aligned, orthogonal, and conflicting
   gradients;
2. effective nested UVD-relative weighting;
3. unused gradients are represented as zeros;
4. combined auxiliary statistics equal a hand-computed vector sum;
5. online clipping metrics distinguish clipped and unclipped steps;
6. the CLI rejects missing objective keys and empty parameter scopes.

A lightweight framework test verifies parameter selection against a synthetic
V2-shaped module tree. Real-checkpoint execution remains an experiment smoke
test, not a unit-test requirement.

## Scientific Interpretation Boundary

Passing tests establishes that the probe computes the intended quantities. It
does not establish that current weights are optimal or that auxiliary geometry
improves policy success. Weight decisions require repeated measurements over
representative fixed batches and at multiple checkpoints, followed by a short
controlled training comparison.
