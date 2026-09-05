# RoboCasa DA3 Feature Alignment Design

## Goal

Add a clean RoboCasa RQ3 `Feature Alignment Only` variant on top of the V2
q32 no-depth-condition model. Eight causal future-geometry query tokens are
supervised by a frozen Depth Anything 3 (DA3) future-image representation,
while numerical current- and future-depth reconstruction are both disabled.

The comparison target is the existing future-depth-reconstruction variant.
Both auxiliary objectives use a coefficient of `0.15`; this is a
coefficient-matched comparison rather than a claim that cosine and Smooth-L1
produce identical gradient magnitudes.

## Fixed Experiment Definition

- Benchmark and dataset: RoboCasa GR1, `fourier_gr1_unified_1000`.
- Student architecture: V2 q32 no-depth-condition.
- Action horizon: 16.
- Future target frame: `min(current_index + 16, terminal_index)`.
- Future latent token count: 8.
- Current-depth reconstruction: disabled.
- Numerical future-depth reconstruction: disabled.
- DA3 feature alignment: enabled with coefficient `0.15`.
- UVD trace objective, action head, prompt, optimizer, training length, batch
  size, and data mixture remain identical to the future-only comparison run.

## Data Flow

The RoboCasa video modality conditionally requests offsets `[0, 16]` when DA3
alignment is enabled. The existing `image` sample field continues to contain
only offset 0 and is the only RGB input visible to Qwen. A new training-only
`future_image` field contains offset 16, with the dataset's existing terminal
clamping behavior for short episode tails.

This avoids leaking the observed future image into the student. The future
image is consumed only inside a no-gradient frozen teacher path.

## DA3 Teacher Target

The teacher uses the local checkpoint:

```text
/root/data/yxz/models/DA3-LARGE-1.1
```

and the official DA3 source package supplied through a configurable source
root. Loading must fail with an actionable error if either path is missing.
The teacher is lazily created only during training, kept in evaluation mode,
and evaluated under `torch.inference_mode()` with bfloat16 autocast on CUDA.

Future RGB is resized to `224 x 224`, converted to RGB, scaled to `[0, 1]`,
and normalized with ImageNet statistics:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

For DA3-LARGE-1.1, the supervised representation is the last configured
backbone stage named layer 23. Because `cat_token=true`, the actual target is
the 2048-dimensional paired representation:

```text
concat(block-22 local feature, block-23 global feature)
```

The representation is taken from the same tensor path consumed by DualDPT and
passed through the DA3 head's input `LayerNorm`. Class/register tokens are not
included. At 224 resolution, the result is a `16 x 16` patch grid:

```text
[B, 1, 256, 2048]
```

The singleton view dimension is removed and a parameter-free
`AdaptiveAvgPool2d((2, 4))` produces eight targets in raster order:

```text
[B, 8, 2048]
```

Teacher targets are detached before loss computation.

## Student Path and Loss

The existing eight future-depth query positions are retained as eight future
latent query positions. Their causal ordering stays unchanged: they precede
the UVD tokens, so UVD tokens can attend to the aligned future representation.
With no-depth-condition enabled, the action expert continues to receive native
Qwen tokens plus UVD tokens, not the eight future latent tokens directly.

The final Qwen hidden states at the eight future positions are mapped with a
student-only projector:

```text
LayerNorm(qwen_hidden_dim)
Linear(qwen_hidden_dim, 2048)
GELU
Linear(2048, 2048)
```

The alignment objective is token-wise cosine distance:

```text
L_DA3 = mean(1 - cosine_similarity(student_token, teacher_token))
```

The total loss is:

```text
L = 1.0 * L_action + 0.62 * L_UVD + 0.15 * L_DA3
```

The existing relative-UVD term remains inside `L_UVD` with its existing
coefficient. Both numerical depth losses are exactly zero and no numerical
depth decoder is run by this variant.

## Isolation, Checkpointing, and Inference

The implementation uses a separately registered framework name and separate
RoboCasa YAML/launcher. Existing V2, UV-only, and depth-reconstruction behavior
must not change.

The frozen DA3 teacher is not registered as a trainable child of the student:

- DA3 parameters do not enter the optimizer.
- DA3 weights are not written to student checkpoints.
- Student checkpoints contain the eight latent queries and alignment
  projector.
- Rollout inference neither loads nor runs DA3.

Checkpoint loading must accept the alignment checkpoint without requiring the
DA3 source tree. Training must require the DA3 source and model paths before
the first teacher forward.

## Logging

Training logs expose at least:

```text
da3_feature_alignment_loss
weighted_da3_feature_alignment_loss
weighted_aux_loss
loss_balance/aux_to_action
```

The run manifest records the DA3 model path, source path, selected layer,
teacher feature dimension, pooling grid, image resolution, and alignment
coefficient.

## Validation and Errors

Initialization validates:

- alignment weight is non-negative;
- selected layer is exactly 23 for this experiment;
- teacher feature dimension is 2048;
- pooling grid contains exactly eight cells;
- future latent query count is eight;
- current and numerical future-depth reconstruction are disabled;
- depth features are not included directly in the action condition.

Runtime validates:

- every training example contains one future RGB image;
- teacher output is finite and has shape `[B, 256, 2048]` before pooling;
- pooled teacher and projected student shapes are both `[B, 8, 2048]`;
- DA3 remains in evaluation mode with all parameters frozen.

## Tests

Automated tests cover:

1. RoboCasa data configuration requests frames `[0, 16]` only for the DA3
   alignment variant and packs current/future RGB into separate fields.
2. Episode-tail future RGB repeats the terminal frame.
3. Fixed `16 x 16 -> 2 x 4` pooling produces eight raster-ordered targets.
4. Cosine alignment is zero for identical finite features and rejects shape
   mismatches/non-finite targets.
5. The framework retains eight latent slots while returning zero numerical
   depth losses.
6. Total loss includes exactly `0.15 * L_DA3` and existing action/UVD terms.
7. DA3 parameters are absent from trainable parameters and student state dicts.
8. Inference does not instantiate or call the DA3 teacher.
9. Existing V2 and UV-only focused regression tests remain green.

## Acceptance Criteria

- The new YAML dry-runs through the existing distributed launcher.
- A real training batch produces finite action, UVD, DA3-alignment, and total
  losses with the expected shapes and units.
- The numerical current/future depth decoder is not executed.
- A saved student checkpoint contains no DA3 checkpoint tensors.
- Existing V2 and UV-only behavior is unchanged.
