# CoT V2 Conditioning, Decoder, and Diagnostics Design

## Goal and Research Priority

CoT V2 prioritizes a clean, testable geometric-reasoning structure over adding
every available geometry feature directly to the policy. UVD trajectory tokens
are the only explicit geometric CoT representation exposed to the Action
Expert. Dense current/future depth remains auxiliary supervision that must shape
the Qwen token computation rather than become a direct policy bypass.

## Inference and Gradient Paths

The Qwen sequence is:

```text
[V, L] -> D_current -> D_future -> UVD
```

The Action Expert condition is fixed to:

```text
[V, L, UVD_final]
```

`D_current_final` and `D_future_final` are not concatenated into the Action
Expert condition. UVD tokens can read preceding depth-token states inside Qwen,
so depth information can influence action only after being compressed into the
UVD reasoning state. V/L remains a direct semantic and visual policy path, so
this is a soft geometric bottleneck rather than a hard information bottleneck.

No UVD intervention or training ablation is required for the first engineering
validation run. Zero/shuffle and latent-only controls remain deferred scientific
experiments.

## Token Layout

Current and future depth each use eight independently initialized learnable
tokens:

```text
D_current: [B, 8, H]
D_future:  [B, 8, H]
```

Eight is the main V2 setting. It preserves comparability with the existing
design while the insertion mechanism changes from the V1 query module to direct
Qwen tokens. The implementation does not add an automatic 1/4/8 sweep.

UVD token count remains benchmark/horizon dependent and time-major. Each UVD
token has one normalized time embedding and, for dual-arm data, one hand
embedding. Same-time hands can interact; different times retain temporal causal
order.

## Depth Readout

The old readout averaged the eight final depth-token states. Replace that mean
with one shared attention-pooling module:

\[
s_i=w^T\operatorname{LN}(h_i),\qquad
\alpha_i=\operatorname{softmax}(s_i),\qquad
q=\sum_i\alpha_i h_i.
\]

The same LayerNorm and scoring projection are used for current and future depth.
The pool returns both the summary `[B,H]` and weights `[B,8]`. It has no
current/future-specific parameters and introduces no token-to-token reasoning.

The pooled summary FiLM-modulates the existing shared ConvStack depth decoder:

```text
Qwen final main-image patch features F
       +
shared attention-pooled depth summary q
       |
Shared FiLM ConvStack
       |
224 x 224 metric depth
```

Current and future calls share the complete pooling and decoder weights. Their
only distinction is the Qwen depth-token states. The decoder remains training
only for the normal action path.

## UVD Numeric Readout

Keep the existing shared pointwise two-layer MLP:

```text
one final UVD token -> shared MLP -> one absolute (u,v,d) point
```

The MLP does not communicate between tokens. Temporal reasoning must occur in
Qwen. U/V remain sigmoid-constrained to `[0,1]`; depth remains positive through
softplus. The output is supervised by all-point absolute Smooth L1 and adjacent
same-hand delta Smooth L1. No autoregressive, temporal-convolution, residual
integration, or Transformer decoder is added.

## Objective Schedule

Use fixed weights from step zero:

\[
L=L_a+0.14L_{dc}+0.15L_{df}
  +0.62L_{u,abs}+0.062L_{u,rel}.
\]

The normal learning-rate warmup already scales early optimizer updates. No
auxiliary-specific ramp-up or decay is introduced. A schedule can be reconsidered
only if checkpoint gradient probes show a repeatable stage-specific conflict.

## Lightweight Online Diagnostics

### Attention pooling utilization

At the existing diagnostics interval, record separately for current and future:

- raw natural-log attention entropy and normalized entropy `H / log(8)` in `[0,1]`;
- maximum attention weight;
- effective token count `exp(H)` in `[1,8]`, using the raw entropy `H`;
- depth-token mean off-diagonal pairwise cosine;
- depth-token covariance effective rank.

Effective rank is `exp(H(p))`, where `p` is the normalized FP32 eigenvalue
distribution of the centered token covariance. Degenerate all-zero variance is
recorded as rank zero, not NaN.

### Decoder reliance

On the existing fixed diagnostic examples, decode four variants without an
optimizer update:

1. correct current/future pooled summaries;
2. zero summary;
3. current and future summaries swapped;
4. summaries cyclically shuffled across samples when batch size is greater than
   one.

Report the change from normal prediction for each intervention and its target
error. A decoder whose future output is unchanged after zero/swap/shuffle is
likely relying primarily on image patches rather than depth tokens.

Decoder-reliance runs only at the configured geometry diagnostics interval. It
is not executed in normal training steps and does not add backward passes.

### UVD representation utilization

At the same interval, record:

- per-time-index U/V pixel MAE and depth MAE;
- per-time-index valid ratio;
- UVD-token mean off-diagonal pairwise cosine;
- UVD-token covariance effective rank;
- existing start/end, path-length, and adjacent-motion metrics.

Dual-arm metrics preserve time-major order and compare same-hand trajectories.

### Gradient clipping

Reuse the norm already returned by `clip_grad_norm_` to record:

- `train/grad_norm_pre_clip`;
- `train/grad_clip_threshold`;
- `train/grad_clip_triggered`;
- `train/grad_clip_scale`.

No additional hook or backward pass is installed for these four fields.

## Offline Gradient Probe

The independently approved offline-probe design remains authoritative:

`design/v2/2026-07-30-offline-gradient-probe-design.md`.

It measures per-loss norms and action cosine on fixed samples and selected shared
parameters. The online representation/decoder diagnostics and the offline
gradient probe are complementary and must remain separate code paths.

## Configuration and Checkpoint Compatibility

The attention pool is a new trainable V2 module. New V2 checkpoints include its
parameters. Strictly loading an older mean-pooling V2 checkpoint into the new
architecture must fail with an explicit incompatibility message; silently using
random pool parameters is forbidden. V1 and baseline checkpoint behavior is not
changed.

All new diagnostics are controlled by the existing `trainer.test_diagnostics`
gate. Subfeatures receive explicit booleans and default to disabled when absent,
so ordinary training pays no intervention cost. Scalar loss logging remains
unchanged.

## Validation Requirements

Tests must establish:

1. shared attention pooling returns exact shapes and normalized weights;
2. current/future calls use the same pool parameters;
3. nonuniform token states can receive nonuniform weights and gradients;
4. zero/swap/shuffle decoder interventions use the intended summaries without
   changing model parameters;
5. entropy, effective token count, cosine, and effective-rank calculations are
   finite on normal and degenerate toy tensors;
6. per-time UVD metrics preserve single- and dual-arm time-major ordering;
7. disabled diagnostics do not add decoder intervention calls;
8. clipping metrics match hand-computed clipped and unclipped cases;
9. V1 and baseline focused regression tests continue to pass.

A real V2 GPU smoke must verify forward/backward shapes and one diagnostic pass.
It establishes implementation behavior, not policy improvement or scientific
support.

## Deferred Evidence

The following are explicitly outside the first implementation:

- UVD condition zero/shuffle benchmark evaluation;
- latent-only V2 training with UVD supervision disabled;
- depth-supervision ablation;
- depth-token count sweep;
- dynamic auxiliary-loss schedules;
- DPT, multi-scale, temporal, or autoregressive decoders.
