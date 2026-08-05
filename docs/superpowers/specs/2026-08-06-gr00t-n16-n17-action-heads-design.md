# GR00T N1.6 and N1.7 Action Heads Design

## Goal

Add faithful, configurable GR00T N1.6 and N1.7 action heads alongside the existing StarVLA N1.5-derived head. Existing checkpoints and call paths must remain unchanged.

## Scope

- Add `GR00T_N16_ActionHeader.py` and `GR00T_N17_ActionHeader.py`.
- Extend the shared flow-matching DiT module with additive `AlternateVLDiT` support.
- Reuse the existing timestep, attention, MLP, and embodiment-conditioned components where their behavior matches the official implementation.
- Add focused CPU tests using small dimensions and shallow DiTs.
- Do not change existing framework wrappers or their `repeated_diffusion_steps` behavior in this change.

## Public Interface

Both heads retain the StarVLA-style leading arguments and add official conditioning inputs:

```python
forward(
    vl_embs,
    actions,
    state=None,
    encoder_attention_mask=None,
    image_mask=None,
    embodiment_id=None,
    action_mask=None,
)
```

- `embodiment_id` defaults to category zero for single-embodiment use.
- `image_mask` is required by `AlternateVLDiT`; missing masks raise a clear error.
- `action_mask` defaults to all ones. When supplied, padded action dimensions do not contribute to the loss.
- Input action width must equal configured `action_dim`. Padding is explicit and never silently inferred.
- `predict_action` accepts the same condition arguments except training-only `actions` and `action_mask`.

## Shared Flow-Matching Behavior

Each input item samples one independent Gaussian noise tensor and one independent flow time per action-head `forward` call. The head does not repeat the batch internally. Multi-noise sampling remains an outer-wrapper policy controlled by `repeated_diffusion_steps`:

- Official N1.6/N1.7 behavior: `repeated_diffusion_steps: 1`.
- StarVLA/CogACT-style ablation: any larger configurable value, such as 8.

The training path remains:

```text
noise, t -> noisy_action = (1 - t) * noise + t * action
target_velocity = action - noise
```

Inference starts from Gaussian noise and performs configurable Euler steps from flow time zero toward one.

## Tokenization and Multi-Embodiment Projection

The new heads remove N1.5 future tokens. Their action-side sequence is:

```text
[one state token, H noisy-action tokens]
```

- State is projected with a category-specific two-layer MLP.
- Noisy actions and the discretized flow time are fused by a category-specific three-layer action encoder.
- Learned absolute position embeddings are added only to action tokens.
- DiT outputs are decoded by a category-specific two-layer MLP.
- Official padded widths are supported through config plus `action_mask`; single-robot heads may configure their native action width directly.

## AlternateVLDiT

`AlternateVLDiT` is added without changing the existing `DiT` class or old checkpoint behavior. With `attend_text_every_n_blocks=2`, blocks repeat this schedule:

```text
non-image/text cross-attention
full action-side self-attention
image cross-attention
full action-side self-attention
```

Every block keeps the existing pre-norm structure:

```text
AdaLayerNorm(flow time) -> attention -> residual
LayerNorm -> FFN -> residual
```

Cross-attention uses the VLM padding mask intersected with either the image mask or its non-image complement. Self-attention is full and non-causal over the state/action sequence.

## Version Differences

### N1.6

- Defaults model the official released family: 1536-wide AlternateVLDiT, 32 heads by 48 dimensions, 32 blocks, four Euler inference steps.
- VLM features receive a lightweight LayerNorm before DiT conditioning.
- State history defaults to one step.

### N1.7

- Reuses the N1.6 state/action encoding, flow matching, AlternateVLDiT, and decoding path.
- Supports flattened multi-step state history in its state encoder.
- Adds an optional VLM refinement stack. Official base-style defaults use four full self-attention blocks at width 2048 before AlternateVLDiT.
- Keeps a compatible inference surface for later RTC overlap/inpainting support. RTC behavior itself is outside this change; tests cover only the base Euler inference path.

All official-size defaults remain overridable so tests and smaller StarVLA experiments can instantiate compact models.

## Error Handling

- Validate action horizon and action width before encoding.
- Validate state width after applying configured history length.
- Validate `image_mask`, VLM padding mask, and VLM sequence lengths.
- Validate embodiment IDs against the configured category count.
- Reject an all-zero `action_mask` instead of dividing by an empty target count.

## Testing

Tests use tiny widths, two or four DiT blocks, and short token sequences. They cover:

1. N1.6 constructs exactly one state token plus `H` action tokens and has no future-token parameter.
2. One head forward samples exactly one noise/time item per input batch item; the head performs no internal batch repeat.
3. Masked action dimensions do not affect loss.
4. Missing or shape-incompatible image masks fail clearly.
5. AlternateVLDiT alternates non-image cross, full self, image cross, full self masks.
6. N1.7 applies its optional VLM refinement stack while N1.6 does not.
7. Both heads run short Euler inference and return `[B, H, action_dim]`.
8. Existing StarVLA `FlowmatchingActionHead` construction and tests remain unaffected.

## Non-Goals

- Loading NVIDIA checkpoint weights directly into StarVLA naming conventions.
- Changing Qwen/Cosmos framework wrappers to select the new heads automatically.
- Changing existing N1.5 repeat defaults.
- Reproducing NVIDIA data preprocessing, normalization statistics, or full training recipes.
