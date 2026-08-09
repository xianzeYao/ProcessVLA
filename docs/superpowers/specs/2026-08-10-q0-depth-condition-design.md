# Q0 Depth-Condition Experiment Design

## Objective

Test whether the supervised current-depth and future-depth latent tokens improve
action success when they are exposed directly to the existing Q0 action head.
This experiment must isolate condition content from action-head architecture.

## Fixed baseline

The experiment keeps the RoboCasa Fourier CoT v2 Q0 baseline unchanged except
for the action-condition token sequence:

- Qwen3.5-4B remains the VLM and runs its existing 32-layer path.
- Current-depth, future-depth, and UVD geometry tokens and losses are unchanged.
- The action head has zero learned future/action query tokens.
- The action DiT remains 16 layers wide 768 (`DiT-B`).
- `repeated_diffusion_steps` remains 8.
- The regular interleaved self/cross DiT remains in use; AlternateVLDiT, VLLN,
  condition adapters, and capacity changes are out of scope.

## Configuration interface

Add a backward-compatible boolean under `framework.geometry`:

```yaml
include_depth_in_action_condition: true
```

The code default is `false`, so existing checkpoints and YAML files retain the
current Q0 behavior. Create a new RoboCasa YAML for this experiment rather than
changing the existing Q0 baseline YAML. The new YAML receives a distinct
`run_id` and sets the flag to `true`.

## Condition construction

With the flag disabled, preserve the current sequence exactly:

```text
[native image/language tokens, UVD tokens]
```

With the flag enabled, construct:

```text
[native image/language tokens,
 current-depth tokens,
 future-depth tokens,
 UVD tokens]
```

The depth tokens are Qwen hidden states from the existing supervised geometry
slots, not decoded depth maps and not ground-truth depth. Training and inference
therefore use the same predicted latent condition and introduce no teacher
forcing or ground-truth leakage.

The action-condition attention mask appends valid entries for all fixed depth
and UVD slots while preserving the native padding mask. Training and inference
must both call the same condition-building helper.

## Compatibility and failure handling

- Reject a non-boolean value for the new option with a clear configuration
  error.
- Preserve the exact old token order, shapes, and masks when the option is
  absent or false.
- Validate that the constructed condition length equals native tokens plus the
  enabled fixed geometry-token counts.
- Do not change checkpoint parameter names because this experiment adds no
  trainable modules.

## Verification

Add focused tests that verify:

1. The default/false path remains `[native, UVD]` with the original mask.
2. The true path is ordered `[native, current depth, future depth, UVD]`.
3. Mask lengths and values match both condition layouts.
4. Training and inference use the same configured layout.
5. The new YAML resolves to Q0, 16-by-768 DiT-B, repeat 8, zero action queries,
   and depth conditioning enabled.

Run the focused unit tests plus a lightweight model/config smoke test. Provide
an exact eight-GPU launch command for the new YAML after verification.

## Success criterion

Compare this run directly against the completed RoboCasa Q0 baseline. The main
criterion is downstream task success; depth/UVD diagnostics must not regress in
a way that indicates the action loss is destabilizing the geometry hierarchy.

