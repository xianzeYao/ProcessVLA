# V2 Q32 + Depth Training Config Design

## Goal

Create clean, from-scratch Q32 counterparts to the validated V2 Q0 + depth
experiments for LIBERO and RoboCasa.

## Configuration

Add two YAML files derived from the existing Q0 + depth configurations:

- `qwen35_gr00t_libero_CoT_v2_q32_depthcond.yaml`
- `qwen35_gr00t_robocasa_fourier_CoT_v2_q32_depthcond.yaml`

Each new file changes only:

1. `run_id`, so outputs cannot overwrite the Q0 run.
2. `framework.action_model.num_target_vision_tokens`, from `0` to `32`.

The files retain `geometry.include_depth_in_action_condition: true`. Dataset,
loss, optimizer, batch-size, DiT, repeat, horizon, and training-step settings
remain identical to their corresponding Q0 + depth experiments.

## Launch and safety

Both runs use the existing eight-GPU V2 common launcher through an explicit
`CONFIG_YAML` environment override. They start from scratch: no checkpoint or
resume option is supplied. Before launch, the operator should ensure the new
output directories do not already contain a run they want to preserve.

## Verification

Verify that each new YAML:

- parses successfully;
- has `num_target_vision_tokens: 32`;
- has `include_depth_in_action_condition: true`;
- differs from the matching Q0 + depth YAML only in `run_id` and query count;
- resolves to a unique output directory through the existing launcher.
