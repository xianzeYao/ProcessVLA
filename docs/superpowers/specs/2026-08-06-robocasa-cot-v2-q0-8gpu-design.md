# RoboCasa Fourier CoT V2 Q0 Eight-GPU Experiment

## Goal

Measure the effect of removing the N1.5-derived action head's 32 learned
future/query tokens from the existing RoboCasa Fourier CoT V2 experiment.

## Configuration

Create
`examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v2_q0.yaml`
as a copy of the current RoboCasa Fourier CoT V2 configuration with exactly
two intentional differences:

1. Set `run_id` to `qwen35_gr00t_robocasa_fourier_CoT_v2_q0_8gpu_bs16`.
2. Set `framework.action_model.num_target_vision_tokens` from `32` to `0`.

All other model, data, optimization, repeat, loss-weight, batch-size, and
diagnostic settings remain unchanged. In particular, the action expert stays
DiT-B with 16 layers, diffusion repeat stays 8, and the per-device batch size
stays 16.

## Launch

Reuse `run_qwen35_gr00t_CoT_v2_common.sh` with GPUs 0 through 7 and
`NUM_PROCESSES=8`. This gives a global batch size of `8 * 16 = 128` with the
existing DeepSpeed ZeRO-2 configuration. No new shell wrapper is required.

## Verification

Before training, use `DRY_RUN=1` to verify the resolved command and run a
small construction/forward smoke test confirming that zero-length future
tokens concatenate correctly. The copied YAML must differ from the source
only in the two fields listed above.
