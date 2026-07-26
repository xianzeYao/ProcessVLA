#!/usr/bin/env bash
set -euo pipefail

cd /home/yxz/CoT/CoT_vla
export WANDB_MODE="${WANDB_MODE:-offline}"

exec /root/data/yxz/miniforge3/envs/CoT/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_GPUS:-4}" \
  --main_process_port "${MAIN_PROCESS_PORT:-0}" \
  starVLA/training/train_starvla_cot_v1.py \
  --config_yaml examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v1.yaml \
  "$@"
