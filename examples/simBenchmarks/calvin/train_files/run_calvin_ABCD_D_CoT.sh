#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
cd "$ROOT_DIR"
DATA_ROOT="${CALVIN_RERENDER_LEROBOT_ROOT:-/root/data/yxz/datasets/calvin/lerobot_rerender}"
DATASET_NAME="calvin_task_ABCD_D"
CONFIG="examples/simBenchmarks/calvin/train_files/qwen35_gr00t_calvin_ABCD_D_CoT_v1.yaml"

case "${1:-train}" in
  verify)
    test -f "$DATA_ROOT/$DATASET_NAME/meta/info.json" || { echo "Missing rerender LeRobot dataset: $DATA_ROOT/$DATASET_NAME" >&2; exit 2; }
    test -f "$DATA_ROOT/$DATASET_NAME/meta/modality.json" || { echo "Missing modality metadata." >&2; exit 2; }
    ;;
  train)
    "$0" verify
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
      --num_processes "${NUM_GPUS:-4}" starVLA/training/train_starvla_cot_v1.py \
      --config_yaml "$CONFIG" \
      --datasets.vla_data.data_root_dir "$DATA_ROOT" \
      --datasets.vla_data.dataset_name "$DATASET_NAME" \
      --run_root_dir "${RUN_ROOT_DIR:-/root/data/yxz/outputs/calvin}" \
      --run_id "${RUN_ID:-calvin_ABCD_D_CoT_v1}" "${@:2}"
    ;;
  *) echo "usage: $0 {verify|train} [extra args]" >&2; exit 2 ;;
esac
