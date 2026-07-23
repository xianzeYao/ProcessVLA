#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/root/data/yxz/miniforge3/envs/CoT/bin/python}"
DATA_ROOT="${CALVIN_LEROBOT_ROOT:-/root/data/yxz/datasets/calvin_lerobot}"
DATASET_NAME="calvin_task_ABCD_D"
# The locally available official subset023 is a complete, metadata-backed
# ABCD_D shard. Set CALVIN_RAW_ROOT to a merged full task_ABCD_D root when it
# is available.
RAW_ROOT="${CALVIN_RAW_ROOT:-/root/data/yxz/datasets/calvin_hf/task_ABCD_D}"
CONFIG="examples/simBenchmarks/calvin/train_files/qwen35_gr00t_calvin_ABCD_D_baseline.yaml"

case "${1:-train}" in
  convert)
    "$PYTHON_BIN" -m examples.simBenchmarks.calvin.data_preparation.build_calvin_relative_lerobot \
      --dataset-root "$RAW_ROOT" --output-root "$DATA_ROOT/$DATASET_NAME" \
      --action-key rel_actions --overwrite "${@:2}"
    ;;
  verify)
    "$PYTHON_BIN" examples/simBenchmarks/calvin/data_preparation/verify_calvin_lerobot.py "$DATA_ROOT/$DATASET_NAME"
    ;;
  train)
    test -f "$DATA_ROOT/$DATASET_NAME/meta/info.json" || { echo "Run '$0 convert' first." >&2; exit 2; }
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
      --num_processes "${NUM_GPUS:-4}" starVLA/training/train_starvla.py \
      --config_yaml "$CONFIG" \
      --datasets.vla_data.data_root_dir "$DATA_ROOT" \
      --datasets.vla_data.dataset_name "$DATASET_NAME" \
      --run_root_dir "${RUN_ROOT_DIR:-/root/data/yxz/outputs/calvin}" \
      --run_id "${RUN_ID:-calvin_ABCD_D_baseline}" "${@:2}"
    ;;
  *) echo "usage: $0 {convert|verify|train} [extra args]" >&2; exit 2 ;;
esac
