#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python}"
CALVIN_PYTHON="${CALVIN_PYTHON:-/root/data/yxz/miniforge3/envs/calvin/bin/python}"
RERENDER_ROOT="${CALVIN_RERENDER_ROOT:-/root/data/yxz/datasets/calvin/rerender_ABC_D}"
DATA_ROOT="${CALVIN_RERENDER_LEROBOT_ROOT:-/root/data/yxz/datasets/calvin/lerobot_rerender}"
DATASET_NAME="calvin_task_ABC_D"
# Default to the persisted, real-frame ABC_D validation slice. Set
# CALVIN_RAW_ROOT to the complete official task_ABC_D root for full training.
RAW_ROOT="${CALVIN_RAW_ROOT:-/root/data/yxz/datasets/calvin/task_ABC_D}"
CONFIG="examples/modelExtensions/CoT/configs/qwen35_gr00t_calvin_ABC_D_baseline.yaml"

case "${1:-train}" in
  prepare)
    if ! test -f "$RAW_ROOT/training/ep_start_end_ids.npy"; then
      test -f "$RAW_ROOT.zip" || { echo "missing official archive: $RAW_ROOT.zip" >&2; exit 2; }
      unzip -q "$RAW_ROOT.zip" -d "$(dirname "$RAW_ROOT")"
    fi
    ;;
  raw_verify)
    "$CALVIN_PYTHON" examples/simBenchmarks/calvin/data_preparation/verify_calvin_raw.py "$RAW_ROOT" --split training --require-config --allow-extra-frames
    ;;
  rerender)
    test -d "$RAW_ROOT/training" || { echo "missing merged raw dataset: $RAW_ROOT" >&2; exit 2; }
    "$0" raw_verify
    if test "${CALVIN_RERENDER_WORKERS:-4}" -gt 1 && test "${MAX_SEGMENTS:-0}" -eq 0 && test "${START_SEGMENT:-0}" -eq 0; then
      CALVIN_RERENDER_GPUS="${CUDA_VISIBLE_DEVICES:-4,5,6,7}" \
        CALVIN_RERENDER_WORKERS="${CALVIN_RERENDER_WORKERS:-4}" \
        bash examples/simBenchmarks/calvin/data_preparation/run_calvin_rerender_parallel.sh "$RAW_ROOT" "$RERENDER_ROOT"
    else
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}" "$CALVIN_PYTHON" -m \
        examples.simBenchmarks.calvin.data_preparation.rerender_calvin_segments \
        --dataset-root "$RAW_ROOT" --output-root "$RERENDER_ROOT" \
        --max-segments "${MAX_SEGMENTS:-0}" --start-segment "${START_SEGMENT:-0}" \
        --source-split training --config-split training --overwrite --no-comparison "${@:2}"
    fi
    ;;
  convert)
    "$0" raw_verify
    test -d "$RERENDER_ROOT" || { echo "missing rerender dataset: $RERENDER_ROOT; run '$0 rerender' first" >&2; exit 2; }
    "$CALVIN_PYTHON" -m examples.simBenchmarks.calvin.data_preparation.build_calvin_rerender_lerobot \
      --dataset-root "$RAW_ROOT" --rerender-root "$RERENDER_ROOT" \
      --output-root "$DATA_ROOT/$DATASET_NAME" --overwrite "${@:2}"
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
      --run_id "${RUN_ID:-calvin_ABC_D_baseline}" "${@:2}"
    ;;
  *) echo "usage: $0 {prepare|raw_verify|rerender|convert|verify|train} [extra args]" >&2; exit 2 ;;
esac
