#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash examples/modelExtensions/CoT/scripts/run_libero_fullft.sh
#   MODEL=qwen3_5 bash examples/modelExtensions/CoT/scripts/run_libero_fullft.sh

export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"

MODEL="${MODEL:-qwen3}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"

RUN_ROOT_DIR="/root/data/yxz/outputs"
LOG_ROOT="examples/modelExtensions/CoT/logs"

case "${MODEL}" in
  qwen3)
    CONFIG_YAML="examples/modelExtensions/CoT/configs/qwen3_gr00t_libero_baseline.yaml"
    RUN_ID="qwen3_gr00t_libero_baseline"
    ;;
  qwen3_5 | qwen35 | qwen3.5)
    CONFIG_YAML="examples/modelExtensions/CoT/configs/qwen35_gr00t_libero_baseline.yaml"
    RUN_ID="qwen35_gr00t_libero_baseline"
    ;;
  *)
    echo "Unknown MODEL=${MODEL}. Use MODEL=qwen3 or MODEL=qwen3_5." >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"
cp "$0" "${OUTPUT_DIR}/"

LOG_FILE="${LOG_ROOT}/${RUN_ID}_$(date +%Y%m%d_%H%M%S).log"

${PYTHON:-python} -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name QwenGR00T \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.gradient_accumulation_steps 1 \
  --trainer.learning_rate.base 3.0e-05 \
  --trainer.max_train_steps 60000 \
  --trainer.save_interval 15000 \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  2>&1 | tee "${LOG_FILE}"
