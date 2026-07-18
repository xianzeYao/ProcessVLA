#!/usr/bin/env bash
set -euo pipefail

export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29510}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-60000}"
RUN_ROOT_DIR="/root/data/yxz/outputs"
RUN_ID="qwen35_gr00t_libero_CoT_v1"
CONFIG_YAML="${REPO_ROOT}/examples/modelExtensions/CoT/configs/qwen35_gr00t_libero_CoT_v1.yaml"
OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
LOG_ROOT="${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"
cp "${BASH_SOURCE[0]}" "${OUTPUT_DIR}/"
cp "${CONFIG_YAML}" "${OUTPUT_DIR}/"

LOG_FILE="${LOG_ROOT}/train_$(date +%Y%m%d_%H%M%S).log"
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  starVLA/training/train_starvla_cot_v1.py \
  --config_yaml "${CONFIG_YAML}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  "$@" \
  2>&1 | tee "${LOG_FILE}"
