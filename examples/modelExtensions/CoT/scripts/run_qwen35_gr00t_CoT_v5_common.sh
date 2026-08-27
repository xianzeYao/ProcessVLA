#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29535}"
: "${CONFIG_YAML:?CONFIG_YAML must point to a V5 YAML relative to the repository root}"
: "${RUN_ID:?RUN_ID must be set by the bench wrapper}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/root/data/yxz/outputs}"

COMMAND=(
  "${PYTHON_BIN}" -m accelerate.commands.launch
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml
  --num_processes "${NUM_PROCESSES}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  starVLA/training/train_starvla_cot_v5.py
  --config_yaml "${CONFIG_YAML}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --run_id "${RUN_ID}"
)

cd "${REPO_ROOT}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}" "$@"
  printf '\n'
  exit 0
fi

OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
LOG_ROOT="${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"
cp "${CONFIG_YAML}" "${OUTPUT_DIR}/"
LOG_FILE="${LOG_ROOT}/train_$(date +%Y%m%d_%H%M%S).log"
"${COMMAND[@]}" "$@" 2>&1 | tee "${LOG_FILE}"
