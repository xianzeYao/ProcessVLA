#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROCESSVLA_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROCESSVLA_ROOT}"

SCRIPT_PATH="${PROCESSVLA_ROOT}/examples/LIBERO/eval_files/auto_eval_scripts/eval_libero_parall.sh"
your_ckpt=${YOUR_CKPT:-${PROCESSVLA_ROOT}/results/Checkpoints/1226_libero4in1_qwen3oft/checkpoints/steps_50000_pytorch_model.pt}
run_index_base=${RUN_INDEX_BASE:-346}

#####################################################
task_suite_name=libero_10 # align with your model
run_index=$((run_index_base + 0))
bash "$SCRIPT_PATH" "$your_ckpt" "$task_suite_name" "$run_index" &
#####################################################

sleep 15
#####################################################
task_suite_name=libero_goal # align with your model
run_index=$((run_index_base + 1))
bash "$SCRIPT_PATH" "$your_ckpt" "$task_suite_name" "$run_index" &
#####################################################
sleep 15
#####################################################
task_suite_name=libero_object # align with your model
run_index=$((run_index_base + 2))
bash "$SCRIPT_PATH" "$your_ckpt" "$task_suite_name" "$run_index" &
#####################################################
sleep 15
####################################################
task_suite_name=libero_spatial # align with your model
run_index=$((run_index_base + 3))
bash "$SCRIPT_PATH" "$your_ckpt" "$task_suite_name" "$run_index" &
#####################################################
