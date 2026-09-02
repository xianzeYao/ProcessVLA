#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_YAML="examples/modelExtensions/CoT/configs/qwen35_gr00t_libero_CoT_v2_q32_nodepthcond_wristdepth_60k_node2.yaml"
export RUN_ID="${RUN_ID:-qwen35_gr00t_libero_CoT_v2_q32_nodepthcond_wristdepth_60k_node2}"
export RUN_ROOT_DIR="${RUN_ROOT_DIR:-/data-training/yyf/yxz/outputs/libero}"
export PYTHON_BIN="${PYTHON_BIN:-/data-training/yyf/yxz/envs/train/bin/python}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29543}"
exec "${SCRIPT_DIR}/run_qwen35_gr00t_CoT_v2_common.sh" "$@"
