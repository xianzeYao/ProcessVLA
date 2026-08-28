#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_YAML="examples/modelExtensions/CoT/configs/qwen35_gr00t_libero_CoT_v2_q0_depthcond_agentview.yaml"
export RUN_ID="${RUN_ID:-qwen35_gr00t_libero_CoT_v2_q0_depthcond_agentview_8gpu_bs16}"
export RUN_ROOT_DIR="${RUN_ROOT_DIR:-/root/data/yxz/outputs}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
exec "${SCRIPT_DIR}/run_qwen35_gr00t_CoT_v2_common.sh" "$@"
