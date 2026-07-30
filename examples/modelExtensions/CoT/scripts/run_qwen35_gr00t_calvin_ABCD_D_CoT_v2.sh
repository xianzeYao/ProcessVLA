#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_YAML="examples/modelExtensions/CoT/configs/qwen35_gr00t_calvin_ABCD_D_CoT_v2.yaml"
export RUN_ID="${RUN_ID:-qwen35_gr00t_calvin_ABCD_D_CoT_v2}"
export RUN_ROOT_DIR="${RUN_ROOT_DIR:-/root/data/yxz/outputs/calvin}"
exec "${SCRIPT_DIR}/run_qwen35_gr00t_CoT_v2_common.sh" "$@"
