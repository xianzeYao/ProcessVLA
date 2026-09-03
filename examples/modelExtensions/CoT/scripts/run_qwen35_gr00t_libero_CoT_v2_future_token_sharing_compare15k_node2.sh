#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_qwen35_gr00t_libero_CoT_v2_q32_nodepthcond_wristdepth_futureonly_shared_gradprobe15k_node2.sh" "$@"
bash "${SCRIPT_DIR}/run_qwen35_gr00t_libero_CoT_v2_q32_nodepthcond_wristdepth_futureonly_separate_gradprobe15k_node2.sh" "$@"
