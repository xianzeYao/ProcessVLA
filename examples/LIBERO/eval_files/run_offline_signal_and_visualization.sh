#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
VIDEO_PATH="${VIDEO_PATH:-/path/to/rollout.mp4}"
INSTRUCTION="${INSTRUCTION:-}"
HIDDEN_DIR="${HIDDEN_DIR:-}"
SIGNAL_NAMES="${SIGNAL_NAMES:-signal}"
CURVE_PATHS="${CURVE_PATHS:-}"

cmd=(
  "${PYTHON_BIN}"
  "examples/LIBERO/eval_files/offline_signal_benchmark.py"
  "--video-path" "${VIDEO_PATH}"
)

if [[ -n "${INSTRUCTION}" ]]; then
  cmd+=("--instruction" "${INSTRUCTION}")
fi

if [[ -n "${HIDDEN_DIR}" ]]; then
  cmd+=("--hidden-dir" "${HIDDEN_DIR}")
fi

if [[ -n "${CURVE_PATHS}" ]]; then
  # shellcheck disable=SC2206
  curve_array=(${CURVE_PATHS})
  cmd+=("--curve-paths" "${curve_array[@]}")
else
  # shellcheck disable=SC2206
  signal_array=(${SIGNAL_NAMES})
  cmd+=("--signal-names" "${signal_array[@]}")
fi

"${cmd[@]}"
