#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5694}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
CKPT_PATH="${CKPT_PATH:-/path/to/checkpoint.pt}"
VIDEO_OUT_PATH="${VIDEO_OUT_PATH:-results/${TASK_SUITE_NAME}/$(basename "${CKPT_PATH}")}"
USE_SIGNAL="${USE_SIGNAL:-false}"
INJECT_SIGNAL="${INJECT_SIGNAL:-false}"
SIGNAL_DUMP_ROOT="${SIGNAL_DUMP_ROOT:-}"

export PYTHONPATH="${LIBERO_HOME}:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cmd=(
  "${LIBERO_PYTHON}"
  "./examples/LIBERO/eval_files/eval_libero.py"
  "--args.pretrained-path" "${CKPT_PATH}"
  "--args.host" "${HOST}"
  "--args.port" "${PORT}"
  "--args.task-suite-name" "${TASK_SUITE_NAME}"
  "--args.num-trials-per-task" "${NUM_TRIALS_PER_TASK}"
  "--args.video-out-path" "${VIDEO_OUT_PATH}"
)

if [[ "${USE_SIGNAL}" == "true" ]]; then
  cmd+=("--args.use-signal" "true")
fi

if [[ "${INJECT_SIGNAL}" == "true" ]]; then
  cmd+=("--args.inject-signal" "true")
fi

if [[ -n "${SIGNAL_DUMP_ROOT}" ]]; then
  cmd+=("--args.signal-dump-root" "${SIGNAL_DUMP_ROOT}")
fi

"${cmd[@]}"
