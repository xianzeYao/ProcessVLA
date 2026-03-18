#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
CKPT_PATH="${CKPT_PATH:-/path/to/checkpoint.pt}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-5694}"
USE_BF16="${USE_BF16:-true}"

cmd=(
  "${STARVLA_PYTHON}"
  "deployment/model_server/server_policy.py"
  "--ckpt_path" "${CKPT_PATH}"
  "--port" "${PORT}"
)

if [[ "${USE_BF16}" == "true" ]]; then
  cmd+=("--use_bf16")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"
