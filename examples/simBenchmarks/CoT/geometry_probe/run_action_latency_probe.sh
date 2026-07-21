#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python}"
GPU="${GPU:-0}"
PORT="${PORT:-10094}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to a 60k checkpoint}"
DATASET_ROOT="${DATASET_ROOT:-/root/data/yxz/datasets/libero_rerender}"
OUTPUT_DIR="${OUTPUT_DIR:?set a unique output directory}"
NUM_SAMPLES="${NUM_SAMPLES:-30}"
SEED="${SEED:-7}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchvision_av}"
VIDEO_THREAD_COUNT="${VIDEO_THREAD_COUNT:-1}"
SUITES="${SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"
SERVER_MODULE="${SERVER_MODULE:-deployment.model_server.server_policy}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
SERVER_LOG="${OUTPUT_DIR}/policy_server.log"
mkdir -p "${OUTPUT_DIR}"

echo "[action-latency] checkpoint=${CHECKPOINT}"
echo "[action-latency] gpu=${GPU} port=${PORT} samples=${NUM_SAMPLES} seed=${SEED}"
echo "[action-latency] output=${OUTPUT_DIR}"

DEBUG= CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m "${SERVER_MODULE}" \
  --ckpt_path "${CHECKPOINT}" \
  --port "${PORT}" \
  --idle_timeout 3600 >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"${PYTHON_BIN}" - "${PORT}" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
deadline = time.time() + 300
while time.time() < deadline:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            print(f"[action-latency] server ready on {port}")
            raise SystemExit(0)
    except OSError:
        time.sleep(1)
raise SystemExit(f"server did not become ready on {port}")
PY

PYTHONPATH="${ROOT_DIR}" "${PYTHON_BIN}" -m \
  examples.simBenchmarks.CoT.geometry_probe.run_action_latency_probe \
  --dataset-root "${DATASET_ROOT}" \
  --suites "${SUITES}" \
  --num-samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  --video-backend "${VIDEO_BACKEND}" \
  --video-thread-count "${VIDEO_THREAD_COUNT}" \
  --output-dir "${OUTPUT_DIR}" \
  --host 127.0.0.1 \
  --port "${PORT}"
