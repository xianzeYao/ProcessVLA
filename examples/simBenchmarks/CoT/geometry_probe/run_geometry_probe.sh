#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

BENCH="${BENCH:-libero}"
GT_MODE="${GT_MODE:-dataset}"
PYTHON_BIN="${PYTHON_BIN:-/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python}"
if [[ -z "${LIBERO_HOME:-}" ]]; then
  if [[ "${BENCH}" == "libero_plus" ]]; then
    LIBERO_HOME="/root/data/yxz/benchmarks/LIBERO-plus"
  else
    LIBERO_HOME="/root/data/yxz/benchmarks/LIBERO"
  fi
fi
SIM_PYTHON="${SIM_PYTHON:-/root/data/yxz/miniforge3/envs/libero/bin/python}"
if [[ "${BENCH}" == "libero_plus" && -z "${SIM_PYTHON_OVERRIDE:-}" ]]; then
  SIM_PYTHON="/root/data/yxz/miniforge3/envs/libero_plus/bin/python"
fi
GPU="${GPU:-0}"
PORT="${PORT:-10093}"
CHECKPOINT="${CHECKPOINT:-/root/data/yxz/outputs/qwen35_gr00t_libero_CoT_v1/checkpoints/steps_60000_pytorch_model.pt}"
DATASET_ROOT="${DATASET_ROOT:-/root/data/yxz/datasets/libero_rerender}"
HDF5_ROOT="${HDF5_ROOT:-/root/data/yxz/datasets/libero_original_hdf5}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/yxz/outputs/qwen35_gr00t_libero_CoT_v1_geometry_probe}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
SEED="${SEED:-7}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchvision_av}"
VIDEO_THREAD_COUNT="${VIDEO_THREAD_COUNT:-1}"
SUITES="${SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"
DRY_RUN="${DRY_RUN:-0}"

export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${ROOT_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}"

SERVER_LOG="${OUTPUT_DIR}/geometry_server.log"
mkdir -p "${OUTPUT_DIR}"

echo "[geometry-probe] bench=${BENCH} gt_mode=${GT_MODE} samples=${NUM_SAMPLES} suites=${SUITES}"
echo "[geometry-probe] checkpoint=${CHECKPOINT}"
echo "[geometry-probe] output=${OUTPUT_DIR}"

if [[ "${GT_MODE}" == "rollout" && "${DRY_RUN}" == "1" ]]; then
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "${SIM_PYTHON}" -m \
    examples.simBenchmarks.CoT.geometry_probe.sim_rollout_probe \
    --bench "${BENCH}" --hdf5-root "${HDF5_ROOT}" --suites "${SUITES}" \
    --num-samples "${NUM_SAMPLES}" --horizon 8 --seed "${SEED}" \
    --output-dir "${OUTPUT_DIR}" --dry-run
  exit 0
fi

COMMON_ARGS=(
  --bench "${BENCH}"
  --gt-mode "${GT_MODE}"
  --dataset-root "${DATASET_ROOT}"
  --suites "${SUITES}"
  --num-samples "${NUM_SAMPLES}"
  --seed "${SEED}"
  --video-backend "${VIDEO_BACKEND}"
  --video-thread-count "${VIDEO_THREAD_COUNT}"
  --output-dir "${OUTPUT_DIR}"
  --host 127.0.0.1
  --port "${PORT}"
)

if [[ "${GT_MODE}" == "dataset" && "${DRY_RUN}" == "1" ]]; then
  PYTHONPATH="${ROOT_DIR}" "${PYTHON_BIN}" -m \
    examples.simBenchmarks.CoT.geometry_probe.run_geometry_probe \
    "${COMMON_ARGS[@]}" --dry-run
  exit 0
fi

SERVER_ARGS=(
  --ckpt_path "${CHECKPOINT}"
  --port "${PORT}"
  --idle_timeout 3600
)

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${ROOT_DIR}" "${PYTHON_BIN}" -m \
  examples.simBenchmarks.CoT.geometry_probe.geometry_server_policy \
  "${SERVER_ARGS[@]}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[geometry-probe] server_pid=${SERVER_PID}; waiting for port ${PORT}"
"${PYTHON_BIN}" - "${PORT}" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
deadline = time.time() + 300
while time.time() < deadline:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            print(f"[geometry-probe] server ready on {port}")
            raise SystemExit(0)
    except OSError:
        time.sleep(1)
raise SystemExit(f"server did not become ready on port {port}")
PY

if [[ "${GT_MODE}" == "dataset" ]]; then
  PYTHONPATH="${ROOT_DIR}" "${PYTHON_BIN}" -m \
    examples.simBenchmarks.CoT.geometry_probe.run_geometry_probe \
    "${COMMON_ARGS[@]}"
elif [[ "${GT_MODE}" == "rollout" ]]; then
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "${SIM_PYTHON}" -m \
    examples.simBenchmarks.CoT.geometry_probe.sim_rollout_probe \
    --bench "${BENCH}" --hdf5-root "${HDF5_ROOT}" --suites "${SUITES}" \
    --num-samples "${NUM_SAMPLES}" --horizon 8 --seed "${SEED}" \
    --output-dir "${OUTPUT_DIR}" --host 127.0.0.1 --port "${PORT}"
else
  echo "unsupported GT_MODE=${GT_MODE}" >&2
  exit 2
fi
