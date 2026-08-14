#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CLI="${SCRIPT_DIR}/run_libero_v3_trace_audit.py"

: "${MODEL_DIR:?MODEL_DIR is required}"
: "${SOURCE_LOG_DIR:?SOURCE_LOG_DIR is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

CKPT_NAME="${CKPT_NAME:-final_model/pytorch_model.pt}"
GPUS="${GPUS:-0,1,2,3}"
BASE_PORT="${BASE_PORT:-6694}"
SEEDS="${SEEDS:-7,8,9,10,11}"
DRY_RUN="${DRY_RUN:-0}"
MAX_CASES="${MAX_CASES:-}"
POLICY_PYTHON="${POLICY_PYTHON:-/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python}"
SIM_PYTHON="${SIM_PYTHON:-/root/data/yxz/miniforge3/envs/libero/bin/python}"
USE_BF16="${USE_BF16:-1}"
HOST="${HOST:-127.0.0.1}"
SUITES=(libero_spatial libero_object libero_goal libero_10)

if [[ "${CKPT_NAME}" = /* ]]; then
  CHECKPOINT="${CKPT_NAME}"
else
  CHECKPOINT="${MODEL_DIR}/${CKPT_NAME}"
fi
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if ((${#GPU_LIST[@]} != 4)); then
  echo "GPUS must contain exactly four explicit IDs; got ${#GPU_LIST[@]}" >&2
  exit 2
fi
declare -A SEEN_GPU=()
for gpu in "${GPU_LIST[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "invalid GPU ID: ${gpu}" >&2; exit 2; }
  [[ -z "${SEEN_GPU[${gpu}]:-}" ]] || { echo "duplicate GPU ID: ${gpu}" >&2; exit 2; }
  SEEN_GPU["${gpu}"]=1
done
[[ "${BASE_PORT}" =~ ^[0-9]+$ ]] && ((BASE_PORT >= 1 && BASE_PORT + 3 <= 65535)) || {
  echo "invalid BASE_PORT: ${BASE_PORT}" >&2
  exit 2
}

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PLAN_CMD=("${SIM_PYTHON}" -m examples.simBenchmarks.CoT.geometry_probe.run_libero_v3_trace_audit
  --phase plan --checkpoint "${CHECKPOINT}" --source-log-dir "${SOURCE_LOG_DIR}"
  --output-dir "${OUTPUT_DIR}" --suites "$(IFS=,; echo "${SUITES[*]}")"
  --seeds "${SEEDS}" --host "${HOST}" --base-port "${BASE_PORT}")
[[ -z "${MAX_CASES}" ]] || PLAN_CMD+=(--max-cases "${MAX_CASES}")
[[ "${DRY_RUN}" == "1" ]] && PLAN_CMD+=(--dry-run)
"${PLAN_CMD[@]}"

WORKER_CMDS=()
for index in "${!SUITES[@]}"; do
  suite="${SUITES[index]}"
  gpu="${GPU_LIST[index]}"
  port=$((BASE_PORT + index))
  WORKER_CMDS[index]="CUDA_VISIBLE_DEVICES=${gpu} ${SIM_PYTHON} -m examples.simBenchmarks.CoT.geometry_probe.run_libero_v3_trace_audit --phase worker --output-dir ${OUTPUT_DIR} --worker-suite ${suite} --host ${HOST} --port ${port}"
  printf '[trace-audit] plan suite=%s gpu=%s port=%s\n' "${suite}" "${gpu}" "${port}"
  printf '[trace-audit] worker command: %s\n' "${WORKER_CMDS[index]}"
done
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[trace-audit] 4 suites, $("${SIM_PYTHON}" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["cases"]))' "${OUTPUT_DIR}/run_manifest.json") cases"
  echo "[trace-audit] DRY_RUN=1; no server or worker was started and no process was killed"
  exit 0
fi

for port_offset in 0 1 2 3; do
  port=$((BASE_PORT + port_offset))
  if (echo >/dev/tcp/"${HOST}"/"${port}") >/dev/null 2>&1; then
    echo "port conflict: ${HOST}:${port} is already accepting connections" >&2
    exit 2
  fi
done

LOG_DIR="${OUTPUT_DIR}/launcher_logs"
mkdir -p "${LOG_DIR}"
SERVER_PIDS=()
WORKER_PIDS=()

terminate_group() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 0
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || true
  fi
}

cleanup() {
  local pid
  for pid in "${WORKER_PIDS[@]:-}" "${SERVER_PIDS[@]:-}"; do
    terminate_group "${pid}"
  done
  for pid in "${WORKER_PIDS[@]:-}" "${SERVER_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

wait_for_server() {
  local port="$1" pid="$2" deadline=$((SECONDS + 180))
  while ((SECONDS < deadline)); do
    kill -0 "${pid}" 2>/dev/null || {
      echo "policy server pid=${pid} exited before ${HOST}:${port} became ready" >&2
      return 1
    }
    if (echo >/dev/tcp/"${HOST}"/"${port}") >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "timed out waiting for policy server on ${HOST}:${port}" >&2
  return 1
}

for index in "${!SUITES[@]}"; do
  suite="${SUITES[index]}"
  gpu="${GPU_LIST[index]}"
  port=$((BASE_PORT + index))
  SERVER_CMD=("${POLICY_PYTHON}" "${REPO_ROOT}/deployment/model_server/server_policy.py"
    --ckpt_path "${CHECKPOINT}" --port "${port}")
  [[ "${USE_BF16}" == "1" ]] && SERVER_CMD+=(--use_bf16)
  CUDA_VISIBLE_DEVICES="${gpu}" setsid "${SERVER_CMD[@]}" >"${LOG_DIR}/${suite}.server.log" 2>&1 &
  SERVER_PIDS+=("$!")
done

for index in "${!SERVER_PIDS[@]}"; do
  wait_for_server "$((BASE_PORT + index))" "${SERVER_PIDS[index]}"
done

for index in "${!SUITES[@]}"; do
  suite="${SUITES[index]}"
  gpu="${GPU_LIST[index]}"
  port=$((BASE_PORT + index))
  WORKER_CMD=("${SIM_PYTHON}" -m examples.simBenchmarks.CoT.geometry_probe.run_libero_v3_trace_audit
    --phase worker --output-dir "${OUTPUT_DIR}" --worker-suite "${suite}"
    --host "${HOST}" --port "${port}")
  CUDA_VISIBLE_DEVICES="${gpu}" setsid "${WORKER_CMD[@]}" >"${LOG_DIR}/${suite}.worker.log" 2>&1 &
  WORKER_PIDS+=("$!")
done

failed=0
for index in "${!WORKER_PIDS[@]}"; do
  worker_pid="${WORKER_PIDS[index]}"
  if ! wait "${worker_pid}"; then
    echo "worker failed: ${SUITES[index]}" >&2
    failed=1
  fi
  WORKER_PIDS[index]=""
done
((failed == 0)) || { echo "workers failed; aggregate/render not started" >&2; exit 1; }

"${SIM_PYTHON}" -m examples.simBenchmarks.CoT.geometry_probe.run_libero_v3_trace_audit \
  --phase aggregate --output-dir "${OUTPUT_DIR}" --render-only
echo "[trace-audit] complete: ${OUTPUT_DIR}/summary.json"
