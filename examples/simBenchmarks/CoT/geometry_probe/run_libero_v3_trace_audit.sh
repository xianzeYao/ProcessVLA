#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

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
LIBERO_HOME="${LIBERO_HOME:-}"
if [[ -n "${LIBERO_HOME}" ]]; then
  LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
else
  LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-}"
fi
MUJOCO_GL_VALUE="${MUJOCO_GL_VALUE:-${MUJOCO_GL:-egl}}"
PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM_VALUE:-${PYOPENGL_PLATFORM:-egl}}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-180}"
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
[[ "${SERVER_READY_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "SERVER_READY_TIMEOUT must be positive" >&2
  exit 2
}

cd "${REPO_ROOT}"
PYTHONPATH_VALUE="${REPO_ROOT}"
[[ -z "${LIBERO_HOME}" ]] || PYTHONPATH_VALUE+=":${LIBERO_HOME}"
export PYTHONPATH="${PYTHONPATH_VALUE}${PYTHONPATH:+:${PYTHONPATH}}"
[[ -z "${LIBERO_HOME}" ]] || export LIBERO_HOME
[[ -z "${LIBERO_CONFIG_PATH}" ]] || export LIBERO_CONFIG_PATH
export MUJOCO_GL="${MUJOCO_GL_VALUE}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM_VALUE}"
SIM_ENV_LIB="$(cd "$(dirname "${SIM_PYTHON}")/../lib" 2>/dev/null && pwd || true)"
if [[ -n "${SIM_ENV_LIB}" && -d "${SIM_ENV_LIB}" ]]; then
  export LD_LIBRARY_PATH="${SIM_ENV_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

print_command() {
  local label="$1"
  shift
  printf '[trace-audit] %s:' "${label}"
  printf ' %q' "$@"
  printf '\n'
}

PLAN_CMD=("${SIM_PYTHON}" -m examples.simBenchmarks.CoT.geometry_probe.run_libero_v3_trace_audit
  --phase plan --checkpoint "${CHECKPOINT}" --source-log-dir "${SOURCE_LOG_DIR}"
  --output-dir "${OUTPUT_DIR}" --suites "$(IFS=,; echo "${SUITES[*]}")"
  --seeds "${SEEDS}" --host "${HOST}" --base-port "${BASE_PORT}"
  --mujoco-gl "${MUJOCO_GL}" --pyopengl-platform "${PYOPENGL_PLATFORM}")
[[ -z "${MAX_CASES}" ]] || PLAN_CMD+=(--max-cases "${MAX_CASES}")
[[ -z "${LIBERO_HOME}" ]] || PLAN_CMD+=(--libero-home "${LIBERO_HOME}")
[[ -z "${LIBERO_CONFIG_PATH}" ]] || PLAN_CMD+=(--libero-config-path "${LIBERO_CONFIG_PATH}")
[[ "${DRY_RUN}" == "1" ]] && PLAN_CMD+=(--dry-run)
"${PLAN_CMD[@]}"

printf '[trace-audit] env LIBERO_HOME=%q LIBERO_CONFIG_PATH=%q MUJOCO_GL=%q PYOPENGL_PLATFORM=%q\n' \
  "${LIBERO_HOME}" "${LIBERO_CONFIG_PATH}" "${MUJOCO_GL}" "${PYOPENGL_PLATFORM}"
print_command "plan command" "${PLAN_CMD[@]}"
for index in "${!SUITES[@]}"; do
  suite="${SUITES[index]}"
  gpu="${GPU_LIST[index]}"
  port=$((BASE_PORT + index))
  SERVER_CMD=(env "CUDA_VISIBLE_DEVICES=${gpu}" "${POLICY_PYTHON}"
    "${REPO_ROOT}/deployment/model_server/server_policy.py" --ckpt_path "${CHECKPOINT}" --port "${port}")
  [[ "${USE_BF16}" == "1" ]] && SERVER_CMD+=(--use_bf16)
  WORKER_CMD=(env "CUDA_VISIBLE_DEVICES=${gpu}" "${SIM_PYTHON}" -m
    examples.simBenchmarks.CoT.geometry_probe.run_libero_v3_trace_audit
    --phase worker --output-dir "${OUTPUT_DIR}" --worker-suite "${suite}"
    --host "${HOST}" --port "${port}")
  printf '[trace-audit] plan suite=%s gpu=%s port=%s\n' "${suite}" "${gpu}" "${port}"
  print_command "server command" "${SERVER_CMD[@]}"
  print_command "worker command" "${WORKER_CMD[@]}"
done
if [[ "${DRY_RUN}" == "1" ]]; then
  case_count="$("${SIM_PYTHON}" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["cases"]))' "${OUTPUT_DIR}/run_manifest.json")"
  echo "[trace-audit] 4 suites, ${case_count} cases"
  echo "[trace-audit] DRY_RUN=1; no server or worker was started and no process was killed"
  exit 0
fi

[[ -d "${LIBERO_HOME}" ]] || { echo "LIBERO_HOME must be an existing directory" >&2; exit 2; }
[[ -f "${LIBERO_CONFIG_PATH}/config.yaml" ]] || {
  echo "LIBERO_CONFIG_PATH/config.yaml does not exist: ${LIBERO_CONFIG_PATH}" >&2
  exit 2
}
[[ -x "${POLICY_PYTHON}" ]] || { echo "POLICY_PYTHON is not executable" >&2; exit 2; }
[[ -x "${SIM_PYTHON}" ]] || { echo "SIM_PYTHON is not executable" >&2; exit 2; }
if ((BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1))); then
  echo "Bash 5.1+ is required for fail-fast wait -n -p" >&2
  exit 2
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
SERVER_READY=(0 0 0 0)

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

wait_for_all_servers() {
  local deadline=$((SECONDS + SERVER_READY_TIMEOUT))
  local index port all_ready
  while ((SECONDS < deadline)); do
    all_ready=1
    for index in "${!SERVER_PIDS[@]}"; do
      if ! kill -0 "${SERVER_PIDS[index]}" 2>/dev/null; then
        echo "policy server suite=${SUITES[index]} exited during global readiness" >&2
        return 1
      fi
      if [[ "${SERVER_READY[index]}" == "0" ]]; then
        port=$((BASE_PORT + index))
        if (echo >/dev/tcp/"${HOST}"/"${port}") >/dev/null 2>&1; then
          SERVER_READY[index]=1
        else
          all_ready=0
        fi
      fi
    done
    ((all_ready == 1)) && return 0
    sleep 1
  done
  echo "global policy server readiness timeout after ${SERVER_READY_TIMEOUT}s" >&2
  return 1
}

wait_for_workers_fail_fast() {
  local -a active
  local remaining=${#WORKER_PIDS[@]} finished_pid status index pid
  while ((remaining > 0)); do
    active=()
    for pid in "${WORKER_PIDS[@]}"; do [[ -n "${pid}" ]] && active+=("${pid}"); done
    finished_pid=""
    if wait -n -p finished_pid "${active[@]}"; then status=0; else status=$?; fi
    for index in "${!WORKER_PIDS[@]}"; do
      [[ "${WORKER_PIDS[index]}" == "${finished_pid}" ]] || continue
      WORKER_PIDS[index]=""
      ((remaining -= 1))
      if ((status != 0)); then
        echo "worker failed: ${SUITES[index]} status=${status}" >&2
        for pid in "${WORKER_PIDS[@]}"; do terminate_group "${pid}"; done
        for pid in "${WORKER_PIDS[@]}"; do
          [[ -n "${pid}" ]] && wait "${pid}" 2>/dev/null || true
        done
        WORKER_PIDS=()
        return 1
      fi
      break
    done
  done
  WORKER_PIDS=()
}

for index in "${!SUITES[@]}"; do
  suite="${SUITES[index]}"; gpu="${GPU_LIST[index]}"; port=$((BASE_PORT + index))
  SERVER_CMD=("${POLICY_PYTHON}" "${REPO_ROOT}/deployment/model_server/server_policy.py"
    --ckpt_path "${CHECKPOINT}" --port "${port}")
  [[ "${USE_BF16}" == "1" ]] && SERVER_CMD+=(--use_bf16)
  CUDA_VISIBLE_DEVICES="${gpu}" setsid "${SERVER_CMD[@]}" >"${LOG_DIR}/${suite}.server.log" 2>&1 &
  SERVER_PIDS+=("$!")
done
wait_for_all_servers

for index in "${!SUITES[@]}"; do
  suite="${SUITES[index]}"; gpu="${GPU_LIST[index]}"; port=$((BASE_PORT + index))
  WORKER_CMD=("${SIM_PYTHON}" -m examples.simBenchmarks.CoT.geometry_probe.run_libero_v3_trace_audit
    --phase worker --output-dir "${OUTPUT_DIR}" --worker-suite "${suite}"
    --host "${HOST}" --port "${port}")
  CUDA_VISIBLE_DEVICES="${gpu}" setsid "${WORKER_CMD[@]}" >"${LOG_DIR}/${suite}.worker.log" 2>&1 &
  WORKER_PIDS+=("$!")
done
wait_for_workers_fail_fast || { echo "workers failed; aggregate/render not started" >&2; exit 1; }

"${SIM_PYTHON}" -m examples.simBenchmarks.CoT.geometry_probe.run_libero_v3_trace_audit \
  --phase aggregate --output-dir "${OUTPUT_DIR}"
echo "[trace-audit] complete: ${OUTPUT_DIR}/summary.json"
