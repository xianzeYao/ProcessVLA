#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate one training output on the four standard LIBERO suites.
# One policy server and one simulator worker are started per suite/GPU.
# The default checkpoint is the numerically latest steps_* checkpoint.
#
# Example:
#   bash examples/simBenchmarks/LIBERO/eval_files/run_multigpu_eval.sh
#
# Overrides:
#   MODEL_DIR=/path/to/run CKPT_NAME=final_model/pytorch_model.pt
#   GPUS=4,5,6,7 NUM_TRIALS_PER_TASK=50 SAVE_VIDEO=1 DRY_RUN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_libero.py"
AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_results.py"
CHECKPOINT_UTILS="${REPO_ROOT}/examples/simBenchmarks/eval_common/checkpoint_utils.py"

MODEL_DIR="${MODEL_DIR:-/root/data/yxz/outputs/qwen35_gr00t_libero_baseline}"
CKPT_NAME="${CKPT_NAME:-}"
LIBERO_HOME="${LIBERO_HOME:-/root/data/yxz/benchmarks/LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/root/data/yxz/miniforge3/envs/libero/bin/python}"
POLICY_PYTHON="${POLICY_PYTHON:-/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_DIR}_eval/libero}"
GPUS="${GPUS:-4,5,6,7}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-all}"
BASE_PORT="${BASE_PORT:-6694}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
USE_BF16="${USE_BF16:-1}"
DRY_RUN="${DRY_RUN:-0}"
MUJOCO_GL_VALUE="${MUJOCO_GL_VALUE:-egl}"
PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM_VALUE:-egl}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${LIBERO_HOME}:${PYTHONPATH:-}"

CHECKPOINT_ARGS=(--model-dir "${MODEL_DIR}")
if [[ -n "${CKPT:-}" ]]; then
  CHECKPOINT_ARGS+=(--checkpoint "${CKPT}")
elif [[ -n "${CKPT_NAME}" ]]; then
  CHECKPOINT_ARGS+=(--checkpoint "${CKPT_NAME}")
fi
CKPT="$("${LIBERO_PYTHON}" "${CHECKPOINT_UTILS}" "${CHECKPOINT_ARGS[@]}")"

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [[ "${TASK_SUITE_NAME}" == "all" ]]; then
  SUITE_LIST=(libero_spatial libero_object libero_goal libero_10)
else
  IFS=',' read -r -a SUITE_LIST <<< "${TASK_SUITE_NAME}"
fi
if ((${#GPU_LIST[@]} < ${#SUITE_LIST[@]})); then
  echo "Need at least one GPU per requested suite; got ${#GPU_LIST[@]} GPUs for ${#SUITE_LIST[@]} suites" >&2
  exit 2
fi
if ((${#GPU_LIST[@]} > ${#SUITE_LIST[@]})); then
  echo "[libero] using first ${#SUITE_LIST[@]} GPUs; standard LIBERO has one job per suite" >&2
fi

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  mkdir -p "${LIBERO_CONFIG_PATH}"
  {
    printf 'benchmark_root: %s\n' "${LIBERO_HOME}/libero/libero"
    printf 'bddl_files: %s\n' "${LIBERO_HOME}/libero/libero/bddl_files"
    printf 'init_states: %s\n' "${LIBERO_HOME}/libero/libero/init_files"
    printf 'datasets: %s\n' "${LIBERO_HOME}/datasets"
    printf 'assets: %s\n' "${LIBERO_HOME}/libero/libero/assets"
  } > "${LIBERO_CONFIG_PATH}/config.yaml"
fi
export MUJOCO_GL="${MUJOCO_GL_VALUE}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM_VALUE}"
SIM_ENV_LIB="$(cd "$(dirname "${LIBERO_PYTHON}")/../lib" 2>/dev/null && pwd || true)"
if [[ -n "${SIM_ENV_LIB}" && -d "${SIM_ENV_LIB}" ]]; then
  export LD_LIBRARY_PATH="${SIM_ENV_LIB}:${LD_LIBRARY_PATH:-}"
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_DIR}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
VIDEO_DIR="${RUN_DIR}/videos"
RESULT_DIR="${RUN_DIR}/suite_results"
mkdir -p "${LOG_DIR}" "${VIDEO_DIR}" "${RESULT_DIR}"

echo "[libero] checkpoint=${CKPT}"
echo "[libero] suites=${SUITE_LIST[*]} gpus=${GPUS} base_port=${BASE_PORT}"
echo "[libero] trials_per_task=${NUM_TRIALS_PER_TASK} save_video=${SAVE_VIDEO}"
echo "[libero] output=${RUN_DIR}"
if [[ "${DRY_RUN}" == "1" ]]; then
  for index in "${!SUITE_LIST[@]}"; do
    echo "[libero] plan suite=${SUITE_LIST[index]} gpu=${GPU_LIST[index]} port=$((BASE_PORT + index))"
  done
  echo "[libero] DRY_RUN=1; no server or worker was started"
  exit 0
fi

SERVER_PIDS=()
WORKER_PIDS=()
JOB_LABELS=()
cleanup() {
  local pid
  for pid in "${WORKER_PIDS[@]:-}" "${SERVER_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] || continue
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local port="$1" pid="$2" deadline=$((SECONDS + 180))
  while ((SECONDS < deadline)); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "policy server pid=${pid} exited before port ${port} became ready" >&2
      return 1
    fi
    if (echo >/dev/tcp/127.0.0.1/"${port}") >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  echo "timed out waiting for policy server on port ${port}" >&2
  return 1
}

for index in "${!SUITE_LIST[@]}"; do
  suite="${SUITE_LIST[index]}"
  gpu="${GPU_LIST[index]}"
  port=$((BASE_PORT + index))
  label="${suite}_gpu${gpu}"
  JOB_LABELS[index]="${label}"
  server_log="${LOG_DIR}/${label}.server.log"
  server_cmd=("${POLICY_PYTHON}" "${REPO_ROOT}/deployment/model_server/server_policy.py" --ckpt_path "${CKPT}" --port "${port}")
  [[ "${USE_BF16}" == "1" ]] && server_cmd+=(--use_bf16)
  echo "[libero] server ${label} port=${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${server_cmd[@]}" >"${server_log}" 2>&1 &
  server_pid="$!"
  SERVER_PIDS+=("${server_pid}")
  wait_for_server "${port}" "${server_pid}"
done

for index in "${!SUITE_LIST[@]}"; do
  suite="${SUITE_LIST[index]}"
  gpu="${GPU_LIST[index]}"
  port=$((BASE_PORT + index))
  label="${JOB_LABELS[index]}"
  worker_log="${LOG_DIR}/${label}.worker.log"
  result_path="${RESULT_DIR}/${suite}.json"
  video_dir="${VIDEO_DIR}/${suite}"
  mkdir -p "${video_dir}"
  worker_cmd=("${LIBERO_PYTHON}" "${EVAL_SCRIPT}" --args.pretrained-path "${CKPT}"
    --args.host 127.0.0.1 --args.port "${port}" --args.task-suite-name "${suite}"
    --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" --args.video-out-path "${video_dir}"
    --args.log-path "${LOG_DIR}" --args.result-path "${result_path}")
  if [[ "${SAVE_VIDEO}" == "1" ]]; then worker_cmd+=(--args.save-video); else worker_cmd+=(--args.no-save-video); fi
  echo "[libero] worker ${label} port=${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${worker_cmd[@]}" >"${worker_log}" 2>&1 &
  WORKER_PIDS+=("$!")
done

failed=0
for index in "${!WORKER_PIDS[@]}"; do
  if ! wait "${WORKER_PIDS[index]}"; then
    echo "[libero] worker failed: ${JOB_LABELS[index]}" >&2
    failed=1
  fi
done
if ((failed)); then
  echo "[libero] at least one worker failed; no aggregate was produced" >&2
  exit 1
fi

"${LIBERO_PYTHON}" "${AGGREGATE_SCRIPT}" --result-dir "${RESULT_DIR}" \
  --output-path "${LOG_DIR}/overall_results.json"
echo "[libero] complete: ${LOG_DIR}/overall_results.json"
