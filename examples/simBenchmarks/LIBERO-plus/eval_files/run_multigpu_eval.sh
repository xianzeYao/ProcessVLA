#!/usr/bin/env bash
set -Eeuo pipefail

# One policy server + one simulator worker per GPU. Every worker receives a
# disjoint half-open task range [start_idx, end_idx).
#
# Example:
#   bash examples/simBenchmarks/LIBERO-plus/eval_files/run_multigpu_eval.sh
#
# Defaults to MODEL_DIR=/root/data/yxz/outputs/qwen35_gr00t_libero_baseline
# and the latest numbered checkpoint (currently steps_60000...). Set
# CKPT_NAME=final_model/pytorch_model.pt to evaluate the final export.
# Set DRY_RUN=1 to print the exact range/port/GPU plan without launching jobs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_libero.py"
PLAN_SCRIPT="${SCRIPT_DIR}/plan_jobs.py"
AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_results.py"
CHECKPOINT_UTILS="${REPO_ROOT}/examples/simBenchmarks/eval_common/checkpoint_utils.py"
PROCESS_CLEANUP="${REPO_ROOT}/examples/simBenchmarks/eval_common/child_process_cleanup.sh"
source "${PROCESS_CLEANUP}"

MODEL_DIR="${MODEL_DIR:-/root/data/yxz/outputs/qwen35_gr00t_libero_baseline}"
CKPT_NAME="${CKPT_NAME:-}"
export LIBERO_HOME="${LIBERO_HOME:-/root/data/yxz/benchmarks/LIBERO-plus}"
POLICY_PYTHON="${POLICY_PYTHON:-/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python}"
SIM_PYTHON="${SIM_PYTHON:-/root/data/yxz/miniforge3/envs/libero_plus/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_DIR}_eval/libero_plus}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-all}"
BASE_PORT="${BASE_PORT:-9883}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
VIDEO_VIEWS="${VIDEO_VIEWS:-all}"
IMAGE_VIEWS="${IMAGE_VIEWS:-all}"
USE_BF16="${USE_BF16:-1}"
DRY_RUN="${DRY_RUN:-0}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${LIBERO_HOME}:${PYTHONPATH:-}"

CHECKPOINT_ARGS=(--model-dir "${MODEL_DIR}")
if [[ -n "${CKPT:-}" ]]; then
  CHECKPOINT_ARGS+=(--checkpoint "${CKPT}")
elif [[ -n "${CKPT_NAME}" ]]; then
  CHECKPOINT_ARGS+=(--checkpoint "${CKPT_NAME}")
fi
CKPT="$("${SIM_PYTHON}" "${CHECKPOINT_UTILS}" "${CHECKPOINT_ARGS[@]}")"
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [[ "${TASK_SUITE_NAME}" == "all" ]]; then
  SUITE_LIST=(libero_spatial libero_object libero_goal libero_10)
else
  IFS=',' read -r -a SUITE_LIST <<< "${TASK_SUITE_NAME}"
fi
((${#GPU_LIST[@]} >= ${#SUITE_LIST[@]})) || {
  echo "Need at least one GPU per requested suite; got ${#GPU_LIST[@]} GPUs for ${#SUITE_LIST[@]} suites" >&2
  exit 2
}

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
# LIBERO normally prompts for this file on first import. Create the equivalent
# non-interactive config so background multi-GPU workers never block on stdin.
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
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
SIM_ENV_LIB="$(cd "$(dirname "${SIM_PYTHON}")/../lib" 2>/dev/null && pwd || true)"
if [[ -n "${SIM_ENV_LIB}" && -d "${SIM_ENV_LIB}" ]]; then
  export LD_LIBRARY_PATH="${SIM_ENV_LIB}:${LD_LIBRARY_PATH:-}"
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_DIR}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
VIDEO_DIR="${RUN_DIR}/videos"
mkdir -p "${LOG_DIR}" "${VIDEO_DIR}"

echo "[multigpu] checkpoint=${CKPT}"
echo "[multigpu] gpus=${GPUS} suites=${SUITE_LIST[*]} base_port=${BASE_PORT}"
echo "[multigpu] trials_per_task=${NUM_TRIALS_PER_TASK} save_video=${SAVE_VIDEO} video_views=${VIDEO_VIEWS} image_views=${IMAGE_VIEWS}"
echo "[multigpu] output=${RUN_DIR}"

SUITES_CSV="$(IFS=,; echo "${SUITE_LIST[*]}")"
mapfile -t JOB_LINES < <(
  "${SIM_PYTHON}" "${PLAN_SCRIPT}" --libero-home "${LIBERO_HOME}" \
    --suites "${SUITES_CSV}" --worker-count "${#GPU_LIST[@]}"
)
((${#JOB_LINES[@]} > 0)) || { echo "No evaluation jobs were planned" >&2; exit 2; }

echo "[multigpu] planned jobs:"
printf '  %s\n' "${JOB_LINES[@]}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[multigpu] DRY_RUN=1; no server or worker was started"
  exit 0
fi

SERVER_PIDS=()
WORKER_PIDS=()
JOB_LABELS=()
cleanup() {
  cleanup_owned_processes
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

# Start all servers first. Port BASE_PORT+i and GPU_LIST[i] are a stable pair.
for index in "${!JOB_LINES[@]}"; do
  IFS=$'\t' read -r suite split_index split_count start_idx end_idx task_count <<< "${JOB_LINES[index]}"
  gpu="${GPU_LIST[index]}"
  port=$((BASE_PORT + index))
  label="${suite}_${start_idx}_${end_idx}_gpu${gpu}"
  JOB_LABELS[index]="${label}"
  server_log="${LOG_DIR}/${label}.server.log"
  server_cmd=("${POLICY_PYTHON}" "${REPO_ROOT}/deployment/model_server/server_policy.py" --ckpt_path "${CKPT}" --port "${port}")
  [[ "${USE_BF16}" == "1" ]] && server_cmd+=(--use_bf16)
  echo "[multigpu] server ${label} port=${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${server_cmd[@]}" >"${server_log}" 2>&1 &
  server_pid="$!"
  SERVER_PIDS+=("${server_pid}")
  remember_owned_process "${server_pid}" || true
  wait_for_server "${port}" "${server_pid}"
done

# Start one simulator worker per server, with the matching task slice.
for index in "${!JOB_LINES[@]}"; do
  IFS=$'\t' read -r suite split_index split_count start_idx end_idx task_count <<< "${JOB_LINES[index]}"
  gpu="${GPU_LIST[index]}"
  port=$((BASE_PORT + index))
  label="${JOB_LABELS[index]}"
  worker_log="${LOG_DIR}/${label}.worker.log"
  episode_jsonl="${LOG_DIR}/${label}.episodes.jsonl"
  video_dir="${VIDEO_DIR}/${label}"
  mkdir -p "${video_dir}"
  worker_cmd=("${SIM_PYTHON}" "${EVAL_SCRIPT}" --args.pretrained-path "${CKPT}"
    --args.host 127.0.0.1 --args.port "${port}" --args.task-suite-name "${suite}"
    --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" --args.start-idx "${start_idx}" --args.end-idx "${end_idx}"
    --args.video-out-path "${video_dir}" --args.video-views "${VIDEO_VIEWS}" --args.image-views "${IMAGE_VIEWS}"
    --args.log-path "${LOG_DIR}" --args.episode-result-path "${episode_jsonl}")
  if [[ "${SAVE_VIDEO}" == "1" ]]; then worker_cmd+=(--args.save-video); else worker_cmd+=(--args.no-save-video); fi
  echo "[multigpu] worker ${label} port=${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${worker_cmd[@]}" >"${worker_log}" 2>&1 &
  worker_pid="$!"
  WORKER_PIDS+=("${worker_pid}")
  remember_owned_process "${worker_pid}" || true
done

failed=0
for index in "${!WORKER_PIDS[@]}"; do
  if ! wait "${WORKER_PIDS[index]}"; then
    echo "[multigpu] worker failed: ${JOB_LABELS[index]}" >&2
    failed=1
  fi
  forget_owned_process "${WORKER_PIDS[index]}"
done
if ((failed)); then
  echo "[multigpu] at least one worker failed; no aggregate was produced" >&2
  exit 1
fi

"${SIM_PYTHON}" "${AGGREGATE_SCRIPT}" --log-dir "${LOG_DIR}" --libero-home "${LIBERO_HOME}" \
  --output-path "${LOG_DIR}/overall_results.json"
echo "[multigpu] complete: ${LOG_DIR}/overall_results.json"
