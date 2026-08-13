#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

: "${TRAIN_PID:?TRAIN_PID must identify the active training launcher}"
: "${MODEL_DIR:?MODEL_DIR must identify the training output directory}"

EXPECTED_STEP="${EXPECTED_STEP:-60000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
GPU_RELEASE_WAIT_SECONDS="${GPU_RELEASE_WAIT_SECONDS:-30}"
LIBERO_EVAL_SCRIPT="${LIBERO_EVAL_SCRIPT:-${REPO_ROOT}/examples/simBenchmarks/LIBERO/eval_files/run_multigpu_eval.sh}"
LIBERO_PLUS_EVAL_SCRIPT="${LIBERO_PLUS_EVAL_SCRIPT:-${REPO_ROOT}/examples/simBenchmarks/LIBERO-plus/eval_files/run_multigpu_eval.sh}"
LOG_PATH="${MODEL_DIR}/posttrain_parallel_eval.log"

[[ "${TRAIN_PID}" =~ ^[1-9][0-9]*$ ]] || {
  echo "TRAIN_PID must be a positive integer: ${TRAIN_PID}" >&2
  exit 2
}
[[ "${EXPECTED_STEP}" =~ ^[1-9][0-9]*$ ]] || {
  echo "EXPECTED_STEP must be a positive integer: ${EXPECTED_STEP}" >&2
  exit 2
}
[[ -d "${MODEL_DIR}" ]] || {
  echo "MODEL_DIR does not exist: ${MODEL_DIR}" >&2
  exit 2
}
[[ -f "${LIBERO_EVAL_SCRIPT}" ]] || {
  echo "standard LIBERO launcher does not exist: ${LIBERO_EVAL_SCRIPT}" >&2
  exit 2
}
[[ -f "${LIBERO_PLUS_EVAL_SCRIPT}" ]] || {
  echo "LIBERO-plus launcher does not exist: ${LIBERO_PLUS_EVAL_SCRIPT}" >&2
  exit 2
}

exec > >(tee -a "${LOG_PATH}") 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

step_checkpoint="${MODEL_DIR}/checkpoints/steps_${EXPECTED_STEP}_pytorch_model.pt"
final_checkpoint="${MODEL_DIR}/final_model/pytorch_model.pt"

log "waiting for training PID ${TRAIN_PID} to exit"
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sleep "${POLL_SECONDS}"
done
log "training PID ${TRAIN_PID} has exited; validating final artifacts"

for artifact in "${step_checkpoint}" "${final_checkpoint}"; do
  if [[ ! -s "${artifact}" ]]; then
    log "required checkpoint artifact is missing or empty: ${artifact}"
    exit 1
  fi
done

if [[ "${GPU_RELEASE_WAIT_SECONDS}" != "0" ]]; then
  log "artifacts are complete; waiting ${GPU_RELEASE_WAIT_SECONDS}s for GPU release"
  sleep "${GPU_RELEASE_WAIT_SECONDS}"
fi

standard_pid=""
plus_pid=""
terminate_children() {
  log "received termination signal; stopping evaluation launchers"
  [[ -z "${standard_pid}" ]] || kill "${standard_pid}" 2>/dev/null || true
  [[ -z "${plus_pid}" ]] || kill "${plus_pid}" 2>/dev/null || true
}
trap terminate_children INT TERM HUP

log "starting standard LIBERO on GPUs 0,1,2,3 (50 trials/task, no video)"
MODEL_DIR="${MODEL_DIR}" \
CKPT="${final_checkpoint}" \
GPUS="0,1,2,3" \
NUM_TRIALS_PER_TASK="50" \
SAVE_VIDEO="0" \
bash "${LIBERO_EVAL_SCRIPT}" &
standard_pid="$!"

log "starting LIBERO-plus on GPUs 0-7 (1 trial/task, no video)"
MODEL_DIR="${MODEL_DIR}" \
CKPT="${final_checkpoint}" \
GPUS="0,1,2,3,4,5,6,7" \
NUM_TRIALS_PER_TASK="1" \
SAVE_VIDEO="0" \
bash "${LIBERO_PLUS_EVAL_SCRIPT}" &
plus_pid="$!"

if wait "${standard_pid}"; then
  standard_status=0
else
  standard_status="$?"
fi
log "standard LIBERO exit status: ${standard_status}"

if wait "${plus_pid}"; then
  plus_status=0
else
  plus_status="$?"
fi
log "LIBERO-plus exit status: ${plus_status}"

trap - INT TERM HUP

if ((standard_status != 0 || plus_status != 0)); then
  log "parallel evaluation finished with failures"
  exit 1
fi

log "parallel evaluation completed successfully"
