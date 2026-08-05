#!/usr/bin/env bash
set -euo pipefail

# Split-aware, one-policy-server/one-CALVIN-worker-per-GPU evaluation.
# Required environment: CALVIN_CONFIG_PATH and either CHECKPOINT or MODEL_DIR.
# Optional split: SPLIT=ABCD_D or SPLIT=ABC_D.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

SPLIT="${SPLIT:-ABCD_D}"
case "${SPLIT}" in
  ABCD_D|ABCD\-\>D) SPLIT="ABCD_D" ;;
  ABC_D|ABC\-\>D) SPLIT="ABC_D" ;;
  *) echo "Unsupported SPLIT=${SPLIT}; use ABCD_D or ABC_D" >&2; exit 2 ;;
esac

POLICY_PYTHON="${POLICY_PYTHON:-/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python}"
CALVIN_PYTHON="${CALVIN_PYTHON:-/root/data/yxz/miniforge3/envs/calvin/bin/python}"
CALVIN_DATA_ROOT="${CALVIN_DATA_ROOT:-/root/data/yxz/datasets/calvin}"
DATASET_PATH="${DATASET_PATH:-${CALVIN_DATA_ROOT}/task_${SPLIT}}"
CALVIN_CONFIG_PATH="${CALVIN_CONFIG_PATH:-${CALVIN_ROOT:-}/calvin_models/conf}"
EVAL_SEQUENCES_PATH="${EVAL_SEQUENCES_PATH:-${REPO_ROOT}/examples/simBenchmarks/calvin/eval_files/eval_sequences.json}"

GPUS="${GPUS:-0,1,2,3}"
BASE_PORT="${BASE_PORT:-5694}"
NUM_SEQUENCES="${NUM_SEQUENCES:-1000}"
ACTION_STRIDE="${ACTION_STRIDE:-0}"
RENDER_BACKEND="${RENDER_BACKEND:-egl}"
USE_BF16="${USE_BF16:-1}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
RESET="${RESET:-0}"
DEBUG="${DEBUG:-0}"

if [[ -z "${CHECKPOINT:-}" ]]; then
  : "${MODEL_DIR:?Set CHECKPOINT or MODEL_DIR}"
  resolver_cmd=("${POLICY_PYTHON}" examples/simBenchmarks/eval_common/checkpoint_utils.py --model-dir "${MODEL_DIR}")
  if [[ -n "${CKPT_NAME:-}" ]]; then resolver_cmd+=(--checkpoint "${CKPT_NAME}"); fi
  CHECKPOINT="$("${resolver_cmd[@]}")"
fi

[[ -f "${CHECKPOINT}" ]] || { echo "checkpoint not found: ${CHECKPOINT}" >&2; exit 2; }
[[ -d "${DATASET_PATH}/validation" ]] || { echo "CALVIN validation directory not found: ${DATASET_PATH}/validation" >&2; exit 2; }
[[ -f "${EVAL_SEQUENCES_PATH}" ]] || { echo "sequence file not found: ${EVAL_SEQUENCES_PATH}" >&2; exit 2; }
[[ -d "${CALVIN_CONFIG_PATH}" ]] || { echo "CALVIN_CONFIG_PATH not found: ${CALVIN_CONFIG_PATH}" >&2; exit 2; }

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NUM_WORKERS="${#GPU_LIST[@]}"
(( NUM_WORKERS > 0 )) || { echo "GPUS must contain at least one GPU" >&2; exit 2; }

RUN_ID="${RUN_ID:-qwen35_gr00t_calvin_${SPLIT}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/yxz/outputs/${RUN_ID}}"
mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/workers"

cat > "${OUTPUT_DIR}/protocol.env" <<EOF
SPLIT=${SPLIT}
CHECKPOINT=${CHECKPOINT}
DATASET_PATH=${DATASET_PATH}
CALVIN_CONFIG_PATH=${CALVIN_CONFIG_PATH}
EVAL_SEQUENCES_PATH=${EVAL_SEQUENCES_PATH}
GPUS=${GPUS}
NUM_SEQUENCES=${NUM_SEQUENCES}
ACTION_STRIDE=${ACTION_STRIDE}
RENDER_BACKEND=${RENDER_BACKEND}
USE_BF16=${USE_BF16}
UNNORM_KEY=${UNNORM_KEY}
EOF

server_pids=()
worker_pids=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${worker_pids[@]:-}" "${server_pids[@]:-}"; do
    [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_port() {
  local port="$1"
  for _ in $(seq 1 600); do
    if (echo > "/dev/tcp/127.0.0.1/${port}") >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

base=$((NUM_SEQUENCES / NUM_WORKERS))
remainder=$((NUM_SEQUENCES % NUM_WORKERS))
worker_jsons=()

for worker_id in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$worker_id]}"
  port=$((BASE_PORT + worker_id))
  if (( worker_id < remainder )); then extra=1; else extra=0; fi
  start=$((worker_id * base + (worker_id < remainder ? worker_id : remainder)))
  end=$((start + base + extra))
  worker_dir="${OUTPUT_DIR}/workers/worker_${worker_id}"
  mkdir -p "${worker_dir}/logs"
  worker_json="${worker_dir}/results.json"
  worker_jsons+=("${worker_json}")

  server_cmd=("${POLICY_PYTHON}" deployment/model_server/server_policy.py
    --ckpt_path "${CHECKPOINT}" --port "${port}")
  if [[ "${USE_BF16}" == "1" ]]; then server_cmd+=(--use_bf16); fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${server_cmd[@]}" \
    >"${OUTPUT_DIR}/logs/server_gpu${gpu}.log" 2>&1 &
  server_pids+=("$!")
  echo "[calvin] server split=${SPLIT} worker=${worker_id} gpu=${gpu} port=${port} range=[${start},${end})"
done

for worker_id in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$worker_id]}"
  port=$((BASE_PORT + worker_id))
  server_pid="${server_pids[$worker_id]}"
  kill -0 "${server_pid}" 2>/dev/null || { echo "server process failed on GPU ${gpu}" >&2; exit 1; }
  wait_for_port "${port}" || { echo "server failed to open port ${port}" >&2; exit 1; }
done

for worker_id in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$worker_id]}"
  port=$((BASE_PORT + worker_id))
  if (( worker_id < remainder )); then extra=1; else extra=0; fi
  start=$((worker_id * base + (worker_id < remainder ? worker_id : remainder)))
  end=$((start + base + extra))
  worker_dir="${OUTPUT_DIR}/workers/worker_${worker_id}"
  worker_cmd=("${CALVIN_PYTHON}" examples/simBenchmarks/calvin/eval_files/eval_calvin.py
    --args.host 127.0.0.1
    --args.port "${port}"
    --args.pretrained-path "${CHECKPOINT}"
    --args.unnorm-key "${UNNORM_KEY}"
    --args.dataset-path "${DATASET_PATH}"
    --args.calvin-config-path "${CALVIN_CONFIG_PATH}"
    --args.eval-sequences-path "${EVAL_SEQUENCES_PATH}"
    --args.num-sequences "${NUM_SEQUENCES}"
    --args.sequence-start "${start}"
    --args.sequence-end "${end}"
    --args.action-stride "${ACTION_STRIDE}"
    --args.split "${SPLIT}"
    --args.eval-log-dir "${worker_dir}/logs"
    --args.output-json "${worker_dir}/results.json")
  if [[ "${DEBUG}" == "1" ]]; then worker_cmd+=(--args.debug); fi
  if [[ "${RESET}" == "1" ]]; then worker_cmd+=(--args.reset); fi
  CUDA_VISIBLE_DEVICES="${gpu}" \
  CALVIN_RENDER_BACKEND="${RENDER_BACKEND}" \
  "${worker_cmd[@]}" >"${worker_dir}/worker.log" 2>&1 &
  worker_pids+=("$!")
done

status=0
for pid in "${worker_pids[@]}"; do
  wait "${pid}" || status=1
done
(( status == 0 )) || { echo "[calvin] at least one worker failed" >&2; exit "${status}"; }

"${POLICY_PYTHON}" examples/simBenchmarks/calvin/eval_files/aggregate_calvin_results.py \
  --output "${OUTPUT_DIR}/overall_results.json" "${worker_jsons[@]}"

echo "[calvin] complete: ${OUTPUT_DIR}/overall_results.json"
