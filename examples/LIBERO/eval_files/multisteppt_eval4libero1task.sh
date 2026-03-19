#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=${LIBERO_HOME:-/path/to/LIBERO}
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}
export LIBERO_Python=${LIBERO_Python:-python3}
export STAR_VLA_Python=${STAR_VLA_Python:-python3}

run_root_dir=${RUN_ROOT_DIR:-/path/to/train_run_root}
task_suite_name=${TASK_SUITE_NAME:-libero_goal}
num_trials_per_task=${NUM_TRIALS_PER_TASK:-5}
steps=(${STEPS:-2000})
inject_signal_infer=${INJECT_SIGNAL_INFER:-True}

host=${HOST:-127.0.0.1}
base_port=${PORT:-5694}
gpu_id=${GPU_ID:-0}
wait_server_seconds=${WAIT_SERVER_SECONDS:-15}
# === End of environment variable configuration ===
###########################################################################################

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH="$(pwd):${PYTHONPATH}"

server_pid=""
cleanup_server() {
    if [[ -n "${server_pid}" ]]; then
        if kill -0 "${server_pid}" 2>/dev/null; then
            kill "${server_pid}" 2>/dev/null || true
            wait "${server_pid}" 2>/dev/null || true
        fi
        server_pid=""
    fi
}
trap cleanup_server EXIT

for step in "${steps[@]}"; do
    your_ckpt="${run_root_dir}/checkpoints/steps_${step}_pytorch_model.pt"
    if [[ ! -f "${your_ckpt}" ]]; then
        echo "[ERROR] Checkpoint not found: ${your_ckpt}"
        exit 1
    fi

    folder_name=$(echo "${your_ckpt}" | awk -F'/' '{print $(NF-2)"_"$NF}')
    eval_root=${EVAL_ROOT:-${REPO_ROOT}/results/libero_eval}
    video_out_path="${eval_root}/${folder_name}/results/${task_suite_name}/"
    LOG_DIR="${eval_root}/${folder_name}/logs/${task_suite_name}/"
    mkdir -p "${LOG_DIR}"

    echo "[INFO] =================================================="
    echo "[INFO] Evaluating checkpoint: ${your_ckpt}"
    echo "[INFO] Task suite: ${task_suite_name}, trials/task: ${num_trials_per_task}"
    echo "[INFO] GPU binding: physical GPU ${gpu_id} (process-local CUDA/EGL device 0)"
    echo "[INFO] Server port: ${base_port}"
    echo "[INFO] Video output: ${video_out_path}"
    echo "[INFO] Log output: ${LOG_DIR}"
    echo "[INFO] inject_signal_infer=${inject_signal_infer:-<yaml>}"

    eval_args=(
        --args.pretrained-path "${your_ckpt}"
        --args.host "${host}"
        --args.port "${base_port}"
        --args.task-suite-name "${task_suite_name}"
        --args.num-trials-per-task "${num_trials_per_task}"
        --args.video-out-path "${video_out_path}"
    )
    if [[ -n "${inject_signal_infer}" ]]; then
        eval_args+=(--args.inject-signal-infer "${inject_signal_infer}")
    fi

    CUDA_VISIBLE_DEVICES="${gpu_id}" EGL_VISIBLE_DEVICES=0 "${STAR_VLA_Python}" deployment/model_server/server_policy.py \
        --ckpt_path "${your_ckpt}" \
        --port "${base_port}" \
        --use_bf16 \
        > "${LOG_DIR}/server.log" 2>&1 &
    server_pid=$!

    sleep "${wait_server_seconds}"
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        echo "[ERROR] Server failed to start for ${your_ckpt}. Check ${LOG_DIR}/server.log"
        exit 1
    fi

    CUDA_VISIBLE_DEVICES="${gpu_id}" EGL_VISIBLE_DEVICES=0 "${LIBERO_Python}" ./examples/LIBERO/eval_files/eval_libero.py \
        "${eval_args[@]}" \
        2>&1 | tee "${LOG_DIR}/eval.log"

    cleanup_server
done

echo "[INFO] All requested checkpoints are evaluated."
