#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=${LIBERO_HOME:-/path/to/LIBERO}
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}
export LIBERO_Python=${LIBERO_Python:-python3}

export PYTHONPATH="${LIBERO_HOME}:${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}" # let eval_libero find local tools

host=${HOST:-127.0.0.1}
base_port=${PORT:-5694}
your_ckpt=${CKPT_PATH:-/path/to/checkpoint.pt}
# export DEBUG=false

# === End of environment variable configuration ===
###########################################################################################

task_suite_name=${TASK_SUITE_NAME:-libero_goal}
num_trials_per_task=${NUM_TRIALS_PER_TASK:-50}
inject_signal_infer=${INJECT_SIGNAL_INFER:-}
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$NF}')
eval_root=${EVAL_ROOT:-${REPO_ROOT}/results/libero_eval}
video_out_path="${eval_root}/${folder_name}/results/${task_suite_name}/"
LOG_DIR="${eval_root}/${folder_name}/logs/${task_suite_name}/"
mkdir -p "${LOG_DIR}"

eval_args=(
    --args.pretrained-path "${your_ckpt}"
    --args.host "$host"
    --args.port "$base_port"
    --args.task-suite-name "$task_suite_name"
    --args.num-trials-per-task "$num_trials_per_task"
    --args.video-out-path "$video_out_path"
)

if [[ -n "${inject_signal_infer}" ]]; then
    eval_args+=(--args.inject-signal-infer "${inject_signal_infer}")
fi

${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    "${eval_args[@]}" \
    2>&1 | tee "${LOG_DIR}/eval.log"
