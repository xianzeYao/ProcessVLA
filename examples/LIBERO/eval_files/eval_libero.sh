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
infer_source=${INFER_SOURCE:-}
vlac_reference_video_path=${VLAC_REFERENCE_VIDEO_PATH:-}
vlac_reference_mode=${VLAC_REFERENCE_MODE:-}
vlac_reference_dataset_name=${VLAC_REFERENCE_DATASET_NAME:-}
vlac_reference_seed=${VLAC_REFERENCE_SEED:-}
vlac_reference_data_root_dir=${VLAC_REFERENCE_DATA_ROOT_DIR:-}
vlac_reference_data_mix=${VLAC_REFERENCE_DATA_MIX:-}
vlac_signal_kind=${VLAC_SIGNAL_KIND:-}
vlac_skip=${VLAC_SKIP:-}
vlac_frame_skip=${VLAC_FRAME_SKIP:-}
vlac_ref_num=${VLAC_REF_NUM:-}
vlac_batch_num=${VLAC_BATCH_NUM:-}
vlac_rich=${VLAC_RICH:-}
vlac_think=${VLAC_THINK:-}
vlac_device=${VLAC_DEVICE:-}
vlac_python=${VLAC_PYTHON:-}
vlac_model_path=${VLAC_MODEL_PATH:-}
vlac_model_type=${VLAC_MODEL_TYPE:-}
vlac_repo_root=${VLAC_REPO_ROOT:-}
liv_python=${LIV_PYTHON:-${LIV_Python:-}}
liv_repo_root=${LIV_REPO_ROOT:-${CRITIC4VLA_LIV_REPO_ROOT:-}}
liv_device=${LIV_DEVICE:-}

normalize_bool_arg() {
    local raw="${1:-}"
    if [[ -z "${raw}" ]]; then
        return 0
    fi
    case "${raw}" in
        true|True|TRUE|1|yes|YES) printf '%s' "True" ;;
        false|False|FALSE|0|no|NO) printf '%s' "False" ;;
        none|None|NONE|null|NULL) printf '%s' "None" ;;
        *) printf '%s' "${raw}" ;;
    esac
}

vlac_frame_skip="$(normalize_bool_arg "${vlac_frame_skip}")"
vlac_rich="$(normalize_bool_arg "${vlac_rich}")"
vlac_think="$(normalize_bool_arg "${vlac_think}")"
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

if [[ -n "${infer_source}" ]]; then
    eval_args+=(--args.infer-source "${infer_source}")
fi
if [[ -n "${vlac_reference_video_path}" ]]; then
    eval_args+=(--args.vlac-reference-video-path "${vlac_reference_video_path}")
fi
if [[ -n "${vlac_reference_mode}" ]]; then
    eval_args+=(--args.vlac-reference-mode "${vlac_reference_mode}")
fi
if [[ -n "${vlac_reference_dataset_name}" ]]; then
    eval_args+=(--args.vlac-reference-dataset-name "${vlac_reference_dataset_name}")
fi
if [[ -n "${vlac_reference_seed}" ]]; then
    eval_args+=(--args.vlac-reference-seed "${vlac_reference_seed}")
fi
if [[ -n "${vlac_reference_data_root_dir}" ]]; then
    eval_args+=(--args.vlac-reference-data-root-dir "${vlac_reference_data_root_dir}")
fi
if [[ -n "${vlac_reference_data_mix}" ]]; then
    eval_args+=(--args.vlac-reference-data-mix "${vlac_reference_data_mix}")
fi
if [[ -n "${vlac_signal_kind}" ]]; then
    eval_args+=(--args.vlac-signal-kind "${vlac_signal_kind}")
fi
if [[ -n "${vlac_skip}" ]]; then
    eval_args+=(--args.vlac-skip "${vlac_skip}")
fi
if [[ -n "${vlac_frame_skip}" ]]; then
    eval_args+=(--args.vlac-frame-skip "${vlac_frame_skip}")
fi
if [[ -n "${vlac_ref_num}" ]]; then
    eval_args+=(--args.vlac-ref-num "${vlac_ref_num}")
fi
if [[ -n "${vlac_batch_num}" ]]; then
    eval_args+=(--args.vlac-batch-num "${vlac_batch_num}")
fi
if [[ -n "${vlac_rich}" ]]; then
    eval_args+=(--args.vlac-rich "${vlac_rich}")
fi
if [[ -n "${vlac_think}" ]]; then
    eval_args+=(--args.vlac-think "${vlac_think}")
fi
if [[ -n "${vlac_device}" ]]; then
    eval_args+=(--args.vlac-device "${vlac_device}")
fi
if [[ -n "${vlac_python}" ]]; then
    eval_args+=(--args.vlac-python "${vlac_python}")
fi
if [[ -n "${vlac_model_path}" ]]; then
    eval_args+=(--args.vlac-model-path "${vlac_model_path}")
fi
if [[ -n "${vlac_model_type}" ]]; then
    eval_args+=(--args.vlac-model-type "${vlac_model_type}")
fi
if [[ -n "${vlac_repo_root}" ]]; then
    eval_args+=(--args.vlac-repo-root "${vlac_repo_root}")
fi
if [[ -n "${liv_python}" ]]; then
    eval_args+=(--args.liv-python "${liv_python}")
fi
if [[ -n "${liv_repo_root}" ]]; then
    eval_args+=(--args.liv-repo-root "${liv_repo_root}")
fi
if [[ -n "${liv_device}" ]]; then
    eval_args+=(--args.liv-device "${liv_device}")
fi

${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    "${eval_args[@]}" \
    2>&1 | tee "${LOG_DIR}/eval.log"
