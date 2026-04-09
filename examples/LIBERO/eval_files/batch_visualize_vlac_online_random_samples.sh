#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VIS_SCRIPT="${VIS_SCRIPT:-${REPO_ROOT}/examples/LIBERO/eval_files/visualize_vlac_online_curve_with_video.py}"
RESULT_ROOT="${RESULT_ROOT:-}"
LOG_DIR="${LOG_DIR:-}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-3}"
TASK_IDS="${TASK_IDS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

if [[ -z "${RESULT_ROOT}" && -z "${LOG_DIR}" ]]; then
    echo "[ERROR] Set RESULT_ROOT=/path/to/results/libero_xxx or LOG_DIR=/path/to/_episode_logs"
    exit 1
fi

if [[ -z "${LOG_DIR}" ]]; then
    LOG_DIR="${RESULT_ROOT}/_episode_logs"
fi
if [[ -z "${RESULT_ROOT}" ]]; then
    RESULT_ROOT="$(cd "${LOG_DIR}/.." && pwd)"
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${RESULT_ROOT}/_vlac_online_overlays_random${SAMPLES_PER_TASK}"
fi

if [[ ! -d "${LOG_DIR}" ]]; then
    echo "[ERROR] Log directory not found: ${LOG_DIR}"
    exit 1
fi
if [[ ! -f "${VIS_SCRIPT}" ]]; then
    echo "[ERROR] Visualization script not found: ${VIS_SCRIPT}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if ! command -v shuf >/dev/null 2>&1; then
    echo "[ERROR] shuf is required for random sampling but was not found."
    exit 1
fi

if [[ -z "${TASK_IDS}" ]]; then
    mapfile -t TASK_ID_LIST < <(
        find "${LOG_DIR}" -maxdepth 1 -type f -name 'task*_ep*.log' -printf '%f\n' \
        | sed -E 's/^task([0-9]{3})_ep[0-9]+\.log$/\1/' \
        | sort -u
    )
else
    read -r -a raw_task_ids <<< "${TASK_IDS}"
    TASK_ID_LIST=()
    for task_id in "${raw_task_ids[@]}"; do
        printf -v padded_task_id "%03d" "${task_id}"
        TASK_ID_LIST+=("${padded_task_id}")
    done
fi

if [[ "${#TASK_ID_LIST[@]}" -eq 0 ]]; then
    echo "[ERROR] No task ids found under ${LOG_DIR}"
    exit 1
fi

selection_manifest="${OUTPUT_DIR}/selected_logs.txt"
: > "${selection_manifest}"

total_jobs=0
failed_jobs=0

echo "[INFO] RESULT_ROOT=${RESULT_ROOT}"
echo "[INFO] LOG_DIR=${LOG_DIR}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] SAMPLES_PER_TASK=${SAMPLES_PER_TASK}"
echo "[INFO] TASK_IDS=${TASK_ID_LIST[*]}"

for task_id in "${TASK_ID_LIST[@]}"; do
    mapfile -t task_logs < <(
        find "${LOG_DIR}" -maxdepth 1 -type f -name "task${task_id}_ep*.log" | sort
    )
    if [[ "${#task_logs[@]}" -eq 0 ]]; then
        echo "[WARN] No logs found for task${task_id}"
        continue
    fi

    sample_count="${SAMPLES_PER_TASK}"
    if (( sample_count > ${#task_logs[@]} )); then
        sample_count="${#task_logs[@]}"
    fi

    mapfile -t selected_logs < <(printf '%s\n' "${task_logs[@]}" | shuf -n "${sample_count}")

    echo "[INFO] task${task_id}: selected ${#selected_logs[@]} / ${#task_logs[@]}"
    for log_path in "${selected_logs[@]}"; do
        log_name="$(basename "${log_path}" .log)"
        output_path="${OUTPUT_DIR}/${log_name}_vlac_online_overlay.mp4"
        printf '%s\n' "${log_path}" >> "${selection_manifest}"
        total_jobs=$((total_jobs + 1))
        echo "[RUN] ${log_path}"
        if ! "${PYTHON_BIN}" "${VIS_SCRIPT}" \
            --log-path "${log_path}" \
            --output-path "${output_path}"; then
            echo "[ERROR] Failed to render ${log_path}"
            failed_jobs=$((failed_jobs + 1))
        fi
    done
done

echo "[DONE] Total selected logs: ${total_jobs}"
echo "[DONE] Failed renders: ${failed_jobs}"
echo "[DONE] Selection manifest: ${selection_manifest}"
echo "[DONE] Overlay directory: ${OUTPUT_DIR}"

if (( failed_jobs > 0 )); then
    exit 1
fi
