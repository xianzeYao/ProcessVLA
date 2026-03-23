#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-/home/yxz/Critic4VLA/ProcessVLA}
WORKSPACE_ROOT=${WORKSPACE_ROOT:-/home/yxz/Critic4VLA}
cd "${REPO_ROOT}"

LIV_ROOT=${LIV_ROOT:-/home/yxz/Critic4VLA/LIV}
LIV_CLIP_ROOT=${LIV_CLIP_ROOT:-${LIV_ROOT}/liv/models/clip}

export PYTHONPATH="${REPO_ROOT}:${LIV_ROOT}:${LIV_CLIP_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

STARVLA_PY=${STARVLA_PY:-/data/yxz/conda/envs/starVLA/bin/python}
read_signal_utils_value() {
    local key="$1"
    python3 - "$REPO_ROOT/starVLA/model/framework/signal_utils.py" "$key" <<'PY'
import ast
import sys

source_path = sys.argv[1]
target_key = sys.argv[2]
with open(source_path, "r", encoding="utf-8") as handle:
    tree = ast.parse(handle.read(), filename=source_path)

for node in tree.body:
    if not isinstance(node, ast.Assign):
        continue
    for target in node.targets:
        if isinstance(target, ast.Name) and target.id == target_key:
            value = ast.literal_eval(node.value)
            print("" if value is None else value)
            raise SystemExit(0)
raise SystemExit(1)
PY
}

LIV_PY=${LIV_PY:-$(read_signal_utils_value DEFAULT_LIV_PYTHON || true)}
VLAC_PY=${VLAC_PY:-$(read_signal_utils_value DEFAULT_VLAC_PYTHON || true)}
ROBODOPAMINE_PY=${ROBODOPAMINE_PY:-$(read_signal_utils_value DEFAULT_ROBODOPAMINE_PYTHON || true)}
ROBOMETER_PY=${ROBOMETER_PY:-$(read_signal_utils_value DEFAULT_ROBOMETER_PYTHON || true)}
LIV_PY=${LIV_PY:-python3}
VLAC_PY=${VLAC_PY:-python3}
ROBODOPAMINE_PY=${ROBODOPAMINE_PY:-python3}
ROBOMETER_PY=${ROBOMETER_PY:-python3}

COMPUTE_SCRIPT=${COMPUTE_SCRIPT:-/home/yxz/Critic4VLA/ProcessVLA/examples/LIBERO/eval_files/compute_external_signal_curve.py}
VIS_SCRIPT=${VIS_SCRIPT:-/home/yxz/Critic4VLA/ProcessVLA/examples/LIBERO/eval_files/visualize_signal_benchmark_with_video.py}

ROOT=${ROOT:-/data/yxz/starvla4libero/libero4in1_qwen2.5gr00t_vlatrain_baseline_steps_40000_pytorch_model.pt/results/libero_goal}
DEFAULT_REF_VIDEO=${DEFAULT_REF_VIDEO:-$ROOT/rollout_open_the_middle_drawer_of_the_cabinet_episode1_success.mp4}
# Optional override. Leave empty to infer instruction from hidden_states metadata.
INSTRUCTION=""
VLAC_REF_NUM=${VLAC_REF_NUM:-6}
VLAC_BATCH_NUM=${VLAC_BATCH_NUM:-5}
VLAC_SKIP=${VLAC_SKIP:-5}
VLAC_RICH=${VLAC_RICH:-0}
VLAC_FRAME_SKIP=${VLAC_FRAME_SKIP:-0}
VLAC_THINK=${VLAC_THINK:-0}

run_one() {
    local video_path="$1"
    local ref_video_path="${2:-$DEFAULT_REF_VIDEO}"
    local wrist_path=""
    local overlay_path="${video_path%.mp4}_signals_overlay.mp4"
    local instruction_args=()
    local vlac_args=(
        --reference-video-path "$ref_video_path"
        --vlac-ref-num "$VLAC_REF_NUM"
        --vlac-batch-num "$VLAC_BATCH_NUM"
        --vlac-skip "$VLAC_SKIP"
    )

    if [[ "$video_path" =~ ^(.*/rollout_.+)_episode([0-9]+)_(success|failure)\.mp4$ ]]; then
        wrist_path="${BASH_REMATCH[1]}_wrist_episode${BASH_REMATCH[2]}_${BASH_REMATCH[3]}.mp4"
    fi
    if [[ ! -f "$ref_video_path" ]]; then
        echo "[ERROR] Reference video not found: $ref_video_path"
        return 1
    fi
    if [[ -n "${INSTRUCTION}" ]]; then
        instruction_args=(--instruction "$INSTRUCTION")
    else
        echo "[INFO] instruction not provided; infer from video/hidden_states metadata"
    fi
    if [[ "${VLAC_RICH}" == "1" || "${VLAC_RICH,,}" == "true" ]]; then
        vlac_args+=(--vlac-rich)
    fi
    if [[ "${VLAC_FRAME_SKIP}" == "1" || "${VLAC_FRAME_SKIP,,}" == "true" ]]; then
        vlac_args+=(--vlac-frame-skip)
    fi
    if [[ "${VLAC_THINK}" == "1" || "${VLAC_THINK,,}" == "true" ]]; then
        vlac_args+=(--vlac-think)
    fi

    echo "[RUN] liv -> $video_path"
    "$LIV_PY" "$COMPUTE_SCRIPT" \
        --model liv \
        --video-path "$video_path" \
        "${instruction_args[@]}"

    echo "[RUN] robometer -> $video_path"
    "$ROBOMETER_PY" "$COMPUTE_SCRIPT" \
        --model robometer \
        --video-path "$video_path" \
        "${instruction_args[@]}"

    echo "[RUN] vlac -> $video_path"
    "$VLAC_PY" "$COMPUTE_SCRIPT" \
        --model vlac \
        --video-path "$video_path" \
        "${instruction_args[@]}" \
        "${vlac_args[@]}"

    echo "[RUN] robodopamine -> $video_path"
    if [[ -n "$wrist_path" && -f "$wrist_path" ]]; then
        "$ROBODOPAMINE_PY" "$COMPUTE_SCRIPT" \
            --model robodopamine \
            --video-path "$video_path" \
            "${instruction_args[@]}" \
            --wrist-video-path "$wrist_path"
    else
        "$ROBODOPAMINE_PY" "$COMPUTE_SCRIPT" \
            --model robodopamine \
            --video-path "$video_path" \
            "${instruction_args[@]}"
    fi

    echo "[RUN] render -> $video_path"
    "$STARVLA_PY" "$VIS_SCRIPT" \
        --video-path "$video_path" \
        "${instruction_args[@]}" \
        --models liv robometer vlac robodopamine \
        --plot-layout subplot

    echo "[DONE] source video: $video_path"
    echo "[DONE] overlay video: $overlay_path"
}

if [[ "${RUN_SIGNAL_DEMO:-0}" != "1" ]]; then
    echo "[INFO] Template only. Set RUN_SIGNAL_DEMO=1 and ROOT=/path/to/libero_rollout_dir to run the demo batch."
    exit 0
fi

# open_the_middle_drawer_of_the_cabinet
run_one "$ROOT/rollout_open_the_middle_drawer_of_the_cabinet_episode0_failure.mp4" \
    "$ROOT/rollout_open_the_middle_drawer_of_the_cabinet_episode1_success.mp4"
run_one "$ROOT/rollout_open_the_middle_drawer_of_the_cabinet_episode0_success.mp4" \
    "$ROOT/rollout_open_the_middle_drawer_of_the_cabinet_episode1_success.mp4"

# open_the_top_drawer_and_put_the_bowl_inside
run_one "$ROOT/rollout_open_the_top_drawer_and_put_the_bowl_inside_episode1_failure.mp4" \
    "$ROOT/rollout_open_the_top_drawer_and_put_the_bowl_inside_episode2_success.mp4"
run_one "$ROOT/rollout_open_the_top_drawer_and_put_the_bowl_inside_episode0_success.mp4" \
    "$ROOT/rollout_open_the_top_drawer_and_put_the_bowl_inside_episode1_success.mp4"

# push_the_plate_to_the_front_of_the_stove
run_one "$ROOT/rollout_push_the_plate_to_the_front_of_the_stove_episode4_failure.mp4" \
    "$ROOT/rollout_push_the_plate_to_the_front_of_the_stove_episode1_success.mp4"
run_one "$ROOT/rollout_push_the_plate_to_the_front_of_the_stove_episode0_success.mp4" \
    "$ROOT/rollout_push_the_plate_to_the_front_of_the_stove_episode1_success.mp4"

# put_the_bowl_on_the_plate
run_one "$ROOT/rollout_put_the_bowl_on_the_plate_episode1_failure.mp4" \
    "$ROOT/rollout_put_the_bowl_on_the_plate_episode2_success.mp4"
run_one "$ROOT/rollout_put_the_bowl_on_the_plate_episode0_success.mp4" \
    "$ROOT/rollout_put_the_bowl_on_the_plate_episode2_success.mp4"

# put_the_bowl_on_the_stove
# No failure rollout was present in the provided file list for this task.
run_one "$ROOT/rollout_put_the_bowl_on_the_stove_episode0_success.mp4" \
    "$ROOT/rollout_put_the_bowl_on_the_stove_episode1_success.mp4"

# put_the_bowl_on_top_of_the_cabinet
run_one "$ROOT/rollout_put_the_bowl_on_top_of_the_cabinet_episode4_failure.mp4" \
    "$ROOT/rollout_put_the_bowl_on_top_of_the_cabinet_episode1_success.mp4"
run_one "$ROOT/rollout_put_the_bowl_on_top_of_the_cabinet_episode0_success.mp4" \
    "$ROOT/rollout_put_the_bowl_on_top_of_the_cabinet_episode1_success.mp4"

# put_the_cream_cheese_in_the_bowl
run_one "$ROOT/rollout_put_the_cream_cheese_in_the_bowl_episode0_failure.mp4" \
    "$ROOT/rollout_put_the_cream_cheese_in_the_bowl_episode1_success.mp4"
run_one "$ROOT/rollout_put_the_cream_cheese_in_the_bowl_episode0_success.mp4" \
    "$ROOT/rollout_put_the_cream_cheese_in_the_bowl_episode1_success.mp4"

# put_the_wine_bottle_on_the_rack
run_one "$ROOT/rollout_put_the_wine_bottle_on_the_rack_episode2_failure.mp4" \
    "$ROOT/rollout_put_the_wine_bottle_on_the_rack_episode1_success.mp4"
run_one "$ROOT/rollout_put_the_wine_bottle_on_the_rack_episode0_success.mp4" \
    "$ROOT/rollout_put_the_wine_bottle_on_the_rack_episode1_success.mp4"

# put_the_wine_bottle_on_top_of_the_cabinet
run_one "$ROOT/rollout_put_the_wine_bottle_on_top_of_the_cabinet_episode0_failure.mp4" \
    "$ROOT/rollout_put_the_wine_bottle_on_top_of_the_cabinet_episode2_success.mp4"
run_one "$ROOT/rollout_put_the_wine_bottle_on_top_of_the_cabinet_episode1_success.mp4" \
    "$ROOT/rollout_put_the_wine_bottle_on_top_of_the_cabinet_episode2_success.mp4"

# turn_on_the_stove
run_one "$ROOT/rollout_turn_on_the_stove_episode6_failure.mp4" \
    "$ROOT/rollout_turn_on_the_stove_episode1_success.mp4"
run_one "$ROOT/rollout_turn_on_the_stove_episode0_success.mp4" \
    "$ROOT/rollout_turn_on_the_stove_episode1_success.mp4"
