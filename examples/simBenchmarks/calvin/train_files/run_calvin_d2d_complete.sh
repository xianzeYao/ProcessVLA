#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/root/data/yxz/miniforge3/envs/CoT/bin/python}"
CALVIN_PYTHON="${CALVIN_PYTHON:-/root/data/yxz/miniforge3/envs/calvin/bin/python}"
RAW_ROOT="${CALVIN_RAW_ROOT:-/root/data/yxz/datasets/calvin/task_D_D}"
OUT_ROOT="${CALVIN_LEROBOT_ROOT:-/root/data/yxz/datasets/calvin_d2d_lerobot/calvin_task_D_D_v3.0}"

case "${1:-train}" in
  setup)
    examples/simBenchmarks/calvin/train_files/setup_calvin_env_v2.sh
    ;;
  convert)
    "$PYTHON_BIN" -m examples.simBenchmarks.calvin.data_preparation.build_calvin_starvla_lerobot \
      --dataset-root "$RAW_ROOT" --output-root "$OUT_ROOT" --overwrite
    ;;
  verify)
    "$PYTHON_BIN" examples/simBenchmarks/calvin/data_preparation/verify_calvin_lerobot.py "$OUT_ROOT"
    ;;
  replay-probe)
    "$CALVIN_PYTHON" -m examples.simBenchmarks.calvin.data_preparation.probe_calvin_replay_rgbd_rgb_only "$RAW_ROOT"
    ;;
  train)
    test -f "$OUT_ROOT/meta/info.json" || { echo "Run '$0 convert' first." >&2; exit 2; }
    WANDB_MODE="${WANDB_MODE:-offline}" accelerate launch \
      --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
      --num_processes "${NUM_GPUS:-8}" starVLA/training/train_starvla.py \
      --config_yaml examples/simBenchmarks/calvin/train_files/starvla_train_calvin_d2d.yaml \
      --datasets.vla_data.data_root_dir "$(dirname "$OUT_ROOT")" \
      --datasets.vla_data.data_mix calvin_task_D_D_v3.0 \
      --datasets.vla_data.video_backend decord \
      --run_root_dir "${RUN_ROOT_DIR:-results/Checkpoints}" \
      --run_id "${RUN_ID:-calvin_d2d}" "${@:2}"
    ;;
  *)
    echo "usage: $0 {setup|convert|verify|replay-probe|train} [extra train args]" >&2
    exit 2
    ;;
esac
