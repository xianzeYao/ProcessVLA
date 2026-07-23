#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${CALVIN_PYTHON:-/root/data/yxz/miniforge3/envs/calvin/bin/python}"
HF_ROOT="${CALVIN_HF_ROOT:-/root/data/yxz/datasets/calvin_hf}"
CONFIG_SOURCE="${CALVIN_CONFIG_SOURCE:-/root/data/yxz/datasets/calvin/task_D_D/training/.hydra}"
MODE="${1:-all}"

prepare_abcd_d() {
  local archive_root="$HF_ROOT/ABCD_D"
  local raw_root="$HF_ROOT/task_ABCD_D"
  for index in $(seq -f "%03g" 0 23); do
    local archive="$archive_root/subset_training_$index.zip"
    test -f "$archive" || { echo "missing $archive; run fetch_calvin_raw_shards.sh abcd_d" >&2; exit 2; }
    if ! test -f "$archive_root/subset_training_$index/training/ep_start_end_ids.npy"; then
      unzip -q "$archive" -d "$archive_root"
    fi
  done
  "$PYTHON_BIN" -m examples.simBenchmarks.calvin.data_preparation.merge_calvin_frame_shards \
    --input-root "$archive_root" --output-root "$raw_root" --overwrite --mode hardlink
  if ! test -e "$raw_root/training/.hydra/merged_config.yaml"; then
    cp -a "$CONFIG_SOURCE" "$raw_root/training/.hydra"
  fi
  echo "prepared $raw_root"
}

prepare_abc_d() {
  local archive_root="$HF_ROOT/ABC_D"
  local raw_root="$HF_ROOT/task_ABC_D"
  for index in $(seq -w 0 23); do
    test -f "$archive_root/packaged_ABC_D$index.tar.gz" || {
      echo "missing packaged_ABC_D$index.tar.gz; run fetch_calvin_raw_shards.sh abc_d" >&2
      exit 2
    }
  done
  if ! test -f "$raw_root/training/ep_start_end_ids.npy"; then
    mkdir -p "$HF_ROOT"
    cat "$archive_root"/packaged_ABC_D??.tar.gz | tar -xzf - -C "$HF_ROOT"
  fi
  test -f "$raw_root/training/ep_start_end_ids.npy" || {
    echo "ABC_D archive did not extract to $raw_root; inspect tar layout" >&2
    exit 3
  }
  if ! test -e "$raw_root/training/.hydra/merged_config.yaml"; then
    cp -a "$CONFIG_SOURCE" "$raw_root/training/.hydra"
  fi
  echo "prepared $raw_root"
}

case "$MODE" in
  abc_d) prepare_abc_d ;;
  abcd_d) prepare_abcd_d ;;
  all) prepare_abc_d; prepare_abcd_d ;;
  *) echo "usage: $0 {abc_d|abcd_d|all}" >&2; exit 2 ;;
esac
