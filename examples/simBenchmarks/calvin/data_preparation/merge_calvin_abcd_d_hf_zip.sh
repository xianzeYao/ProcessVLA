#!/usr/bin/env bash
set -euo pipefail

# Reassemble the 22-volume ABCD_D archive downloaded from the HF shard mirror.
# This intentionally does not delete the downloaded volumes.
ROOT="${CALVIN_DATA_ROOT:-/root/data/yxz/datasets/calvin}"
SHARDS="${CALVIN_ABCD_D_SHARDS:-$ROOT/ABCD_D_hf_shards}"
RAW_ROOT="${CALVIN_ABCD_D_RAW_ROOT:-$ROOT/task_ABCD_D}"
BASE="$SHARDS/calvin_ABCD_D.zip"
MERGED="$SHARDS/calvin_ABCD_D_merged.zip"

test -d "$SHARDS" || { echo "missing shard directory: $SHARDS" >&2; exit 2; }
test -f "$BASE" || { echo "missing final volume: $BASE" >&2; exit 2; }
for index in $(seq -w 1 21); do
  test -f "$SHARDS/calvin_ABCD_D.z$index" || {
    echo "missing volume: $SHARDS/calvin_ABCD_D.z$index" >&2
    exit 2
  }
done

if test -f "$MERGED"; then
  echo "using existing merged archive: $MERGED"
else
  echo "checking split archive and writing: $MERGED"
  zip -F "$BASE" --out "$MERGED"
fi

if test -f "$RAW_ROOT/training/ep_start_end_ids.npy"; then
  echo "raw dataset already extracted: $RAW_ROOT"
  exit 0
fi

EXTRACT_ROOT="$SHARDS/.abcd_d_extract"
mkdir -p "$EXTRACT_ROOT"
echo "extracting merged archive to: $EXTRACT_ROOT"
unzip -q "$MERGED" -d "$EXTRACT_ROOT"

TRAIN_META="$(find "$EXTRACT_ROOT" -type f -path '*/training/ep_start_end_ids.npy' -print -quit)"
test -n "$TRAIN_META" || {
  echo "archive extracted but no training/ep_start_end_ids.npy was found" >&2
  exit 3
}
DATASET_ROOT="$(dirname "$(dirname "$TRAIN_META")")"
mkdir -p "$RAW_ROOT"
if test "$DATASET_ROOT" = "$RAW_ROOT"; then
  :
else
  cp -a "$DATASET_ROOT"/. "$RAW_ROOT"/
fi

test -f "$RAW_ROOT/training/ep_start_end_ids.npy" || {
  echo "normalized raw dataset is missing episode boundaries: $RAW_ROOT" >&2
  exit 4
}
echo "prepared raw dataset: $RAW_ROOT"
echo "merged archive retained: $MERGED"
echo "split volumes retained: $SHARDS/calvin_ABCD_D.z01 ... .z21"
