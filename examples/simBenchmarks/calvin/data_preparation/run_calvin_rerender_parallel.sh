#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
cd "$ROOT_DIR"
CALVIN_PYTHON="${CALVIN_PYTHON:-/root/data/yxz/miniforge3/envs/calvin/bin/python}"
RAW_ROOT="${1:?usage: $0 RAW_ROOT OUTPUT_ROOT}"
OUTPUT_ROOT="${2:?usage: $0 RAW_ROOT OUTPUT_ROOT}"
NUM_WORKERS="${CALVIN_RERENDER_WORKERS:-4}"
GPU_LIST="${CALVIN_RERENDER_GPUS:-4,5,6,7}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

test -f "$RAW_ROOT/training/ep_start_end_ids.npy" || { echo "missing raw dataset: $RAW_ROOT" >&2; exit 2; }
test -f "$RAW_ROOT/training/lang_annotations/auto_lang_ann.npy" || { echo "missing language annotations: $RAW_ROOT" >&2; exit 2; }
test "${#GPUS[@]}" -ge "$NUM_WORKERS" || { echo "need at least $NUM_WORKERS GPUs in CALVIN_RERENDER_GPUS" >&2; exit 2; }

TOTAL_SEGMENTS="$("$CALVIN_PYTHON" - "$RAW_ROOT" <<'PY'
import sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1]) / "training"
ann = np.load(root / "lang_annotations" / "auto_lang_ann.npy", allow_pickle=True).item()
print(len(ann["info"]["indx"]))
PY
)"
test "$TOTAL_SEGMENTS" -gt 0 || { echo "no language segments found" >&2; exit 3; }
CHUNK=$(( (TOTAL_SEGMENTS + NUM_WORKERS - 1) / NUM_WORKERS ))
mkdir -p "$OUTPUT_ROOT"

PIDS=()
for ((worker=0; worker<NUM_WORKERS; worker++)); do
  start=$((worker * CHUNK))
  count=$((TOTAL_SEGMENTS - start))
  (( count > CHUNK )) && count=$CHUNK
  (( count > 0 )) || continue
  worker_root="$OUTPUT_ROOT/worker-$worker"
  mkdir -p "$worker_root"
  gpu="${GPUS[$worker]}"
  echo "starting worker=$worker gpu=$gpu start=$start count=$count output=$worker_root"
  CUDA_VISIBLE_DEVICES="$gpu" "$CALVIN_PYTHON" -m \
    examples.simBenchmarks.calvin.data_preparation.rerender_calvin_segments \
    --dataset-root "$RAW_ROOT" --output-root "$worker_root" \
    --max-segments "$count" --start-segment "$start" \
    --source-split training --config-split training --no-comparison --overwrite \
    >"$worker_root/run.log" 2>&1 &
  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if (( failed )); then
  echo "one or more rerender workers failed; inspect worker-*/run.log" >&2
  exit 4
fi

H5_COUNT="$(find "$OUTPUT_ROOT" -type f -name 'episode_*.h5' | wc -l)"
cat >"$OUTPUT_ROOT/parallel_summary.json" <<EOF
{
  "status": "success",
  "dataset_root": "${RAW_ROOT}",
  "output_root": "${OUTPUT_ROOT}",
  "workers": ${NUM_WORKERS},
  "segments_requested": ${TOTAL_SEGMENTS},
  "rerender_h5_files": ${H5_COUNT},
  "comparison_visualizations": false
}
EOF
test "$H5_COUNT" -eq "$TOTAL_SEGMENTS" || {
  echo "H5 count $H5_COUNT != expected segments $TOTAL_SEGMENTS" >&2
  exit 5
}
echo "parallel rerender complete: $H5_COUNT H5 files under $OUTPUT_ROOT"
