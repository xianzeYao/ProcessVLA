#!/usr/bin/env bash
set -euo pipefail

ROOT="${CALVIN_HF_ROOT:-/root/data/yxz/datasets/calvin_hf}"
ABC_ROOT="$ROOT/ABC_D"
ABCD_ROOT="$ROOT/ABCD_D"
MODE="${1:-all}"
mkdir -p "$ABC_ROOT" "$ABCD_ROOT"

download_abc_d() {
  for index in $(seq -w 0 23); do
    file="packaged_ABC_D$index.tar.gz"
    curl -L --fail --retry 10 --retry-delay 10 --continue-at - \
      -o "$ABC_ROOT/$file" \
      "https://huggingface.co/datasets/nikooo6666/calvin_packaged_ABC_D/resolve/main/$file"
  done
}

download_abcd_d() {
  for index in $(seq -f "%03g" 0 23); do
    file="subset_training_$index.zip"
    if test -f "$ABCD_ROOT/$file" && unzip -t "$ABCD_ROOT/$file" >/dev/null 2>&1; then echo "skip complete $file"; continue; fi
    curl -L --fail --retry 10 --retry-delay 10 --continue-at - \
      -o "$ABCD_ROOT/$file" \
      "https://huggingface.co/datasets/VyoJ/calvin-ABCD-D-subsets/resolve/main/training/$file"
  done
}

case "$MODE" in
  abc_d) download_abc_d ;;
  abcd_d) download_abcd_d ;;
  all) download_abc_d; download_abcd_d ;;
  *) echo "usage: $0 {abc_d|abcd_d|all}" >&2; exit 2 ;;
esac
