#!/bin/bash
#SBATCH --job-name=build_libero_all_vlac
#SBATCH --partition=lrc-xlong
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=/home/n84416302/ProcessVLA/slurm/slurm_build_libero_all_vlac_%j.out
#SBATCH --error=/home/n84416302/ProcessVLA/slurm/slurm_build_libero_all_vlac_%j.err

set -euo pipefail

mkdir -p /home/n84416302/ProcessVLA/slurm

eval "$(conda shell.bash hook)"
conda activate starvla
cd /home/n84416302/ProcessVLA

export DATA_ROOT_DIR=${DATA_ROOT_DIR:-/home/n84416302/dataset/LEROBOT_LIBERO_DATA}
export DATA_MIX=${DATA_MIX:-libero_all}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/home/n84416302/ProcessVLA/vlac_cache_libero_all}
export SIGNAL_NAME=${SIGNAL_NAME:-vlac}
export SIGNAL_KIND=${SIGNAL_KIND:-value}
export REFERENCE_MODE=${REFERENCE_MODE:-same_task_random}
export REFERENCE_SEED=${REFERENCE_SEED:-42}
export VLAC_REF_NUM=${VLAC_REF_NUM:-6}
export VLAC_BATCH_NUM=${VLAC_BATCH_NUM:-5}
export VLAC_SKIP=${VLAC_SKIP:-5}
export VLAC_RICH=${VLAC_RICH:-false}
export VLAC_FRAME_SKIP=${VLAC_FRAME_SKIP:-false}
export VLAC_THINK=${VLAC_THINK:-false}
export OVERWRITE=${OVERWRITE:-false}
export MAX_TRAJECTORIES=${MAX_TRAJECTORIES:-}
export DEVICE=${DEVICE:-cuda:0}

export VLAC_PYTHON=${VLAC_PYTHON:-/home/n84416302/miniconda3/envs/VLAC/bin/python}
export VLAC_MODEL_PATH=${VLAC_MODEL_PATH:-/home/n84416302/ckpts/VLAC}
export VLAC_MODEL_TYPE=${VLAC_MODEL_TYPE:-internvl2}

export CRITIC4VLA_VLAC_PYTHON="${VLAC_PYTHON}"
export CRITIC4VLA_VLAC_MODEL_PATH="${VLAC_MODEL_PATH}"
export CRITIC4VLA_VLAC_MODEL_TYPE="${VLAC_MODEL_TYPE}"

python3 - <<'PY'
import os
from pathlib import Path

from examples.LIBERO.train_files.build_offline_vlac_signal_cache import (
    build_offline_vlac_signal_cache,
)


def parse_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, str(default))
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


max_trajectories_raw = os.environ.get("MAX_TRAJECTORIES", "").strip()
max_trajectories = int(max_trajectories_raw) if max_trajectories_raw else None

summary = build_offline_vlac_signal_cache(
    data_root_dir=Path(os.environ["DATA_ROOT_DIR"]).expanduser().resolve(),
    data_mix=os.environ["DATA_MIX"],
    output_root=Path(os.environ["OUTPUT_ROOT"]).expanduser().resolve(),
    signal_name=os.environ["SIGNAL_NAME"],
    signal_kind=os.environ["SIGNAL_KIND"],
    device=os.environ["DEVICE"],
    reference_mode=os.environ["REFERENCE_MODE"],
    reference_seed=int(os.environ["REFERENCE_SEED"]),
    vlac_ref_num=int(os.environ["VLAC_REF_NUM"]),
    vlac_batch_num=int(os.environ["VLAC_BATCH_NUM"]),
    vlac_skip=int(os.environ["VLAC_SKIP"]),
    vlac_rich=parse_bool("VLAC_RICH", False),
    vlac_frame_skip=parse_bool("VLAC_FRAME_SKIP", False),
    vlac_think=parse_bool("VLAC_THINK", False),
    vlac_python=Path(os.environ["VLAC_PYTHON"]).expanduser().resolve(),
    overwrite=parse_bool("OVERWRITE", False),
    max_trajectories=max_trajectories,
)

print("[summary]", summary)
PY
