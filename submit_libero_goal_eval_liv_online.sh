#!/bin/bash
# Note: this overrides the checkpoint's eval signal to `liv_online`.
# If the checkpoint was trained with `vlac_cache`/`vlac_online` instead of `liv_online`,
# the run can execute, but the result is not directly comparable to a LIV-trained model.
#
# Requirements:
# - LIBERO env for simulator
# - STAR_VLA env for policy server
# - LIV env for subprocess signal inference
# - Pretrained LIV weights available under ~/.liv/resnet50 in the LIV env
#   or network access enabled for first-time auto-download
#
# Submit with:
#   sbatch submit_libero_goal_eval_liv_online.sh
#
# Override examples:
#   RUN_ROOT_DIR=/path/to/run STEPS="5000 10000" sbatch submit_libero_goal_eval_liv_online.sh

#SBATCH --job-name=qwen3vl_goal_eval_liv
#SBATCH --partition=lrc-xlong
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=/home/n84416302/ProcessVLA/slurm/slurm_eval_goal_liv_%j.out
#SBATCH --error=/home/n84416302/ProcessVLA/slurm/slurm_eval_goal_liv_%j.err

set -euo pipefail

mkdir -p /home/n84416302/ProcessVLA/slurm
cd /home/n84416302/ProcessVLA

export LIBERO_HOME=/home/n84416302/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/home/n84416302/miniconda3/envs/libero/bin/python
export STAR_VLA_Python=/home/n84416302/miniconda3/envs/starvla/bin/python

# Dedicated LIV subprocess runtime.
export LIV_PYTHON=${LIV_PYTHON:-/home/n84416302/miniconda3/envs/liv/bin/python}
export LIV_REPO_ROOT=${LIV_REPO_ROOT:-/home/n84416302/ProcessVLA/LIV}
export LIV_DEVICE=${LIV_DEVICE:-cuda}

# Keep the old env for direct-import fallback paths.
export CRITIC4VLA_LIV_REPO_ROOT=${CRITIC4VLA_LIV_REPO_ROOT:-${LIV_REPO_ROOT}}

export RUN_ROOT_DIR=/home/n84416302/ProcessVLA/ckpts/qwen3vl_gr00t_libero_all_4gpu_20260327_090314
export TASK_SUITE_NAME=libero_goal
export NUM_TRIALS_PER_TASK=50
export STEPS="30000"
export LIBERO_EVAL_MODE=subprocess_episode
export LIBERO_EPISODE_TIMEOUT_SECONDS=1800

export INFER_SOURCE=liv_online

export GPU_ID=0
export PORT=5694
export WAIT_SERVER_SECONDS=15

bash examples/LIBERO/eval_files/multisteppt_eval4libero1task.sh
