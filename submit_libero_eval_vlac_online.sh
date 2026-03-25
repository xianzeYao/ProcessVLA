#!/bin/bash
#SBATCH --job-name=qwen3vl_libero_eval_vlac
#SBATCH --partition=lrc-xlong
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=/home/n84416302/ProcessVLA/slurm/slurm_eval_%j.out
#SBATCH --error=/home/n84416302/ProcessVLA/slurm/slurm_eval_%j.err

set -euo pipefail

mkdir -p /home/n84416302/ProcessVLA/slurm
cd /home/n84416302/ProcessVLA

export LIBERO_HOME=/home/n84416302/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/home/n84416302/miniconda3/envs/libero/bin/python
export STAR_VLA_Python=/home/n84416302/miniconda3/envs/starvla/bin/python

export RUN_ROOT_DIR=/home/n84416302/ProcessVLA/ckpts/qwen3vl_gr00t_libero_goal_4gpu_20260323_183741
export TASK_SUITE_NAME=libero_goal
export NUM_TRIALS_PER_TASK=5
export STEPS="10000 30000"
export LIBERO_EVAL_MODE=subprocess_episode
export LIBERO_EPISODE_TIMEOUT_SECONDS=1800

export INFER_SOURCE=vlac_online
export VLAC_REFERENCE_MODE=dataset_same_task_seeded
export VLAC_REFERENCE_DATASET_NAME=libero_goal_no_noops_1.0.0_lerobot
export VLAC_REFERENCE_SEED=10
export VLAC_REFERENCE_DATA_ROOT_DIR=/home/n84416302/dataset/LEROBOT_LIBERO_DATA
export VLAC_REFERENCE_DATA_MIX=libero_goal

export VLAC_SIGNAL_KIND=value
export VLAC_SKIP=5
export VLAC_FRAME_SKIP=false
export VLAC_REF_NUM=6
export VLAC_BATCH_NUM=5
export VLAC_THINK=false
export VLAC_RICH=false
export VLAC_DEVICE=cuda

export VLAC_PYTHON=/home/n84416302/miniconda3/envs/VLAC/bin/python
export VLAC_MODEL_PATH=/home/n84416302/ckpts/VLAC
export VLAC_MODEL_TYPE=internvl2
export VLAC_REPO_ROOT=/home/n84416302/ProcessVLA/VLAC

export GPU_ID=0
export PORT=5694
export WAIT_SERVER_SECONDS=15

bash examples/LIBERO/eval_files/multisteppt_eval4libero1task.sh
