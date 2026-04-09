#!/bin/bash
#SBATCH --job-name=libero_goal_vlac_vis
#SBATCH --partition=lrc-xlong
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=/home/n84416302/ProcessVLA/slurm/slurm_vlac_vis_goal_%j.out
#SBATCH --error=/home/n84416302/ProcessVLA/slurm/slurm_vlac_vis_goal_%j.err

set -euo pipefail

mkdir -p /home/n84416302/ProcessVLA/slurm
cd /home/n84416302/ProcessVLA

export STAR_VLA_Python=/home/n84416302/miniconda3/envs/starvla/bin/python

export RESULT_ROOT=${RESULT_ROOT:-/home/n84416302/ProcessVLA/results/libero_eval/qwen3vl_gr00t_libero_all_4gpu_20260327_090314_steps_30000_pytorch_model.pt/results/libero_goal}
export SAMPLES_PER_TASK=${SAMPLES_PER_TASK:-3}
export TASK_IDS=${TASK_IDS:-"0 1 2 3 4 5 6 7 8 9"}
export OUTPUT_DIR=${OUTPUT_DIR:-${RESULT_ROOT}/_vlac_online_overlays_random${SAMPLES_PER_TASK}}
export PYTHON_BIN=${PYTHON_BIN:-${STAR_VLA_Python}}
export VIS_SCRIPT=${VIS_SCRIPT:-/home/n84416302/ProcessVLA/examples/LIBERO/eval_files/visualize_vlac_online_curve_with_video.py}

bash examples/LIBERO/eval_files/batch_visualize_vlac_online_random_samples.sh
