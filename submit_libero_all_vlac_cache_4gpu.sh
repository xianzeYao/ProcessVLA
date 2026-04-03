#!/bin/bash
#SBATCH --job-name=qwen3vl_libero_all_vlac
#SBATCH --partition=lrc-xlong
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --output=/home/n84416302/ProcessVLA/slurm/slurm_libero_all_vlac_%j.out
#SBATCH --error=/home/n84416302/ProcessVLA/slurm/slurm_libero_all_vlac_%j.err

set -euo pipefail

mkdir -p /home/n84416302/ProcessVLA/slurm

eval "$(conda shell.bash hook)"
conda activate starvla
cd /home/n84416302/ProcessVLA

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  --main_process_port 29513 \
  starVLA/training/train_starvla.py \
  --config_yaml /home/n84416302/ProcessVLA/examples/LIBERO/train_files/starvla_cotrain_libero_vlac_cache.yaml \
  --run_id qwen3vl_gr00t_libero_all_4gpu_$(date +%Y%m%d_%H%M%S) \
  --run_root_dir /home/n84416302/ckpts/starvla_train \
  --framework.qwenvl.base_vlm /home/n84416302/ckpts/base_models/Qwen3-VL-4B-Instruct \
  --datasets.vla_data.data_root_dir /home/n84416302/dataset/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_all \
  --signal.cache_root /home/n84416302/ProcessVLA/vlac_cache_libero_all \
  --datasets.vlm_data.dataset_use asv2_conversation_en,asv2_detailed_description_en,asv2_region_captioning_en,coco_internvl_longcap_en,coco_karpathy_train_567_en,coco_negative_gpt4o_en,coco_poetry_zh,coco_rem_en_zh,cocorem_exist_yorn_en,cocotextv2_en,cocotextv2_gpt4o_en,okvqa_en,refcoco_grounding_aug_en,refcoco_grounding_en,tallyqa_coco_en,toloka_grounding_aug_en,vqav2_en,vsr_en \
  --datasets.vlm_data.eval_dataset aokvqa_cauldron_llava_format \
  --datasets.vlm_data.max_pixels 12845056 \
  --datasets.vlm_data.min_pixels 3136 \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 10000 \ 
  --trainer.eval_interval 1000 \
  --trainer.learning_rate.base 4.0e-05 \
  --trainer.logging_frequency 100 \
  --trainer.gradient_accumulation_steps 2 \
  --signal.train_source vlac_cache \
  --signal.infer_source none \
  --wandb_entity yao-xian-ze \
  --wandb_project starVLA4train
