#!/bin/bash
set -euo pipefail

srun --qos normal --partition=lrc-dev --ntasks=1 --gres=gpu:h200:2 --time=01:59:59 --pty bash -lc '
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate dream_vla_temp
cd /home/n84416302/ProcessVLA

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 2 \
  --main_process_port 29513 \
  starVLA/training/train_starvla.py \
  --config_yaml /home/n84416302/ProcessVLA/examples/LIBERO/train_files/starvla_cotrain_libero_vlac_cache.yaml \
  --run_id qwen3vl_gr00t_libero_goal_vlac_cache_test_$(date +%Y%m%d_%H%M%S) \
  --run_root_dir /home/n84416302/ckpts/starvla_train \
  --datasets.vlm_data.dataset_use asv2_conversation_en,asv2_detailed_description_en,asv2_region_captioning_en,coco_internvl_longcap_en,coco_karpathy_train_567_en,coco_negative_gpt4o_en,coco_poetry_zh,coco_rem_en_zh,cocorem_exist_yorn_en,cocotextv2_en,cocotextv2_gpt4o_en,okvqa_en,refcoco_grounding_aug_en,refcoco_grounding_en,tallyqa_coco_en,toloka_grounding_aug_en,vqav2_en,vsr_en \
  --datasets.vlm_data.eval_dataset aokvqa_cauldron_llava_format \
  --datasets.vlm_data.max_pixels 12845056 \
  --datasets.vlm_data.min_pixels 3136 \
  --trainer.learning_rate.base 4.0e-05 \
  --trainer.gradient_accumulation_steps 2 \
  --trainer.max_train_steps 20 \
  --trainer.save_interval 20 \
  --trainer.eval_interval 20 \
  --trainer.logging_frequency 1
'
