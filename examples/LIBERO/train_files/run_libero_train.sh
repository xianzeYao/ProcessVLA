
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROCESSVLA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROCESSVLA_ROOT}"


export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenOFT
freeze_module_list=${FREEZE_MODULE_LIST:-}
base_vlm=${BASE_VLM:-${PROCESSVLA_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct}
config_yaml=${CONFIG_YAML:-${PROCESSVLA_ROOT}/examples/LIBERO/train_files/starvla_cotrain_libero.yaml}
libero_data_root=${LIBERO_DATA_ROOT:-${PROCESSVLA_ROOT}/playground/Datasets/LEROBOT_LIBERO_DATA}
data_mix=${DATA_MIX:-libero_all}
run_root_dir=${RUN_ROOT_DIR:-${PROCESSVLA_ROOT}/results/Checkpoints}
run_id=${RUN_ID:-1229_libero4in1_qwen3oft}
config_file=${CONFIG_FILE:-${PROCESSVLA_ROOT}/starVLA/config/deepseeds/deepspeed_zero2.yaml}
train_script=${TRAIN_SCRIPT:-${PROCESSVLA_ROOT}/starVLA/training/train_starvla.py}
wandb_project=${WANDB_PROJECT:-starVLA_Libero}
wandb_entity=${WANDB_ENTITY:-your_wandb_entity}
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp "$0" "${output_dir}/"


accelerate launch \
  --config_file "${config_file}" \
  --num_processes 8 \
  "${train_script}" \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --datasets.vla_data.data_root_dir "${libero_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project "${wandb_project}" \
  --wandb_entity "${wandb_entity}" \
  # --is_debug True



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
