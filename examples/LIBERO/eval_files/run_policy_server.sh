#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}" # let LIBERO find the websocket tools from main repo
export star_vla_python=${STAR_VLA_PY:-python3}
your_ckpt=${CKPT_PATH:-/path/to/checkpoint.pt}
gpu_id=${GPU_ID:-0}
port=${PORT:-5694}
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id "${star_vla_python}" deployment/model_server/server_policy.py \
    --ckpt_path "${your_ckpt}" \
    --port "${port}" \
    --use_bf16

# #################################
