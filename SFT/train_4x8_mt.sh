#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

cd /home/jd/OpenSearch-VL-main/SFT

pip install -e ".[metrics,deepspeed]" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install qwen-vl-utils pillow av decord torchvision flash-attn -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install swanlab==0.7.16 -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip install deepspeed -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install metrics -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install metrics
# pip install peft==0.18.1 -i https://pypi.tuna.tsinghua.edu.cn/simple


# ===================== Experiment ID (multi-node safe) =====================
if [[ -z "${EXP_ID:-}" ]]; then
  EXP_ID_FILE="${LOG_DIR}/CURRENT_EXP_ID"
  if [[ "${RANK:-0}" == "0" ]]; then
    LOCK_DIR="${LOG_DIR}/.exp_alloc.lock"
    until mkdir "${LOCK_DIR}" 2>/dev/null; do
      sleep 0.2
    done
    trap 'rm -rf "${LOCK_DIR}"' EXIT
    EXP_ID="$(
      find "${LOG_DIR}" -maxdepth 1 -type d -name 'exp_*' -printf '%f\n' 2>/dev/null \
        | sed -n 's/^exp_\([0-9][0-9]*\)$/\1/p' \
        | sort -n \
        | tail -1
    )"
    EXP_ID="${EXP_ID:-0}"
    EXP_ID="$((EXP_ID + 1))"
    mkdir -p "${LOG_DIR}/exp_${EXP_ID}"
    echo "${EXP_ID}" > "${LOG_DIR}/exp_${EXP_ID}/EXP_ID"
    echo "${EXP_ID}" > "${EXP_ID_FILE}.tmp"
    mv "${EXP_ID_FILE}.tmp" "${EXP_ID_FILE}"
    rm -rf "${LOCK_DIR}"
    trap - EXIT
  else
    while [[ ! -s "${EXP_ID_FILE}" ]]; do
      sleep 0.2
    done
    EXP_ID="$(cat "${EXP_ID_FILE}")"
  fi
fi

EXP_LOG_DIR="${LOG_DIR}/exp_${EXP_ID}"
mkdir -p "${EXP_LOG_DIR}/node_logs"
DEBUG_LOG="${EXP_LOG_DIR}/debug_rank_${RANK:-0}.log"
: > "${DEBUG_LOG}"
exec > >(tee "${DEBUG_LOG}") 2>&1
echo "[INFO] Writing debug log to ${DEBUG_LOG}"
echo "[INFO] Experiment log dir: ${EXP_LOG_DIR}"

# ===================== Distributed env =====================
export WORLD_SIZE=${WORLD_SIZE:-4}
export HOSTFILE=${HOSTFILE:-/home/jd/OpenSearch-VL-main/SFT/hostfile.txt}
export MASTER_ADDR=${MASTER_ADDR:-10.121.33.67}
export MASTER_PORT=${MASTER_PORT:-34237}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# ===================== MCCL =====================
export MUSA_LAUNCH_BLOCKING=1 

export MUSA_EXECUTION_TIMEOUT="${MUSA_EXECUTION_TIMEOUT:-3200000}"
export ACCELERATOR_BACKEND="${ACCELERATOR_BACKEND:-musa}"

export MCCL_PROTOS="${MCCL_PROTOS:-2}"
export MCCL_ALGOS="${MCCL_ALGOS:-1}"
export MCCL_BUFFSIZE="${MCCL_BUFFSIZE:-20971520}"
export MCCL_MAX_NCHANNELS="${MCCL_MAX_NCHANNELS:-14}"
export MCCL_CHECK_POINTERS="${MCCL_CHECK_POINTERS:-0}"
export MCCL_IB_GID_INDEX="${MCCL_IB_GID_INDEX:-3}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# ===================== Misc env =====================
export NVTE_FP8=0
export NVTE_DISABLE_FP8=1
export NVTE_FUSED_ATTN=0
export NVTE_LAYERNORM_FWD_USE_CUDNN=0
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export GOMAXPROCS=8
export TORCH_NUM_THREADS=1
export ARROW_NUM_THREADS=1
export PYTORCH_MUSA_ALLOC_CONF="expandable_segments:True"
export TORCH_MCCL_AVOID_RECORD_STREAMS=1
# ===================== Launch =====================
echo "[INFO] Launching multi-node training"
echo "       Master node: ${MASTER_ADDR}"
echo "       Nodes total: ${WORLD_SIZE}"
echo "       GPUs per node: ${NPROC_PER_NODE}"
#echo "       NODE_RANK: ${RANK}"

export PYTHONPATH=/home/jd/OpenSearch-VL-main/SFT/src
cd /home/jd/OpenSearch-VL-main/SFT

YAML_CONFIG=/home/jd/OpenSearch-VL-main/SFT/examples/agentic_full/qwen3_vl_full_sft_30_3b.yaml
#DEBUG_YAML="${EXP_LOG_DIR}/$(basename "${YAML_CONFIG}" .yaml).node_${RANK}.debug.yaml"
DEBUG_YAML="${EXP_LOG_DIR}/$(basename "${YAML_CONFIG}" .yaml).debug.yaml"

cp "${YAML_CONFIG}" "${DEBUG_YAML}"

echo "[INFO] Using yaml: ${DEBUG_YAML}"

FORCE_TORCHRUN=1 \
RDZV_ID=opensearch-vl-sft \
NNODES=$WORLD_SIZE \
#NODE_RANK=$RANK \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
python -m llamafactory.cli train "${DEBUG_YAML}"
