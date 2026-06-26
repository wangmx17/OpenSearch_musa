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
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-34237}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# ===================== NCCL =====================
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export TORCH_NCCL_TIMEOUT=7200
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export MUSA_LAUNCH_BLOCKING=1 

# IB GID detection
SHOW_GIDS_BIN="${SHOW_GIDS_BIN:-/mnt/workspace/yeshenglong/tmp_0508/show_gids}"
if [[ -n "${SHOW_GIDS_BIN}" && -x "${SHOW_GIDS_BIN}" ]]; then
  NCCL_IB_GID_INDEX=""
  output="$("${SHOW_GIDS_BIN}" | grep v2 || true)"
  while IFS= read -r line; do
    ipv4="$(echo "${line}" | awk '{print $5}')"
    if [[ -n "${ipv4}" && "${ipv4}" != "0000:0000:0000:0000:0000:ffff:0000:0000" && "${ipv4}" =~ [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ ]]; then
      NCCL_IB_GID_INDEX="$(echo "${line}" | awk '{print $3}')"
      break
    fi
  done <<<"${output}"
  if [[ -n "${NCCL_IB_GID_INDEX}" ]]; then
    export NCCL_IB_GID_INDEX
  fi
fi
echo "NCCL_IB_GID_INDEX ----- ${NCCL_IB_GID_INDEX:-<unset>}"

export NCCL_IB_TC=${NCCL_IB_TC:-138}
export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-8}

# IB HCA detection
if [[ -z "${NCCL_IB_HCA:-}" ]]; then
  HCA_CANDIDATES="mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8,mlx5_bond_9"
  if [[ -d /sys/class/infiniband ]]; then
    HCA_LIST=""
    IFS=',' read -ra HCAS <<<"${HCA_CANDIDATES}"
    for hca in "${HCAS[@]}"; do
      [[ -e "/sys/class/infiniband/${hca}" ]] && HCA_LIST="${HCA_LIST:+${HCA_LIST},}${hca}"
    done
    [[ -n "${HCA_LIST}" ]] && export NCCL_IB_HCA="${HCA_LIST}"
  fi
fi

# Socket ifname detection
pick_ifname() {
  local name
  for name in bond0 ib0 ib1 eno1 ens1 enp1s0; do
    [[ -d "/sys/class/net/${name}" ]] && { echo "${name}"; return 0; }
  done
  return 1
}
if [[ -z "${NCCL_SOCKET_IFNAME:-}" || ! -d "/sys/class/net/${NCCL_SOCKET_IFNAME}" ]]; then
  if IFNAME="$(pick_ifname)"; then
    export NCCL_SOCKET_IFNAME="${IFNAME}"
  else
    unset NCCL_SOCKET_IFNAME
  fi
fi
if [[ -z "${GLOO_SOCKET_IFNAME:-}" || ! -d "/sys/class/net/${GLOO_SOCKET_IFNAME}" ]]; then
  if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
    export GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
  else
    unset GLOO_SOCKET_IFNAME
  fi
fi
echo "NCCL_SOCKET_IFNAME ----- ${NCCL_SOCKET_IFNAME:-<auto>}"
echo "GLOO_SOCKET_IFNAME ----- ${GLOO_SOCKET_IFNAME:-<auto>}"
echo "NCCL_IB_HCA ----- ${NCCL_IB_HCA:-<auto>}"

export TORCH_NCCL_TRACE_BUFFER_SIZE=1000

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

# ===================== Launch =====================
echo "[INFO] Launching multi-node training"
echo "       Master node: ${MASTER_ADDR}"
echo "       Nodes total: ${WORLD_SIZE}"
echo "       GPUs per node: ${NPROC_PER_NODE}"
echo "       NODE_RANK: ${RANK}"

export PYTHONPATH=/home/jd/OpenSearch-VL-main/SFT/src
cd /home/jd/OpenSearch-VL-main/SFT

YAML_CONFIG=/home/jd/OpenSearch-VL-main/SFT/examples/agentic_full/qwen3_vl_full_sft_8b.yaml
DEBUG_YAML="${EXP_LOG_DIR}/$(basename "${YAML_CONFIG}" .yaml).node_${RANK}.debug.yaml"
cp "${YAML_CONFIG}" "${DEBUG_YAML}"

echo "[INFO] Using yaml: ${DEBUG_YAML}"

FORCE_TORCHRUN=1 \
RDZV_ID=opensearch-vl-sft \
NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
python -m llamafactory.cli train "${DEBUG_YAML}"
