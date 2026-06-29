#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
LOG_DIR="${PROJECT_ROOT}/logs"
HOSTFILE="${HOSTFILE:-${PROJECT_ROOT}/hostfile.txt}"
mkdir -p "${LOG_DIR}"

pick_first_host() {
  local file="$1"
  if [[ -f "${file}" ]]; then
    awk 'NF {print $1; exit}' "${file}"
  fi
}

resolve_node_rank() {
  local file="$1"
  local host="${CURRENT_HOSTNAME:-}"

  if [[ -n "${NODE_RANK:-}" ]]; then
    echo "${NODE_RANK}"
    return 0
  fi

  if [[ -n "${OMPI_COMM_WORLD_RANK:-}" ]]; then
    echo "${OMPI_COMM_WORLD_RANK}"
    return 0
  fi

  if [[ -n "${SLURM_NODEID:-}" ]]; then
    echo "${SLURM_NODEID}"
    return 0
  fi

  if [[ -n "${RANK:-}" && -z "${LOCAL_RANK:-}" && -z "${WORLD_SIZE:-}" ]]; then
    echo "${RANK}"
    return 0
  fi

  if [[ -z "${host}" ]]; then
    host="$(cat /etc/hostname 2>/dev/null || true)"
  fi
  host="${host%%.*}"

  if [[ -n "${host}" && -f "${file}" ]]; then
    awk -v host="${host}" '
      NF {
        split($1, parts, ".")
        if ($1 == host || parts[1] == host) {
          print NR - 1
          exit
        }
      }
    ' "${file}"
    return 0
  fi

  echo "0"
}

resolve_nnodes() {
  local file="$1"

  if [[ -n "${NNODES:-}" ]]; then
    echo "${NNODES}"
    return 0
  fi

  if [[ -n "${WORLD_SIZE:-}" && -z "${LOCAL_WORLD_SIZE:-}" ]]; then
    echo "${WORLD_SIZE}"
    return 0
  fi

  if [[ -n "${OMPI_COMM_WORLD_SIZE:-}" ]]; then
    echo "${OMPI_COMM_WORLD_SIZE}"
    return 0
  fi

  if [[ -f "${file}" ]]; then
    awk 'NF {count += 1} END {print count + 0}' "${file}"
    return 0
  fi

  echo "1"
}

pick_socket_ifname() {
  local name
  for name in bond1 bond0 ib0 ib1 eth0 eno1 ens1 enp1s0; do
    [[ -d "/sys/class/net/${name}" ]] && { echo "${name}"; return 0; }
  done
  return 1
}

pick_mccl_ib_device() {
  local name
  for name in mubd0 mubd1 mlx5_bond_0 mlx5_bond_1; do
    [[ -d "/sys/class/infiniband/${name}" ]] && { echo "${name}"; return 0; }
  done
  return 1
}

cd "${PROJECT_ROOT}"

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  pip install -e ".[metrics,deepspeed]" -i https://pypi.tuna.tsinghua.edu.cn/simple
  pip install qwen-vl-utils pillow av decord torchvision flash-attn -i https://pypi.tuna.tsinghua.edu.cn/simple
  pip install swanlab==0.7.16 -i https://pypi.tuna.tsinghua.edu.cn/simple
  pip install metrics -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

CURRENT_HOSTNAME="$(cat /etc/hostname 2>/dev/null || true)"
MASTER_ADDR="${MASTER_ADDR:-$(pick_first_host "${HOSTFILE}")}"
MASTER_PORT="${MASTER_PORT:-34237}"
NNODES="$(resolve_nnodes "${HOSTFILE}")"
NODE_RANK="$(resolve_node_rank "${HOSTFILE}")"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

if [[ -z "${MASTER_ADDR}" ]]; then
  echo "[ERROR] MASTER_ADDR is empty. Set MASTER_ADDR explicitly or provide a valid ${HOSTFILE}."
  exit 1
fi

if ! [[ "${NNODES}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] NNODES must be an integer, got: ${NNODES}"
  exit 1
fi

if ! [[ "${NODE_RANK}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] NODE_RANK must be an integer, got: ${NODE_RANK}"
  exit 1
fi

if (( NODE_RANK >= NNODES )); then
  echo "[ERROR] NODE_RANK (${NODE_RANK}) must be smaller than NNODES (${NNODES})."
  exit 1
fi

# ===================== Log timestamp (multi-node safe) =====================
if [[ -z "${LOG_TIMESTAMP:-}" ]]; then
  LOG_TIMESTAMP_FILE="${LOG_DIR}/CURRENT_LOG_TIMESTAMP"
  if [[ "${NODE_RANK}" == "0" ]]; then
    LOCK_DIR="${LOG_DIR}/.log_timestamp.lock"
    until mkdir "${LOCK_DIR}" 2>/dev/null; do
      sleep 0.2
    done
    trap 'rm -rf "${LOCK_DIR}"' EXIT
    LOG_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    while [[ -e "${LOG_DIR}/${LOG_TIMESTAMP}" ]]; do
      LOG_TIMESTAMP="$(date +%Y%m%d_%H%M%S_%N)"
    done
    mkdir -p "${LOG_DIR}/${LOG_TIMESTAMP}"
    echo "${LOG_TIMESTAMP}" > "${LOG_DIR}/${LOG_TIMESTAMP}/LOG_TIMESTAMP"
    echo "${LOG_TIMESTAMP}" > "${LOG_TIMESTAMP_FILE}.tmp"
    mv "${LOG_TIMESTAMP_FILE}.tmp" "${LOG_TIMESTAMP_FILE}"
    rm -rf "${LOCK_DIR}"
    trap - EXIT
  else
    while [[ ! -s "${LOG_TIMESTAMP_FILE}" ]]; do
      sleep 0.2
    done
    LOG_TIMESTAMP="$(cat "${LOG_TIMESTAMP_FILE}")"
  fi
fi

EXP_LOG_DIR="${LOG_DIR}/${LOG_TIMESTAMP}"
mkdir -p "${EXP_LOG_DIR}"
DEBUG_LOG="${EXP_LOG_DIR}/rank_${NODE_RANK}.log"
: > "${DEBUG_LOG}"
exec > >(tee "${DEBUG_LOG}") 2>&1

echo "[INFO] Writing debug log to ${DEBUG_LOG}"
echo "[INFO] Log timestamp dir: ${EXP_LOG_DIR}"
echo "[INFO] Hostfile: ${HOSTFILE}"
echo "[INFO] Current host: ${CURRENT_HOSTNAME:-unknown}"

# ===================== Distributed env =====================
export NNODES
export NODE_RANK
export MASTER_ADDR
export MASTER_PORT
export NPROC_PER_NODE

if [[ -z "${MCCL_SOCKET_IFNAME:-}" || ! -d "/sys/class/net/${MCCL_SOCKET_IFNAME}" ]]; then
  if IFNAME="$(pick_socket_ifname)"; then
    export MCCL_SOCKET_IFNAME="${IFNAME}"
  fi
fi

if [[ -z "${GLOO_SOCKET_IFNAME:-}" || ! -d "/sys/class/net/${GLOO_SOCKET_IFNAME}" ]]; then
  if [[ -n "${MCCL_SOCKET_IFNAME:-}" ]]; then
    export GLOO_SOCKET_IFNAME="${MCCL_SOCKET_IFNAME}"
  fi
fi

if [[ -z "${MCCL_IB_HCA:-}" ]]; then
  if HCA="$(pick_mccl_ib_device)"; then
    export MCCL_IB_HCA="${HCA}"
  fi
fi

# ===================== MUSA / MCCL =====================
export ACCELERATOR_BACKEND="${ACCELERATOR_BACKEND:-musa}"
export MUSA_LAUNCH_BLOCKING="${MUSA_LAUNCH_BLOCKING:-0}"
export MUSA_EXECUTION_TIMEOUT="${MUSA_EXECUTION_TIMEOUT:-3200000}"
export MCCL_PROTOS="${MCCL_PROTOS:-2}"
export MCCL_ALGOS="${MCCL_ALGOS:-1}"
export MCCL_BUFFSIZE="${MCCL_BUFFSIZE:-20971520}"
export MCCL_MAX_NCHANNELS="${MCCL_MAX_NCHANNELS:-14}"
export MCCL_CHECK_POINTERS="${MCCL_CHECK_POINTERS:-0}"
export MCCL_IB_GID_INDEX="${MCCL_IB_GID_INDEX:-3}"
export MCCL_IB_TC="${MCCL_IB_TC:-41}"
export MCCL_IB_TIMEOUT="${MCCL_IB_TIMEOUT:-22}"
export MCCL_DEBUG="${MCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export MUSA_LAUNCH_BLOCKING=1 
# ===================== Misc env =====================
export NVTE_FP8=0
export NVTE_DISABLE_FP8=1
export NVTE_FUSED_ATTN=0
export NVTE_LAYERNORM_FWD_USE_CUDNN=0
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export GOMAXPROCS="${GOMAXPROCS:-8}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export ARROW_NUM_THREADS="${ARROW_NUM_THREADS:-1}"
export PYTORCH_MUSA_ALLOC_CONF="${PYTORCH_MUSA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_MCCL_AVOID_RECORD_STREAMS="${TORCH_MCCL_AVOID_RECORD_STREAMS:-1}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# ===================== Launch =====================
YAML_CONFIG="${YAML_CONFIG:-${PROJECT_ROOT}/examples/agentic_full/qwen3_vl_full_sft_8b_every_step.yaml}"
DEBUG_YAML="${EXP_LOG_DIR}/$(basename "${YAML_CONFIG}" .yaml).node_${NODE_RANK}.yaml"
cp "${YAML_CONFIG}" "${DEBUG_YAML}"

echo "[INFO] Launching multi-node training"
echo "       Master node: ${MASTER_ADDR}:${MASTER_PORT}"
echo "       Nodes total: ${NNODES}"
echo "       Node rank: ${NODE_RANK}"
echo "       GPUs per node: ${NPROC_PER_NODE}"
echo "       MCCL_SOCKET_IFNAME: ${MCCL_SOCKET_IFNAME:-<auto>}"
echo "       GLOO_SOCKET_IFNAME: ${GLOO_SOCKET_IFNAME:-<auto>}"
echo "       MCCL_IB_HCA: ${MCCL_IB_HCA:-<auto>}"
echo "[INFO] Using yaml: ${DEBUG_YAML}"

FORCE_TORCHRUN=1 \
RDZV_ID="${RDZV_ID:-opensearch-vl-sft}" \
NNODES="${NNODES}" \
NODE_RANK="${NODE_RANK}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
python -m llamafactory.cli train "${DEBUG_YAML}"

