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

# ===================== Experiment ID (multi-node safe) =====================
if [[ -z "${EXP_ID:-}" ]]; then
  EXP_ID_FILE="${LOG_DIR}/CURRENT_EXP_ID"
  if [[ "${NODE_RANK}" == "0" ]]; then
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
DEBUG_LOG="${EXP_LOG_DIR}/debug_node_${NODE_RANK}.log"
: > "${DEBUG_LOG}"
exec > >(tee "${DEBUG_LOG}") 2>&1

echo "[INFO] Writing debug log to ${DEBUG_LOG}"
echo "[INFO] Experiment log dir: ${EXP_LOG_DIR}"
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

# Keep Qwen3-VL-MoE routing deterministic on torch-musa 2.7.x.
export OPENSEARCH_MUSA_STABLE_MOE_TOPK="${OPENSEARCH_MUSA_STABLE_MOE_TOPK:-1}"
export OPENSEARCH_MUSA_ROPE_BMM_WORKAROUND="${OPENSEARCH_MUSA_ROPE_BMM_WORKAROUND:-0}"

# Cross-version first-forward diagnostics. Only global rank 0 writes sampled
# module inputs/outputs; the existing precision callback records per-rank loss.
export OPENSEARCH_MODULE_TRACE="${OPENSEARCH_MODULE_TRACE:-0}"
export OPENSEARCH_MODULE_TRACE_RANKS="${OPENSEARCH_MODULE_TRACE_RANKS:-0}"
export OPENSEARCH_MODULE_TRACE_DIR="${OPENSEARCH_MODULE_TRACE_DIR:-${EXP_LOG_DIR}/module_trace}"
export OPENSEARCH_MODULE_TRACE_MAX_ROOT_CALLS="${OPENSEARCH_MODULE_TRACE_MAX_ROOT_CALLS:-1}"
export OPENSEARCH_MODULE_TRACE_SAMPLE_NUMEL="${OPENSEARCH_MODULE_TRACE_SAMPLE_NUMEL:-64}"
export OPENSEARCH_MODULE_TRACE_FULL_HASH_NUMEL="${OPENSEARCH_MODULE_TRACE_FULL_HASH_NUMEL:-131072}"
export OPENSEARCH_ATTENTION_TRACE="${OPENSEARCH_ATTENTION_TRACE:-0}"
export OPENSEARCH_ATTENTION_TRACE_DIR="${OPENSEARCH_ATTENTION_TRACE_DIR:-${EXP_LOG_DIR}/attention_trace}"
export OPENSEARCH_ATTENTION_TRACE_MIN_SEQ_LEN="${OPENSEARCH_ATTENTION_TRACE_MIN_SEQ_LEN:-1024}"
export OPENSEARCH_PRECISION_DEBUG="${OPENSEARCH_PRECISION_DEBUG:-0}"
export OPENSEARCH_PRECISION_DIR="${OPENSEARCH_PRECISION_DIR:-${EXP_LOG_DIR}/precision_debug}"

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
export PYTORCH_MUSA_ALLOC_CONF="${PYTORCH_MUSA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"
export TORCH_MCCL_AVOID_RECORD_STREAMS="${TORCH_MCCL_AVOID_RECORD_STREAMS:-1}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# ===================== Torch profiler trace =====================
export OPENSEARCH_TRACE="${OPENSEARCH_TRACE:-0}"
export OPENSEARCH_TRACE_DIR="${OPENSEARCH_TRACE_DIR:-${EXP_LOG_DIR}/trace}"
export OPENSEARCH_TRACE_RANKS="${OPENSEARCH_TRACE_RANKS:-0}"
export OPENSEARCH_TRACE_WAIT="${OPENSEARCH_TRACE_WAIT:-1}"
export OPENSEARCH_TRACE_WARMUP="${OPENSEARCH_TRACE_WARMUP:-1}"
export OPENSEARCH_TRACE_ACTIVE="${OPENSEARCH_TRACE_ACTIVE:-3}"
export OPENSEARCH_TRACE_REPEAT="${OPENSEARCH_TRACE_REPEAT:-1}"
export OPENSEARCH_TRACE_WITH_STACK="${OPENSEARCH_TRACE_WITH_STACK:-1}"
export OPENSEARCH_TRACE_RECORD_SHAPES="${OPENSEARCH_TRACE_RECORD_SHAPES:-1}"
export OPENSEARCH_TRACE_PROFILE_MEMORY="${OPENSEARCH_TRACE_PROFILE_MEMORY:-0}"
export OPENSEARCH_TRACE_WITH_FLOPS="${OPENSEARCH_TRACE_WITH_FLOPS:-0}"

export ALLOW_TORCH29_CONV3D=1
export OPENSEARCH_MUSA_ALLOW_TF32="${OPENSEARCH_MUSA_ALLOW_TF32:-0}"
export OPENSEARCH_USE_MUSA_FUSED_ADAMW="${OPENSEARCH_USE_MUSA_FUSED_ADAMW:-1}"
# ===================== Launch =====================
YAML_CONFIG="${YAML_CONFIG:-${PROJECT_ROOT}/examples/agentic_full/qwen3_vl_full_sft_30_3b_trace.yaml}"
DEBUG_YAML="${EXP_LOG_DIR}/$(basename "${YAML_CONFIG}" .yaml).node_${NODE_RANK}.yaml"
cp "${YAML_CONFIG}" "${DEBUG_YAML}"
sed -i 's#^deepspeed: .*#deepspeed: examples/deepspeed/ds_z3_config_change.json#' "${DEBUG_YAML}"

echo "[INFO] Launching multi-node training"
echo "       Master node: ${MASTER_ADDR}:${MASTER_PORT}"
echo "       Nodes total: ${NNODES}"
echo "       Node rank: ${NODE_RANK}"
echo "       GPUs per node: ${NPROC_PER_NODE}"
echo "       MCCL_SOCKET_IFNAME: ${MCCL_SOCKET_IFNAME:-<auto>}"
echo "       GLOO_SOCKET_IFNAME: ${GLOO_SOCKET_IFNAME:-<auto>}"
echo "       MCCL_IB_HCA: ${MCCL_IB_HCA:-<auto>}"
echo "       OPENSEARCH_MUSA_ALLOW_TF32: ${OPENSEARCH_MUSA_ALLOW_TF32}"
echo "       OPENSEARCH_TRACE: ${OPENSEARCH_TRACE}"
echo "       OPENSEARCH_TRACE_DIR: ${OPENSEARCH_TRACE_DIR:-<disabled>}"
echo "       OPENSEARCH_TRACE_RANKS: ${OPENSEARCH_TRACE_RANKS:-<disabled>}"
echo "       OPENSEARCH_TRACE_WITH_STACK: ${OPENSEARCH_TRACE_WITH_STACK:-<disabled>}"
echo "       OPENSEARCH_TRACE_RECORD_SHAPES: ${OPENSEARCH_TRACE_RECORD_SHAPES:-<disabled>}"
echo "       OPENSEARCH_USE_MUSA_FUSED_ADAMW: ${OPENSEARCH_USE_MUSA_FUSED_ADAMW}"
echo "       Rendezvous mode: static torchrun (--node_rank ${NODE_RANK})"
echo "[INFO] Using yaml: ${DEBUG_YAML}"
echo "[DEBUG] MUSA_LAUNCH_BLOCKING=${MUSA_LAUNCH_BLOCKING:-<unset>}"

# Force the static torchrun branch in llamafactory/launcher.py. This branch
# passes --node_rank explicitly and avoids elastic rendezvous membership state.
unset RDZV_ID

FORCE_TORCHRUN=1 \
NNODES="${NNODES}" \
NODE_RANK="${NODE_RANK}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
python -m llamafactory.cli train "${DEBUG_YAML}"
