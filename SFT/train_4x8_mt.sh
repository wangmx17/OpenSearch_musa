#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
LOG_DIR="${PROJECT_ROOT}/logs"

# ===================== [MOD 1] hostfile 参数支持 =====================
# 原来是写死 HOSTFILE=xxx
HOSTFILE="${1:-${HOSTFILE:-${PROJECT_ROOT}/hostfile}}"

mkdir -p "${LOG_DIR}"

pick_first_host() {
  local file="$1"
  [[ -f "${file}" ]] || return 1
  awk 'NF {print $1; exit}' "${file}"
}

# ===================== [MOD 2] auto launch 控制 =====================
AUTO_LAUNCH="${AUTO_LAUNCH:-0}"

# ===================== node utils =====================
resolve_node_rank() {
  local file="$1"
  local host="${CURRENT_HOSTNAME:-}"
  local rank=""

  if [[ -n "${NODE_RANK:-}" ]]; then
    echo "${NODE_RANK}"
    return 0
  fi

  for rank in "${RANK:-}" "${SLURM_NODEID:-}" "${OMPI_COMM_WORLD_RANK:-}" "${PADDLE_TRAINER_ID:-}"; do
    if [[ "${rank}" =~ ^[0-9]+$ ]]; then
      echo "${rank}"
      return 0
    fi
  done

  if [[ -z "${host}" ]]; then
    host="$(cat /etc/hostname 2>/dev/null || true)"
  fi
  host="${host%%.*}"

  if [[ -f "${file}" ]]; then
    rank="$(awk -v host="${host}" '
      NF {
        if ($1 == host) { print NR-1; exit }
      }
    ' "${file}")"
    if [[ "${rank}" =~ ^[0-9]+$ ]]; then
      echo "${rank}"
      return 0
    fi
  fi

  if [[ "${NNODES:-1}" == "1" ]]; then
    echo "0"
    return 0
  fi

  echo "[ERROR] Cannot resolve NODE_RANK for host '${host}'. Set NODE_RANK explicitly or fix HOSTFILE=${file}." >&2
  return 1
}

resolve_nnodes() {
  local file="$1"
  if [[ -n "${NNODES:-}" ]]; then echo "${NNODES}"; return 0; fi
  if [[ -f "${file}" ]]; then awk 'NF{c++}END{print c+0}' "${file}"; return 0; fi
  echo "1"
}

# ===================== [MOD 3] 一键 SSH launcher =====================
launch_cluster() {
  local file="$1"
  local master_addr="$2"
  local master_port="$3"

  local i=0
  while read -r host; do
    [[ -z "$host" ]] && continue

    echo "[AUTO] launching rank=$i on $host"

    ssh -o StrictHostKeyChecking=no "$host" \
      "cd ${PROJECT_ROOT} && \
       HOSTFILE=${file} \
       MASTER_ADDR=${master_addr} \
       MASTER_PORT=${master_port} \
       NNODES=$(resolve_nnodes "$file") \
       NODE_RANK=${i} \
       AUTO_LAUNCH=0 \
       SKIP_INSTALL=1 \
       bash $0" &

    i=$((i+1))
  done < "$file"

  wait
}

# ===================== install =====================
cd "${PROJECT_ROOT}"

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  pip install -e ".[metrics,deepspeed]" -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

# ===================== env =====================
CURRENT_HOSTNAME="$(cat /etc/hostname 2>/dev/null || true)"

MASTER_ADDR="${MASTER_ADDR:-$(pick_first_host "${HOSTFILE}")}"
MASTER_PORT="${MASTER_PORT:-34237}"
NNODES="$(resolve_nnodes "${HOSTFILE}")"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

echo "[INFO] MASTER_ADDR=$MASTER_ADDR"
echo "[INFO] NNODES=$NNODES"

# ===================== [MOD 4] 一键入口 =====================
if [[ "${AUTO_LAUNCH}" == "1" ]]; then
  echo "[INFO] AUTO_LAUNCH enabled, starting cluster..."
  launch_cluster "${HOSTFILE}" "${MASTER_ADDR}" "${MASTER_PORT}"
  exit 0
fi

NODE_RANK="$(resolve_node_rank "${HOSTFILE}")"
echo "[INFO] NODE_RANK=$NODE_RANK"

# ===================== distributed env =====================
export NNODES NODE_RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE

# ===================== launch =====================
YAML_CONFIG="${YAML_CONFIG:-${PROJECT_ROOT}/config.yaml}"

FORCE_TORCHRUN=1 \
NNODES="${NNODES}" \
NODE_RANK="${NODE_RANK}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
python -m llamafactory.cli train "${YAML_CONFIG}"

