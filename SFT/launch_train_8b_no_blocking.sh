#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 <hostfile>"
  exit 1
fi

HOSTFILE="$1"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_8b_test_no_blocking.sh}"
MASTER_PORT="${MASTER_PORT:-34237}"
SKIP_INSTALL="${SKIP_INSTALL:-1}"
WORKDIR="${WORKDIR:-$(pwd)}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10}"
#LOG_TIMESTAMP="${LOG_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_TIMESTAMP="${LOG_TIMESTAMP:-multinode_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/${LOG_TIMESTAMP}}"
#LOG_DIR="${LOG_DIR:-logs/multinode_$(date +%Y%m%d_%H%M%S)}"

host_addr() {
  local host="$1"
  host="${host#*@}"
  host="${host%%:*}"
  echo "${host}"
}

is_local_host() {
  local host
  host="$(host_addr "$1")"

  case "${host}" in
    localhost|127.0.0.1|::1)
      return 0
      ;;
  esac

  if [[ "${host}" == "$(hostname 2>/dev/null || true)" ]]; then
    return 0
  fi

  if [[ "${host}" == "$(hostname -f 2>/dev/null || true)" ]]; then
    return 0
  fi

  if hostname -I 2>/dev/null | tr ' ' '\n' | grep -Fxq "${host}"; then
    return 0
  fi

  return 1
}

if [[ ! -f "$HOSTFILE" ]]; then
  echo "hostfile not found: $HOSTFILE"
  exit 1
fi

mapfile -t HOSTS < <(sed 's/#.*//' "$HOSTFILE" | awk 'NF {print $1}')

if [[ "${#HOSTS[@]}" -eq 0 ]]; then
  echo "no valid hosts found in: $HOSTFILE"
  exit 1
fi

MASTER_ADDR="${MASTER_ADDR:-$(host_addr "${HOSTS[0]}")}"
NNODES="${NNODES:-${#HOSTS[@]}}"

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}"
echo "WORKDIR=${WORKDIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "SSH_OPTS=${SSH_OPTS}"

mkdir -p "${LOG_DIR}"

for rank in "${!HOSTS[@]}"; do
  host="${HOSTS[$rank]}"

  if is_local_host "${host}"; then
    if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
      echo "[ERROR] local train script not found: ${WORKDIR}/${TRAIN_SCRIPT}" >&2
      exit 1
    fi
    continue
  fi

  echo "Checking ssh access for rank ${rank} on ${host}"
  ssh ${SSH_OPTS} -n "${host}" \
      "cd '${WORKDIR}' && test -f '${TRAIN_SCRIPT}' && mkdir -p '${LOG_DIR}'"
done

for rank in "${!HOSTS[@]}"; do
  host="${HOSTS[$rank]}"
  safe_host="${host//[^A-Za-z0-9_.-]/_}"
  log_file="${LOG_DIR}/rank${rank}_${safe_host}.log"

  echo "Launching rank ${rank} on ${host}, log: ${log_file}"

  if is_local_host "${host}"; then
    (
      cd "${WORKDIR}"
      mkdir -p "${LOG_DIR}"
      setsid env MASTER_ADDR="${MASTER_ADDR}" MASTER_PORT="${MASTER_PORT}" NNODES="${NNODES}" NODE_RANK="${rank}" SKIP_INSTALL="${SKIP_INSTALL}" LOG_TIMESTAMP="${LOG_TIMESTAMP}" bash "${TRAIN_SCRIPT}" > "${log_file}" 2>&1 < /dev/null &
    )
  else
    ssh ${SSH_OPTS} -n -f "${host}" \
        "cd '${WORKDIR}' && mkdir -p '${LOG_DIR}' && setsid env MASTER_ADDR='${MASTER_ADDR}' MASTER_PORT='${MASTER_PORT}' NNODES='${NNODES}' NODE_RANK='${rank}' SKIP_INSTALL='${SKIP_INSTALL}' LOG_TIMESTAMP='${LOG_TIMESTAMP}' bash '${TRAIN_SCRIPT}' > '${log_file}' 2>&1 < /dev/null &"
  fi
done

echo "All ranks launched."
echo "Check logs on each node under: ${WORKDIR}/${LOG_DIR}"

 
 
