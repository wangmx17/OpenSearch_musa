#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 <hostfile>"
  exit 1
fi

HOSTFILE="$1"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_30b_trace.sh}"
MASTER_PORT="${MASTER_PORT:-34237}"
SKIP_INSTALL="${SKIP_INSTALL:-1}"
WORKDIR="${WORKDIR:-$(pwd)}"
LOG_DIR="${LOG_DIR:-logs/multinode_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "$HOSTFILE" ]]; then
  echo "hostfile not found: $HOSTFILE"
  exit 1
fi

HOSTFILE="$(cd "$(dirname "$HOSTFILE")" && pwd)/$(basename "$HOSTFILE")"

mapfile -t HOSTS < <(sed 's/#.*//' "$HOSTFILE" | awk 'NF {print $1}')

if [[ "${#HOSTS[@]}" -eq 0 ]]; then
  echo "no valid hosts found in: $HOSTFILE"
  exit 1
fi

MASTER_ADDR="${MASTER_ADDR:-${HOSTS[0]}}"
NNODES="${#HOSTS[@]}"

UNIQUE_HOSTS="$(printf '%s\n' "${HOSTS[@]}" | sort -u | wc -l)"
if [[ "${UNIQUE_HOSTS}" -ne "${NNODES}" ]]; then
  echo "duplicate hosts found in: ${HOSTFILE}"
  exit 1
fi

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}"
echo "WORKDIR=${WORKDIR}"
echo "LOG_DIR=${LOG_DIR}"

for rank in "${!HOSTS[@]}"; do
  host="${HOSTS[$rank]}"
  log_file="${LOG_DIR}/rank${rank}_${host}.log"

  echo "Launching rank ${rank} on ${host}, log: ${log_file}"

  ssh -n -f "$host" \
      "cd '$WORKDIR' && mkdir -p '$LOG_DIR' && setsid env HOSTFILE='$HOSTFILE' MASTER_ADDR='$MASTER_ADDR' MASTER_PORT='$MASTER_PORT' NNODES='$NNODES' NODE_RANK='$rank' SKIP_INSTALL='$SKIP_INSTALL' bash '$TRAIN_SCRIPT' > '$log_file' 2>&1 < /dev/null &"
	# "cd '$WORKDIR' && nohup setsid env MASTER_ADDR='$MASTER_ADDR' MASTER_PORT='$MASTER_PORT' NNODES='$NNODES' NODE_RANK='$rank' SKIP_INSTALL='$SKIP_INSTALL' bash '$TRAIN_SCRIPT' > '$log_file' 2>&1 < /dev/null &"
done

echo "All ranks launched."
echo "Check logs on each node under: ${WORKDIR}/${LOG_DIR}"

 
 
