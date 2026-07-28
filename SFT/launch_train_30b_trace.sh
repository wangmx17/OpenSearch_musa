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

# Allocate the experiment directory once on the launcher and propagate the
# same EXP_ID to every node. Letting each node read CURRENT_EXP_ID races on
# the shared filesystem and can split one distributed job across exp_N dirs.
EXP_ROOT="${WORKDIR}/logs"
mkdir -p "${EXP_ROOT}"
if [[ -z "${EXP_ID:-}" ]]; then
  EXP_ALLOC_LOCK="${EXP_ROOT}/.exp_launcher_alloc.lock"
  until mkdir "${EXP_ALLOC_LOCK}" 2>/dev/null; do
    sleep 0.2
  done
  trap 'rmdir "${EXP_ALLOC_LOCK}" 2>/dev/null || true' EXIT

  LAST_EXP_ID="$(find "${EXP_ROOT}" -maxdepth 1 -type d -name 'exp_[0-9]*' -printf '%f\n' 2>/dev/null \
    | sed -n 's/^exp_\([0-9][0-9]*\)$/\1/p' | sort -n | tail -1)"
  LAST_EXP_ID="${LAST_EXP_ID:-0}"
  EXP_ID="$((LAST_EXP_ID + 1))"
  mkdir -p "${EXP_ROOT}/exp_${EXP_ID}"
  printf '%s\n' "${EXP_ID}" > "${EXP_ROOT}/exp_${EXP_ID}/EXP_ID"

  rmdir "${EXP_ALLOC_LOCK}"
  trap - EXIT
fi

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}"
echo "WORKDIR=${WORKDIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "EXP_ID=${EXP_ID}"

for rank in "${!HOSTS[@]}"; do
  host="${HOSTS[$rank]}"
  log_file="${LOG_DIR}/rank${rank}_${host}.log"

  echo "Launching rank ${rank} on ${host}, log: ${log_file}"

  ssh -n -f "$host" \
      "cd '$WORKDIR' && mkdir -p '$LOG_DIR' && setsid env HOSTFILE='$HOSTFILE' MASTER_ADDR='$MASTER_ADDR' MASTER_PORT='$MASTER_PORT' NNODES='$NNODES' NODE_RANK='$rank' SKIP_INSTALL='$SKIP_INSTALL' EXP_ID='$EXP_ID' bash '$TRAIN_SCRIPT' > '$log_file' 2>&1 < /dev/null &"
	# "cd '$WORKDIR' && nohup setsid env MASTER_ADDR='$MASTER_ADDR' MASTER_PORT='$MASTER_PORT' NNODES='$NNODES' NODE_RANK='$rank' SKIP_INSTALL='$SKIP_INSTALL' bash '$TRAIN_SCRIPT' > '$log_file' 2>&1 < /dev/null &"
done

echo "All ranks launched."
echo "Check logs on each node under: ${WORKDIR}/${LOG_DIR}"

 
 
