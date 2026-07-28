#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
OPENSEARCH_MCCL_PREFLIGHT="${OPENSEARCH_MCCL_PREFLIGHT:-1}"
MCCL_PREFLIGHT_SCRIPT="${MCCL_PREFLIGHT_SCRIPT:-${SCRIPT_DIR}/scripts/musa_mccl_health_check.py}"
MCCL_PREFLIGHT_LOG_DIR="${MCCL_PREFLIGHT_LOG_DIR:-${LOG_DIR}/mccl_preflight}"

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

# Run one MPI process per node and let mccl-tests cover every local GPU. This
# finishes before any training rank is dispatched, so a failed/slow topology
# cannot strand half of the torchrun workers in ZeRO initialization.
case "${OPENSEARCH_MCCL_PREFLIGHT,,}" in
  1|true|yes|on)
    if [[ ! -f "${MCCL_PREFLIGHT_SCRIPT}" ]]; then
      echo "MCCL preflight script not found: ${MCCL_PREFLIGHT_SCRIPT}"
      exit 1
    fi

    MCCL_PREFLIGHT_ARGS=(
      --hostfile "${HOSTFILE}"
      --workdir "${WORKDIR}"
      --log-dir "${MCCL_PREFLIGHT_LOG_DIR}"
      --gpus-per-node "${NPROC_PER_NODE}"
      --message-bytes "${MCCL_PREFLIGHT_MESSAGE_BYTES:-1M}"
      --warmup-iterations "${MCCL_PREFLIGHT_WARMUP_ITERATIONS:-1}"
      --iterations "${MCCL_PREFLIGHT_ITERATIONS:-2}"
      --stream-timeout-seconds "${MCCL_PREFLIGHT_STREAM_TIMEOUT_SECONDS:-60}"
      --mpi-timeout-seconds "${MCCL_PREFLIGHT_MPI_TIMEOUT_SECONDS:-90}"
      --timeout-seconds "${MCCL_PREFLIGHT_TIMEOUT_SECONDS:-120}"
      --min-busbw-gbps "${MCCL_PREFLIGHT_MIN_BUSBW_GBPS:-0}"
    )
    if [[ -n "${MCCL_PREFLIGHT_TEST_BIN:-}" ]]; then
      MCCL_PREFLIGHT_ARGS+=(--test-bin "${MCCL_PREFLIGHT_TEST_BIN}")
    fi
    if [[ -n "${MCCL_PREFLIGHT_MPIRUN:-}" ]]; then
      MCCL_PREFLIGHT_ARGS+=(--mpirun "${MCCL_PREFLIGHT_MPIRUN}")
    fi

    echo "Running MCCL preflight before training dispatch..."
    python3 "${MCCL_PREFLIGHT_SCRIPT}" "${MCCL_PREFLIGHT_ARGS[@]}"
    ;;
  0|false|no|off)
    echo "WARNING: MCCL preflight is disabled (OPENSEARCH_MCCL_PREFLIGHT=${OPENSEARCH_MCCL_PREFLIGHT})."
    ;;
  *)
    echo "invalid OPENSEARCH_MCCL_PREFLIGHT value: ${OPENSEARCH_MCCL_PREFLIGHT}"
    exit 1
    ;;
esac

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
echo "MCCL_PREFLIGHT=${OPENSEARCH_MCCL_PREFLIGHT}"

for rank in "${!HOSTS[@]}"; do
  host="${HOSTS[$rank]}"
  log_file="${LOG_DIR}/rank${rank}_${host}.log"

  echo "Launching rank ${rank} on ${host}, log: ${log_file}"

  ssh -n -f "$host" \
      "cd '$WORKDIR' && mkdir -p '$LOG_DIR' && setsid env HOSTFILE='$HOSTFILE' MASTER_ADDR='$MASTER_ADDR' MASTER_PORT='$MASTER_PORT' NNODES='$NNODES' NODE_RANK='$rank' NPROC_PER_NODE='$NPROC_PER_NODE' SKIP_INSTALL='$SKIP_INSTALL' EXP_ID='$EXP_ID' MCCL_PROTOS='${MCCL_PROTOS:-2}' MCCL_ALGOS='${MCCL_ALGOS:-1}' MCCL_BUFFSIZE='${MCCL_BUFFSIZE:-20971520}' MCCL_MAX_NCHANNELS='${MCCL_MAX_NCHANNELS:-14}' MCCL_CHECK_POINTERS='${MCCL_CHECK_POINTERS:-0}' MCCL_IB_GID_INDEX='${MCCL_IB_GID_INDEX:-3}' MCCL_IB_TC='${MCCL_IB_TC:-41}' MCCL_IB_TIMEOUT='${MCCL_IB_TIMEOUT:-22}' MCCL_SOCKET_IFNAME='${MCCL_SOCKET_IFNAME:-}' MCCL_IB_HCA='${MCCL_IB_HCA:-}' MCCL_DEBUG='${MCCL_DEBUG:-WARN}' PYTORCH_MUSA_ALLOC_CONF='${PYTORCH_MUSA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}' TORCH_MCCL_AVOID_RECORD_STREAMS='${TORCH_MCCL_AVOID_RECORD_STREAMS:-1}' OPENSEARCH_MUSA_STABLE_MOE_TOPK='${OPENSEARCH_MUSA_STABLE_MOE_TOPK:-1}' OPENSEARCH_MUSA_ROPE_BMM_WORKAROUND='${OPENSEARCH_MUSA_ROPE_BMM_WORKAROUND:-1}' OPENSEARCH_DIAGNOSTIC_DATASET='${OPENSEARCH_DIAGNOSTIC_DATASET:-}' OPENSEARCH_DIAGNOSTIC_DISABLE_SHUFFLING='${OPENSEARCH_DIAGNOSTIC_DISABLE_SHUFFLING:-0}' OPENSEARCH_DIAGNOSTIC_EXPERTS_IMPLEMENTATION='${OPENSEARCH_DIAGNOSTIC_EXPERTS_IMPLEMENTATION:-}' OPENSEARCH_DIAGNOSTIC_MAX_STEPS='${OPENSEARCH_DIAGNOSTIC_MAX_STEPS:-}' OPENSEARCH_MODULE_TRACE='${OPENSEARCH_MODULE_TRACE:-0}' OPENSEARCH_MODULE_TRACE_TARGETS='${OPENSEARCH_MODULE_TRACE_TARGETS:-}' OPENSEARCH_MODULE_TRACE_FULL_HASH_NUMEL='${OPENSEARCH_MODULE_TRACE_FULL_HASH_NUMEL:-131072}' OPENSEARCH_ATTENTION_TRACE='${OPENSEARCH_ATTENTION_TRACE:-0}' OPENSEARCH_ATTENTION_TRACE_MIN_SEQ_LEN='${OPENSEARCH_ATTENTION_TRACE_MIN_SEQ_LEN:-1024}' OPENSEARCH_PRECISION_DEBUG='${OPENSEARCH_PRECISION_DEBUG:-0}' bash '$TRAIN_SCRIPT' > '$log_file' 2>&1 < /dev/null &"
	# "cd '$WORKDIR' && nohup setsid env MASTER_ADDR='$MASTER_ADDR' MASTER_PORT='$MASTER_PORT' NNODES='$NNODES' NODE_RANK='$rank' SKIP_INSTALL='$SKIP_INSTALL' bash '$TRAIN_SCRIPT' > '$log_file' 2>&1 < /dev/null &"
done

echo "All ranks launched."
echo "Check logs on each node under: ${WORKDIR}/${LOG_DIR}"

 
 
