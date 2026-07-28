#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: MUSA_VMM_REPRO_ALLOW=1 bash $0 <hostfile> [probe arguments...]"
  exit 2
fi

if [[ "${MUSA_VMM_REPRO_ALLOW:-0}" != "1" ]]; then
  echo "Refusing to run a stress reproduction without MUSA_VMM_REPRO_ALLOW=1."
  echo "Use only on dedicated idle GPUs; never run beside an active training job."
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKER="${MUSA_VMM_REPRO_WORKER:-${SCRIPT_DIR}/repro_musa_vmm_mccl_hang.py}"
HOSTFILE="$1"
shift

if [[ ! -f "${HOSTFILE}" ]]; then
  echo "hostfile not found: ${HOSTFILE}"
  exit 2
fi
if [[ ! -f "${WORKER}" ]]; then
  echo "probe worker not found: ${WORKER}"
  exit 2
fi

HOSTFILE="$(cd "$(dirname "${HOSTFILE}")" && pwd)/$(basename "${HOSTFILE}")"
mapfile -t HOSTS < <(sed 's/#.*//' "${HOSTFILE}" | awk 'NF {print $1}')
if [[ "${#HOSTS[@]}" -eq 0 ]]; then
  echo "no valid hosts found in: ${HOSTFILE}"
  exit 2
fi
if [[ "$(printf '%s\n' "${HOSTS[@]}" | sort -u | wc -l)" -ne "${#HOSTS[@]}" ]]; then
  echo "duplicate hosts found in: ${HOSTFILE}"
  exit 2
fi

GPUS_PER_NODE="${MUSA_VMM_REPRO_GPUS_PER_NODE:-8}"
if ! [[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MUSA_VMM_REPRO_GPUS_PER_NODE must be positive: ${GPUS_PER_NODE}"
  exit 2
fi

if [[ -n "${MUSA_VMM_REPRO_MPIRUN:-}" ]]; then
  MPIRUN="${MUSA_VMM_REPRO_MPIRUN}"
elif [[ -x /usr/local/openmpi/bin/mpirun ]]; then
  MPIRUN=/usr/local/openmpi/bin/mpirun
else
  MPIRUN="$(command -v mpirun || true)"
fi
if [[ -z "${MPIRUN}" || ! -x "${MPIRUN}" ]]; then
  echo "OpenMPI mpirun was not found."
  exit 2
fi

pick_first_path() {
  local root="$1"
  shift
  local name
  for name in "$@"; do
    if [[ -e "${root}/${name}" ]]; then
      echo "${name}"
      return 0
    fi
  done
  return 1
}

if [[ -z "${MCCL_SOCKET_IFNAME:-}" ]]; then
  MCCL_SOCKET_IFNAME="$(pick_first_path /sys/class/net bond1 bond0 ib0 ib1 eth0 eno1 ens1 enp1s0 || true)"
fi
if [[ -z "${MCCL_IB_HCA:-}" ]]; then
  MCCL_IB_HCA="$(pick_first_path /sys/class/infiniband mubd0 mubd1 mlx5_bond_0 mlx5_bond_1 || true)"
fi

export MASTER_ADDR="${MASTER_ADDR:-${HOSTS[0]}}"
export MASTER_PORT="${MUSA_VMM_REPRO_MASTER_PORT:-34239}"
export PYTORCH_MUSA_ALLOC_CONF="${PYTORCH_MUSA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"
export MCCL_PROTOS="${MCCL_PROTOS:-2}"
export MCCL_ALGOS="${MCCL_ALGOS:-1}"
export MCCL_BUFFSIZE="${MCCL_BUFFSIZE:-20971520}"
export MCCL_CHECK_POINTERS="${MCCL_CHECK_POINTERS:-0}"
export MCCL_IB_GID_INDEX="${MCCL_IB_GID_INDEX:-3}"
export MCCL_IB_TIMEOUT="${MCCL_IB_TIMEOUT:-22}"
export MCCL_DEBUG="${MCCL_DEBUG:-INFO}"
export MCCL_DEBUG_SUBSYS="${MCCL_DEBUG_SUBSYS:-INIT,NET,COLL}"
if [[ -n "${MCCL_SOCKET_IFNAME}" ]]; then
  export MCCL_SOCKET_IFNAME
fi
if [[ -n "${MCCL_IB_HCA}" ]]; then
  export MCCL_IB_HCA
fi

RUN_ID="${MUSA_VMM_REPRO_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${MUSA_VMM_REPRO_LOG_DIR:-${PROJECT_ROOT}/logs/musa_vmm_mccl_repro_${RUN_ID}}"
mkdir -p "${LOG_DIR}"
export MCCL_DEBUG_FILE="${MCCL_DEBUG_FILE:-${LOG_DIR}/mccl_%h_%p.log}"

TEMP_DIR="$(mktemp -d)"
PROBE_HOSTFILE="${TEMP_DIR}/hostfile.probe"
cleanup() {
  rm -f -- "${PROBE_HOSTFILE}"
  rmdir -- "${TEMP_DIR}" 2>/dev/null || true
}
trap cleanup EXIT
for host in "${HOSTS[@]}"; do
  printf '%s slots=%s\n' "${host}" "${GPUS_PER_NODE}" >> "${PROBE_HOSTFILE}"
done

TOTAL_RANKS="$(( ${#HOSTS[@]} * GPUS_PER_NODE ))"
TOTAL_TIMEOUT_SECONDS="${MUSA_VMM_REPRO_TIMEOUT_SECONDS:-600}"
if ! [[ "${TOTAL_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] || (( TOTAL_TIMEOUT_SECONDS < 20 )); then
  echo "MUSA_VMM_REPRO_TIMEOUT_SECONDS must be an integer of at least 20 seconds."
  exit 2
fi
MPI_TIMEOUT_SECONDS="${MUSA_VMM_REPRO_MPI_TIMEOUT_SECONDS:-$((TOTAL_TIMEOUT_SECONDS - 10))}"
if ! [[ "${MPI_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] || (( MPI_TIMEOUT_SECONDS >= TOTAL_TIMEOUT_SECONDS )); then
  echo "MUSA_VMM_REPRO_MPI_TIMEOUT_SECONDS must be positive and below the outer timeout."
  exit 2
fi
if ! [[ "${MASTER_PORT}" =~ ^[1-9][0-9]*$ ]] || (( MASTER_PORT > 65535 )); then
  echo "MUSA_VMM_REPRO_MASTER_PORT must be in 1..65535."
  exit 2
fi
PYTHON_BIN="${MUSA_VMM_REPRO_PYTHON:-python3}"

MPI_ARGS=(
  --allow-run-as-root
  --hostfile "${PROBE_HOSTFILE}"
  --map-by "ppr:${GPUS_PER_NODE}:node"
  -np "${TOTAL_RANKS}"
  --bind-to none
  --timeout "${MPI_TIMEOUT_SECONDS}"
  --prtemca prte_keep_fqdn_hostnames 1
  --wdir "${PROJECT_ROOT}"
)
if [[ -n "${MCCL_SOCKET_IFNAME}" ]]; then
  MPI_ARGS+=(--mca btl_tcp_if_include "${MCCL_SOCKET_IFNAME}")
fi

for key in PATH LD_LIBRARY_PATH MASTER_ADDR MASTER_PORT PYTORCH_MUSA_ALLOC_CONF \
  MCCL_PROTOS MCCL_ALGOS MCCL_BUFFSIZE MCCL_CHECK_POINTERS MCCL_IB_GID_INDEX \
  MCCL_IB_TIMEOUT MCCL_SOCKET_IFNAME MCCL_IB_HCA MCCL_DEBUG MCCL_DEBUG_SUBSYS MCCL_DEBUG_FILE; do
  if [[ -n "${!key:-}" ]]; then
    MPI_ARGS+=(-x "${key}=${!key}")
  fi
done

echo "MUSA VMM/MCCL reproduction"
echo "  nodes=${#HOSTS[@]} ranks=${TOTAL_RANKS} gpus_per_node=${GPUS_PER_NODE}"
echo "  allocator=${PYTORCH_MUSA_ALLOC_CONF}"
echo "  log_dir=${LOG_DIR}"
echo "  timeout=${TOTAL_TIMEOUT_SECONDS}s"

set +e
timeout --signal=TERM --kill-after=10s "${TOTAL_TIMEOUT_SECONDS}s" \
  "${MPIRUN}" "${MPI_ARGS[@]}" \
  "${PYTHON_BIN}" "${WORKER}" --log-dir "${LOG_DIR}" "$@" \
  2>&1 | tee "${LOG_DIR}/launcher.log"
STATUS=$?
set -e

if [[ "${STATUS}" -eq 0 ]]; then
  echo "Probe passed. Timelines: ${LOG_DIR}/rank_*.jsonl"
else
  echo "Probe failed with status ${STATUS}. Inspect the last event in every rank timeline."
  if [[ "${STATUS}" -eq 124 ]]; then
    echo "The external timeout fired; verify that no probe MPI/Python process remains before another run."
  fi
fi
exit "${STATUS}"
