#!/usr/bin/env bash
set -euo pipefail

HOSTFILE="${HOSTFILE:-./hostfile}"
TARGET_WORKDIR="${TARGET_WORKDIR:-$(pwd)}"
GRACE_SECONDS="${GRACE_SECONDS:-8}"
ALL_PYTHON=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: bash stop_all.sh [hostfile] [--all-python] [--dry-run]

Stops distributed training on all hosts from hostfile.

Options:
  hostfile      Host list file. Default: ./hostfile or $HOSTFILE.
  --all-python  Kill all python processes, matching the referenced VeOmni script.
  --dry-run     Print matched processes without killing them.

Environment:
  TARGET_WORKDIR   Workdir used to narrow default process matching. Default: pwd.
  GRACE_SECONDS    Seconds between TERM and KILL. Default: 8.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --all-python)
      ALL_PYTHON=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      HOSTFILE="$1"
      shift
      ;;
  esac
done

if [[ ! -f "${HOSTFILE}" ]]; then
  echo "[ERROR] hostfile not found: ${HOSTFILE}"
  exit 1
fi

mapfile -t HOSTS < <(
  awk 'NF > 0 && $1 !~ /^#/ { print $1 }' "${HOSTFILE}" | awk '!seen[$0]++'
)

if [[ ${#HOSTS[@]} -eq 0 ]]; then
  echo "[ERROR] no valid hosts found in ${HOSTFILE}"
  exit 1
fi

MODE="training"
if [[ "${ALL_PYTHON}" == "1" ]]; then
  MODE="all-python"
fi

echo "[INFO] hosts to stop: ${HOSTS[*]}"
echo "[INFO] target workdir: ${TARGET_WORKDIR}"
if [[ "${MODE}" == "all-python" ]]; then
  echo "[WARN] --all-python enabled: this will kill every python process on each host."
fi

remote_stop_cmd() {
  local mode="$1"
  local dry_run="$2"
  local workdir="$3"
  local grace="$4"

  cat <<REMOTE
set -u
MODE='${mode}'
DRY_RUN='${dry_run}'
TARGET_WORKDIR='${workdir}'
GRACE_SECONDS='${grace}'

collect_pids() {
  ps -eo pid=,ppid=,args= | awk -v mode="\$MODE" -v workdir="\$TARGET_WORKDIR" -v self="\$\$" '
    {
      pid=\$1; ppid=\$2;
      args=\$0;
      sub(/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+/, "", args);
      if (pid == self || ppid == self) next;
      if (index(args, "collect_pids()") || index(args, "show_matches()") || index(args, "kill_pids()")) next;
      if (mode == "all-python") {
        if (args ~ /(^|[[:space:]])python([0-9.]*)?([[:space:]]|$)/ || args ~ /\/python([0-9.]*)?([[:space:]]|$)/) print pid;
        next;
      }
      if (index(args, workdir "/src/llamafactory/launcher.py") > 0) print pid;
      else if (index(args, "python -m llamafactory.cli train " workdir "/") > 0) print pid;
      else if (index(args, "torchrun") > 0 && index(args, workdir "/src/llamafactory/launcher.py") > 0) print pid;
      else if (args ~ /^bash train_8b_test_no_blocking\.sh/) print pid;
      else if (args ~ /^bash .*launch_train_.*\.sh/) print pid;
    }
  ' | awk '!seen[\$0]++'
}

show_matches() {
  local pids
  pids="\$(collect_pids | xargs echo)"
  if [[ -z "\$pids" ]]; then
    echo '[INFO] no matched processes.'
  else
    ps -o pid=,ppid=,stat=,args= -p \$pids || true
  fi
}

kill_pids() {
  local signal="\$1"
  local pids
  pids="\$(collect_pids | xargs echo)"
  if [[ -n "\$pids" ]]; then
    kill -"\$signal" \$pids 2>/dev/null || true
  fi
}

echo '[INFO] matched before stop:'
show_matches

if [[ "\$DRY_RUN" == "1" ]]; then
  echo '[INFO] dry-run only; no process killed.'
  exit 0
fi

echo '[INFO] sending TERM ...'
kill_pids TERM
sleep "\$GRACE_SECONDS"

if collect_pids | grep -q .; then
  echo '[WARN] residual processes after TERM:'
  show_matches
  echo '[WARN] sending KILL ...'
  kill_pids KILL
  sleep 2
fi

echo '[INFO] matched after stop:'
show_matches
REMOTE
}

for host in "${HOSTS[@]}"; do
  echo "========== ${host} =========="
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${host}" "$(remote_stop_cmd "${MODE}" "${DRY_RUN}" "${TARGET_WORKDIR}" "${GRACE_SECONDS}")" \
    || echo "[WARN] failed to stop processes on ${host}"
done

echo "[INFO] stop command finished."
