#!/usr/bin/env bash
# 4-node Qwen3-VL-30B SFT with MATE full grouped GEMM (fwd+dX+dW)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_ROOT="${SFT_ROOT:-$SCRIPT_DIR}"
cd "$SFT_ROOT"
export WORKDIR="${WORKDIR:-$SFT_ROOT}"
export TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_30b_mate_groupgemm.sh}"
export SKIP_INSTALL="${SKIP_INSTALL:-1}"
HOSTFILE="${1:-${SFT_ROOT}/hostfile.txt}"
echo "[INFO] WORKDIR=$WORKDIR"
echo "[INFO] TRAIN_SCRIPT=$TRAIN_SCRIPT"
echo "[INFO] HOSTFILE=$HOSTFILE"
bash launch_train_30b_trace.sh "$HOSTFILE"
