#!/usr/bin/env bash
# Same as train_30b_trace.sh but defaults to MATE full grouped-GEMM YAML.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export YAML_CONFIG="${YAML_CONFIG:-${SCRIPT_DIR}/examples/agentic_full/qwen3_vl_full_sft_30_3b_mate_groupgemm.yaml}"
echo "[INFO] MATE train wrapper YAML_CONFIG=${YAML_CONFIG}"
exec bash "${SCRIPT_DIR}/train_30b_trace.sh" "$@"
