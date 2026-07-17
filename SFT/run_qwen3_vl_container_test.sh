#!/usr/bin/env bash
# Run Qwen3-VL smoke test inside the DeepSpeed MUSA container and record full logs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${SCRIPT_DIR}/logs/qwen3_vl_smoke_${TIMESTAMP}"
MASTER_LOG="${LOG_ROOT}/master.log"
TRAIN_LOG="${LOG_ROOT}/training.log"
RESULT_FILE="${LOG_ROOT}/result.txt"

mkdir -p "${LOG_ROOT}"

exec > >(tee -a "${MASTER_LOG}") 2>&1

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

on_exit() {
  local exit_code=$?
  if [[ ${exit_code} -eq 0 ]]; then
    echo "STATUS=SUCCESS" > "${RESULT_FILE}"
    log "Test finished successfully."
  else
    echo "STATUS=FAILED" > "${RESULT_FILE}"
    echo "EXIT_CODE=${exit_code}" >> "${RESULT_FILE}"
    log "Test failed with exit code ${exit_code}."
  fi
  log "Logs saved under ${LOG_ROOT}"
}
trap on_exit EXIT

log "========== Qwen3-VL Smoke Test =========="
log "Host: $(hostname)"
log "Workdir: ${SCRIPT_DIR}"
log "Log dir: ${LOG_ROOT}"

export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DS_ACCELERATOR="${DS_ACCELERATOR:-musa}"
export ACCELERATOR_BACKEND="${ACCELERATOR_BACKEND:-musa}"
export MCCL_PROTOS="${MCCL_PROTOS:-2}"
export CE_COMM_ENABLED="${CE_COMM_ENABLED:-0}"
export MUSA_MEMCPY_PATH="${MUSA_MEMCPY_PATH:-3}"
export MUSA_EXECUTION_TIMEOUT="${MUSA_EXECUTION_TIMEOUT:-3200000}"
export MCCL_ALGOS="${MCCL_ALGOS:-1}"
export MCCL_BUFFSIZE="${MCCL_BUFFSIZE:-20971520}"
export MCCL_CHECK_POINTERS="${MCCL_CHECK_POINTERS:-0}"
export MCCL_IB_GID_INDEX="${MCCL_IB_GID_INDEX:-3}"
export MCCL_DEBUG="${MCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTORCH_MUSA_ALLOC_CONF="${PYTORCH_MUSA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_MCCL_AVOID_RECORD_STREAMS="${TORCH_MCCL_AVOID_RECORD_STREAMS:-1}"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

export MODEL_PATH="${MODEL_PATH:-/home/jd/model/Qwen3-VL-30B-A3B-Instruct}"
export DATASET_ROOT="${DATASET_ROOT:-/home/jd/OpenSearch-VL-Search-VL-SFT-36K}"
export YAML_CONFIG="${YAML_CONFIG:-${SCRIPT_DIR}/examples/agentic_full/qwen3_vl_full_sft_30_3b_smoke.yaml}"
export SKIP_INSTALL="${SKIP_INSTALL:-0}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-34237}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

log "Environment:"
log "  MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES}"
log "  MODEL_PATH=${MODEL_PATH}"
log "  DATASET_ROOT=${DATASET_ROOT}"
log "  YAML_CONFIG=${YAML_CONFIG}"
log "  NNODES=${NNODES}, NPROC_PER_NODE=${NPROC_PER_NODE}"

log "Checking MUSA availability..."
python3 - <<'PY'
import torch
print(f"torch={torch.__version__}")
if hasattr(torch, "musa"):
    print(f"musa_available={torch.musa.is_available()}")
    print(f"musa_device_count={torch.musa.device_count()}")
else:
    raise SystemExit("torch.musa not found")
PY

log "Checking model and dataset..."
python3 - <<PY
import json, os
model = "${MODEL_PATH}"
data_dir = "${SCRIPT_DIR}/data"
assert os.path.isdir(model), f"model missing: {model}"
assert os.path.isfile(os.path.join(data_dir, "dataset_info.json"))
with open(os.path.join(data_dir, "new_fvqa/fvqa_llama_factory_clean.json")) as f:
    samples = json.load(f)
img = samples[0]["images"][0]
img_path = os.path.join(data_dir, "new_fvqa", img)
assert os.path.isfile(img_path), f"image missing: {img_path}"
print(f"model_ok={model}")
print(f"dataset_samples={len(samples)}")
print(f"sample_image_ok={img_path}")
PY


if [[ "${SKIP_INSTALL}" != "1" ]]; then
  log "Installing LLaMA-Factory and VL dependencies..."
  pip install -e "${SCRIPT_DIR}[metrics,deepspeed]" -i https://pypi.tuna.tsinghua.edu.cn/simple
  pip install qwen-vl-utils av decord torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

log "Refreshing dataset symlinks..."
bash "${SCRIPT_DIR}/run_qwen3_vl_test.sh" --help >/dev/null 2>&1 || true
# Reuse link logic without launching training.
link_dataset() {
  local sft_subdir="$1"
  local raw_subdir="$2"
  local json_name="$3"
  local sft_data_dir="${SCRIPT_DIR}/data"
  mkdir -p "${sft_data_dir}/${sft_subdir}"
  ln -sfn "${DATASET_ROOT}/${raw_subdir}/${json_name}" "${sft_data_dir}/${sft_subdir}/${json_name}"
  if [[ -d "${DATASET_ROOT}/${raw_subdir}/images/images" ]]; then
    ln -sfn "${DATASET_ROOT}/${raw_subdir}/images/images" "${sft_data_dir}/${sft_subdir}/images"
  elif [[ -d "${DATASET_ROOT}/${raw_subdir}/images" ]]; then
    ln -sfn "${DATASET_ROOT}/${raw_subdir}/images" "${sft_data_dir}/${sft_subdir}/images"
  fi
}
link_dataset new_fvqa fvqa fvqa_llama_factory_clean.json

DEBUG_YAML="${LOG_ROOT}/smoke.node_${NODE_RANK}.yaml"
cp "${YAML_CONFIG}" "${DEBUG_YAML}"
sed -i 's#^deepspeed: .*#deepspeed: examples/deepspeed/ds_z3_config_change.json#' "${DEBUG_YAML}"
sed -i "s#^model_name_or_path: .*#model_name_or_path: ${MODEL_PATH}#" "${DEBUG_YAML}"
sed -i "s#^dataset_dir: .*#dataset_dir: ${SCRIPT_DIR}/data#" "${DEBUG_YAML}"
sed -i "s#^output_dir: .*#output_dir: ${LOG_ROOT}/checkpoints#" "${DEBUG_YAML}"

log "Launching training with ${DEBUG_YAML}"
log "Training output -> ${TRAIN_LOG}"

set +e
FORCE_TORCHRUN=1 \
NNODES="${NNODES}" \
NODE_RANK="${NODE_RANK}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
python -m llamafactory.cli train "${DEBUG_YAML}" 2>&1 | tee -a "${TRAIN_LOG}"
train_exit=${PIPESTATUS[0]}
set -e

if [[ ${train_exit} -ne 0 ]]; then
  log "Training command failed."
  exit ${train_exit}
fi

if grep -E "Qwen3VLMoe|qwen3_vl_moe|vision" "${TRAIN_LOG}" >/dev/null; then
  echo "MODEL_LOADED=Qwen3-VL" >> "${RESULT_FILE}"
fi

if grep -E "'loss':|loss =" "${TRAIN_LOG}" | tail -1 | tee -a "${RESULT_FILE}"; then
  echo "TRAIN_LOSS_FOUND=1" >> "${RESULT_FILE}"
else
  echo "TRAIN_LOSS_FOUND=0" >> "${RESULT_FILE}"
  log "Warning: no loss line found in training log."
fi

log "Smoke test completed."
