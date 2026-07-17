#!/usr/bin/env bash
# Qwen3-VL-30B-A3B smoke test launcher.
# Minimal wrapper over train_30b_test.sh with local paths and a tiny dataset slice.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export YAML_CONFIG="${YAML_CONFIG:-${SCRIPT_DIR}/examples/agentic_full/qwen3_vl_full_sft_30_3b_smoke.yaml}"
export MODEL_PATH="${MODEL_PATH:-/home/jd/model/Qwen3-VL-30B-A3B-Instruct}"
export DATASET_ROOT="${DATASET_ROOT:-/home/jd/OpenSearch-VL-Search-VL-SFT-36K}"
export SFT_DATA_DIR="${SFT_DATA_DIR:-${SCRIPT_DIR}/data}"

echo "[INFO] Qwen3-VL smoke test"
echo "       YAML_CONFIG : ${YAML_CONFIG}"
echo "       MODEL_PATH  : ${MODEL_PATH}"
echo "       DATASET_ROOT: ${DATASET_ROOT}"
echo "       SFT_DATA_DIR: ${SFT_DATA_DIR}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] Model not found: ${MODEL_PATH}"
  exit 1
fi

if [[ ! -f "${YAML_CONFIG}" ]]; then
  echo "[ERROR] YAML not found: ${YAML_CONFIG}"
  exit 1
fi

if [[ ! -f "${SFT_DATA_DIR}/dataset_info.json" ]]; then
  echo "[ERROR] dataset_info.json not found under ${SFT_DATA_DIR}"
  exit 1
fi

# Ensure dataset symlinks exist (idempotent).
link_dataset() {
  local sft_subdir="$1"
  local raw_subdir="$2"
  local json_name="$3"

  mkdir -p "${SFT_DATA_DIR}/${sft_subdir}"
  ln -sfn "${DATASET_ROOT}/${raw_subdir}/${json_name}" \
    "${SFT_DATA_DIR}/${sft_subdir}/${json_name}"

  if [[ -d "${DATASET_ROOT}/${raw_subdir}/images/images" ]]; then
    ln -sfn "${DATASET_ROOT}/${raw_subdir}/images/images" \
      "${SFT_DATA_DIR}/${sft_subdir}/images"
  elif [[ -d "${DATASET_ROOT}/${raw_subdir}/images" ]]; then
    ln -sfn "${DATASET_ROOT}/${raw_subdir}/images" \
      "${SFT_DATA_DIR}/${sft_subdir}/images"
  fi
}

link_dataset new_fvqa fvqa fvqa_llama_factory_clean.json
link_dataset palace palace palace_llama_factory_filtered.json
link_dataset WebQA webqa webqa_llama_factory_filtered.json
link_dataset new_livevqa livevqa livevqa_llama_factory_filtered.json
link_dataset wikiart wiki_art wikiart_llama_factory_filtered.json
link_dataset wiki_zh wiki_zh wiki_zh_llama_factory_filtered.json
link_dataset wiki_en wiki_en wiki_en_llama_factory_filtered.json

# Full-dataset run: override YAML to the trace config.
if [[ "${FULL_DATASET:-0}" == "1" ]]; then
  export YAML_CONFIG="${SCRIPT_DIR}/examples/agentic_full/qwen3_vl_full_sft_30_3b_trace_local.yaml"
  echo "[INFO] FULL_DATASET=1, switching to ${YAML_CONFIG}"
fi

exec bash "${SCRIPT_DIR}/train_30b_test.sh"
