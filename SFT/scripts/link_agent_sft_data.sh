#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
SOURCE_ROOT="${SOURCE_ROOT:-/home/jd/OpenSearch-VL-Search-VL-SFT-36K}"

link_path() {
  local src="$1"
  local dst="$2"

  if [[ ! -e "${src}" ]]; then
    echo "[ERROR] source missing: ${src}" >&2
    exit 1
  fi

  if [[ -L "${dst}" ]]; then
    local current
    current="$(readlink "${dst}")"
    if [[ "${current}" == "${src}" ]]; then
      echo "[OK] ${dst} -> ${src}"
      return 0
    fi
    echo "[UPDATE] ${dst}: ${current} -> ${src}"
    ln -sfn "${src}" "${dst}"
    return 0
  fi

  if [[ -e "${dst}" ]]; then
    echo "[ERROR] target exists and is not a symlink: ${dst}" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${dst}")"
  ln -s "${src}" "${dst}"
  echo "[LINK] ${dst} -> ${src}"
}

link_dataset() {
  local dst_dir="$1"
  local src_dir="$2"
  local json_name="$3"

  mkdir -p "${DATA_DIR}/${dst_dir}"
  link_path "${SOURCE_ROOT}/${src_dir}/${json_name}" "${DATA_DIR}/${dst_dir}/${json_name}"
  if [[ -d "${SOURCE_ROOT}/${src_dir}/images/images" ]]; then
    link_path "${SOURCE_ROOT}/${src_dir}/images/images" "${DATA_DIR}/${dst_dir}/images"
  else
    link_path "${SOURCE_ROOT}/${src_dir}/images" "${DATA_DIR}/${dst_dir}/images"
  fi
}

if [[ ! -f "${DATA_DIR}/dataset_info.json" ]]; then
  echo "[ERROR] missing ${DATA_DIR}/dataset_info.json" >&2
  exit 1
fi

link_dataset "new_fvqa" "fvqa" "fvqa_llama_factory_clean.json"
link_dataset "palace" "palace" "palace_llama_factory_filtered.json"
link_dataset "WebQA" "webqa" "webqa_llama_factory_filtered.json"
link_dataset "new_livevqa" "livevqa" "livevqa_llama_factory_filtered.json"
link_dataset "wikiart" "wiki_art" "wikiart_llama_factory_filtered.json"
link_dataset "wiki_en" "wiki_en" "wiki_en_llama_factory_filtered.json"
link_dataset "wiki_zh" "wiki_zh" "wiki_zh_llama_factory_filtered.json"

echo "[DONE] Agent SFT data links are ready under ${DATA_DIR}"
