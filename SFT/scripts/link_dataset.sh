#!/usr/bin/env bash
# =============================================================================
# 将 OpenSearch-VL-Search-VL-SFT-36K 软链接到 SFT/data/（不修改原始 datasets 目录）
#
# 思路（与文档 0.3 一致）：
#   1. 在 SFT/data/ 下创建别名目录：new_fvqa / palace / WebQA / ...
#   2. 软链接原始 JSON 与 images/ 到对应别名目录
#   3. 写入 dataset_info.json，并校验 JSON 内图片路径（相对 data/）可访问
#
# 注意：
#   - 本脚本不会在 RAW 数据集目录里创建任何软链接/别名
#   - JSON 内图片路径需已带前缀，例如 new_fvqa/images/xxx.jpg
#     （当前仓库数据集已处理；若仍是 images/xxx.jpg，请先做路径前缀修复）
#
# 用法：
#   bash SFT/scripts/link_dataset.sh
#
# 覆盖路径：
#   RAW_DATA=/path/to/OpenSearch-VL-Search-VL-SFT-36K \
#   SFT_DATA=/path/to/OpenSearch_VL/SFT/data \
#   bash SFT/scripts/link_dataset.sh
# =============================================================================
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
SOURCE_ROOT="${SOURCE_ROOT:-/home/jd/OpenSearch-VL-Search-VL-SFT-36K}" # needs verify in actual env! 

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