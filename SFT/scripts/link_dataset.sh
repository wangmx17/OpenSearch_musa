#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAW_DATASET_DIR="${RAW_DATASET_DIR:-/data/bevformer_bk/liang.geng/jd_test/datasets/OpenSearch-VL-Search-VL-SFT-36K}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"

if [[ ! -d "${RAW_DATASET_DIR}" ]]; then
  echo "[ERROR] RAW_DATASET_DIR not found: ${RAW_DATASET_DIR}" >&2
  exit 1
fi

mkdir -p "${DATA_DIR}"

image_target() {
  local source_dir="$1"
  if [[ -d "${RAW_DATASET_DIR}/${source_dir}/images/images" ]]; then
    echo "${RAW_DATASET_DIR}/${source_dir}/images/images"
  else
    echo "${RAW_DATASET_DIR}/${source_dir}/images"
  fi
}

link_path() {
  local target="$1"
  local link="$2"

  if [[ ! -e "${target}" ]]; then
    echo "[ERROR] Link target does not exist: ${target}" >&2
    exit 1
  fi

  if [[ -L "${link}" || ! -e "${link}" ]]; then
    ln -sfn "${target}" "${link}"
    return
  fi

  echo "[ERROR] Refusing to replace non-symlink path: ${link}" >&2
  exit 1
}

link_dataset_dir() {
  local alias="$1"
  local source_dir="$2"
  local json_file="$3"
  local dest_dir="${DATA_DIR}/${alias}"

  mkdir -p "${dest_dir}"
  link_path "${RAW_DATASET_DIR}/${source_dir}/${json_file}" "${dest_dir}/${json_file}"
  link_path "$(image_target "${source_dir}")" "${dest_dir}/images"
}

ensure_raw_alias() {
  local alias="$1"
  local source_dir="$2"
  local json_file="$3"
  local alias_dir="${RAW_DATASET_DIR}/${alias}"

  mkdir -p "${alias_dir}"
  link_path "${RAW_DATASET_DIR}/${source_dir}/${json_file}" "${alias_dir}/${json_file}"
  link_path "$(image_target "${source_dir}")" "${alias_dir}/images"
}

link_dataset_dir "new_fvqa" "fvqa" "fvqa_llama_factory_clean.json"
link_dataset_dir "palace" "palace" "palace_llama_factory_filtered.json"
link_dataset_dir "WebQA" "webqa" "webqa_llama_factory_filtered.json"
link_dataset_dir "new_livevqa" "livevqa" "livevqa_llama_factory_filtered.json"
link_dataset_dir "wikiart" "wiki_art" "wikiart_llama_factory_filtered.json"
link_dataset_dir "wiki_en" "wiki_en" "wiki_en_llama_factory_filtered.json"
link_dataset_dir "wiki_zh" "wiki_zh" "wiki_zh_llama_factory_filtered.json"

ensure_raw_alias "new_fvqa" "fvqa" "fvqa_llama_factory_clean.json"
ensure_raw_alias "WebQA" "webqa" "webqa_llama_factory_filtered.json"
ensure_raw_alias "new_livevqa" "livevqa" "livevqa_llama_factory_filtered.json"
ensure_raw_alias "wikiart" "wiki_art" "wikiart_llama_factory_filtered.json"

DATA_DIR="${DATA_DIR}" python3 - <<'PY'
import json
import os
from pathlib import Path

data_dir = Path(os.environ["DATA_DIR"])

columns = {
    "messages": "conversations",
    "images": "images",
    "system": "system",
    "tools": "tools",
}
tags = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
}

files = {
    "new_fvqa_agent_sft": "new_fvqa/fvqa_llama_factory_clean.json",
    "palace_agent_sft": "palace/palace_llama_factory_filtered.json",
    "webqa_agent_sft": "WebQA/webqa_llama_factory_filtered.json",
    "livevqa_agent_sft": "new_livevqa/livevqa_llama_factory_filtered.json",
    "wikiart_agent_sft": "wikiart/wikiart_llama_factory_filtered.json",
    "wiki_en_agent_sft": "wiki_en/wiki_en_llama_factory_filtered.json",
    "wiki_zh_agent_sft": "wiki_zh/wiki_zh_llama_factory_filtered.json",
}

info = {
    name: {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": columns,
        "tags": tags,
    }
    for name, file_name in files.items()
}

(data_dir / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

total_rows = 0
total_refs = 0
missing = []
for name, meta in info.items():
    path = data_dir / meta["file_name"]
    rows = json.loads(path.read_text(encoding="utf-8"))
    refs = 0
    for item in rows:
        for image in item.get("images") or []:
            refs += 1
            if not (data_dir / image).exists() and len(missing) < 20:
                missing.append((name, image))

    total_rows += len(rows)
    total_refs += refs
    print(f"{name}: rows={len(rows)} image_refs={refs}")

if missing:
    print("[ERROR] Missing image samples:")
    for name, image in missing:
        print(f"  {name}: {image}")
    raise SystemExit(1)

print(f"TOTAL: rows={total_rows} image_refs={total_refs} missing=0")
PY

echo "[INFO] Dataset prepared under: ${DATA_DIR}"
