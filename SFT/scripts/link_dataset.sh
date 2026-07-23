#!/usr/bin/env bash
# =============================================================================
# 将 OpenSearch-VL-Search-VL-SFT-36K 接入 SFT/data/，不修改原始数据集。
#
# 处理方式：
#   1. 将每个原始数据集的 images/ 软链接到 SFT/data/<alias>/images。
#   2. 基于原始 JSON 生成派生 JSON 到 SFT/data/<alias>/，为相对媒体路径
#      添加数据集别名前缀，例如 images/a.jpg -> new_fvqa/images/a.jpg。
#   3. 校验派生后的全部本地媒体路径；发现缺失文件时退出非零。
#   4. 统计 ShareGPT 对话轮数或角色顺序异常，但不擅自修改对话语义。
#
# 用法：
#   bash SFT/scripts/link_dataset.sh
#   bash SFT/scripts/link_dataset.sh --check-only
#
# 覆盖路径：
#   SOURCE_ROOT=/path/to/OpenSearch-VL-Search-VL-SFT-36K \
#   DATA_DIR=/path/to/OpenSearch_vl_musa/SFT/data \
#   bash SFT/scripts/link_dataset.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
SOURCE_ROOT="${SOURCE_ROOT:-/home/jd/liang.geng/datasets/OpenSearch-VL-Search-VL-SFT-36K}"
MODE="apply"

usage() {
  cat <<'EOF'
Usage: bash SFT/scripts/link_dataset.sh [--check-only]

Options:
  --check-only  Validate source JSON/media and print planned rewrites without writing.
  -h, --help    Show this help message.
EOF
}

case "${1:-}" in
  "") ;;
  --check-only) MODE="check" ;;
  -h|--help) usage; exit 0 ;;
  *) echo "[ERROR] unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac

link_path() {
  local src="$1"
  local dst="$2"

  if [[ ! -e "${src}" ]]; then
    echo "[ERROR] source missing: ${src}" >&2
    return 1
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
    return 1
  fi

  mkdir -p "$(dirname "${dst}")"
  ln -s "${src}" "${dst}"
  echo "[LINK] ${dst} -> ${src}"
}

prepare_json() {
  local src_json="$1"
  local dst_json="$2"
  local alias="$3"
  local src_dataset_dir="$4"

  python3 - \
    "${src_json}" \
    "${dst_json}" \
    "${alias}" \
    "${src_dataset_dir}" \
    "${DATA_DIR}" \
    "${MODE}" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse


src_json = Path(sys.argv[1])
dst_json = Path(sys.argv[2])
alias = sys.argv[3]
src_dataset_dir = Path(sys.argv[4])
data_dir = Path(sys.argv[5])
mode = sys.argv[6]


def is_remote(value: str) -> bool:
    return urlparse(value).scheme.lower() in {"http", "https", "s3", "gs", "oss"}


def transform_media(value, stats):
    if isinstance(value, list):
        return [transform_media(item, stats) for item in value]
    if not isinstance(value, str):
        return value

    stats["refs"] += 1
    if is_remote(value):
        stats["remote"] += 1
        return value

    source_value = value.replace("\\", "/")
    while source_value.startswith("./"):
        source_value = source_value[2:]

    path = PurePosixPath(source_value)
    if ".." in path.parts:
        stats["missing"].append(f"unsafe path: {value}")
        return value

    if path.is_absolute():
        rewritten = source_value
        candidate = Path(rewritten)
    elif path.parts and path.parts[0] == alias:
        rewritten = source_value
        candidate = data_dir / rewritten
    else:
        rewritten = f"{alias}/{source_value}"
        candidate = data_dir / rewritten
        stats["rewritten"] += 1

    if not candidate.is_file():
        # --check-only must also work before the target image symlink is created.
        source_relative = PurePosixPath(rewritten)
        if source_relative.parts and source_relative.parts[0] == alias:
            source_relative = PurePosixPath(*source_relative.parts[1:])
        source_candidate = src_dataset_dir / source_relative
        if not source_candidate.is_file():
            stats["missing"].append(rewritten)

    return rewritten


def audit_conversations(rows):
    invalid_count = 0
    invalid_role = 0
    for row in rows:
        messages = row.get("conversations")
        if not isinstance(messages, list):
            continue
        if messages and isinstance(messages[0], dict) and messages[0].get("from") == "system":
            messages = messages[1:]

        aligned_count = 0
        role_is_invalid = False
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                role_is_invalid = True
                break
            allowed = {"human", "observation"} if index % 2 == 0 else {"gpt", "function_call"}
            if message.get("from") not in allowed:
                role_is_invalid = True
                break
            aligned_count += 1

        if role_is_invalid:
            invalid_role += 1
        if aligned_count % 2 != 0:
            invalid_count += 1
    return invalid_count, invalid_role


if not src_json.is_file():
    raise SystemExit(f"[ERROR] source JSON missing: {src_json}")

with src_json.open("r", encoding="utf-8") as handle:
    rows = json.load(handle)

if not isinstance(rows, list):
    raise SystemExit(f"[ERROR] expected a JSON array: {src_json}")

stats = {"refs": 0, "rewritten": 0, "remote": 0, "missing": []}
for row in rows:
    if not isinstance(row, dict):
        raise SystemExit(f"[ERROR] dataset row is not an object: {src_json}")
    for field in ("images", "videos", "audios"):
        if field in row and row[field] is not None:
            row[field] = transform_media(row[field], stats)

invalid_count, invalid_role = audit_conversations(rows)
print(
    f"[CHECK] {alias}: records={len(rows)}, media_refs={stats['refs']}, "
    f"rewritten={stats['rewritten']}, remote={stats['remote']}, "
    f"missing={len(stats['missing'])}, invalid_message_count={invalid_count}, "
    f"invalid_role={invalid_role}"
)

if stats["missing"]:
    for missing in stats["missing"][:20]:
        print(f"[MISSING] {alias}: {missing}", file=sys.stderr)
    if len(stats["missing"]) > 20:
        print(
            f"[MISSING] {alias}: ... and {len(stats['missing']) - 20} more",
            file=sys.stderr,
        )
    raise SystemExit(1)

if mode == "check":
    raise SystemExit(0)

dst_json.parent.mkdir(parents=True, exist_ok=True)
fd, temporary_name = tempfile.mkstemp(
    prefix=f".{dst_json.name}.", suffix=".tmp", dir=dst_json.parent
)
temporary_path = Path(temporary_name)
try:
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(rows, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary_path, 0o644)
    os.replace(temporary_path, dst_json)
finally:
    if temporary_path.exists():
        temporary_path.unlink()

print(f"[WRITE] {dst_json}")
PY
}

process_dataset() {
  local alias="$1"
  local src_dir="$2"
  local json_name="$3"
  local src_dataset_dir="${SOURCE_ROOT}/${src_dir}"
  local src_json="${src_dataset_dir}/${json_name}"
  local dst_dir="${DATA_DIR}/${alias}"
  local dst_json="${dst_dir}/${json_name}"
  local src_images

  if [[ -d "${src_dataset_dir}/images/images" ]]; then
    src_images="${src_dataset_dir}/images/images"
  else
    src_images="${src_dataset_dir}/images"
  fi

  if [[ ! -f "${src_json}" ]]; then
    echo "[ERROR] source JSON missing: ${src_json}" >&2
    return 1
  fi
  if [[ ! -d "${src_images}" ]]; then
    echo "[ERROR] source images directory missing: ${src_images}" >&2
    return 1
  fi

  if [[ "${MODE}" == "apply" ]]; then
    mkdir -p "${dst_dir}"
    link_path "${src_images}" "${dst_dir}/images"
  else
    echo "[PLAN] ${dst_dir}/images -> ${src_images}"
  fi

  prepare_json "${src_json}" "${dst_json}" "${alias}" "${src_dataset_dir}"
}

if [[ ! -f "${DATA_DIR}/dataset_info.json" ]]; then
  echo "[ERROR] missing ${DATA_DIR}/dataset_info.json" >&2
  exit 1
fi

process_dataset "new_fvqa" "fvqa" "fvqa_llama_factory_clean.json"
process_dataset "palace" "palace" "palace_llama_factory_filtered.json"
process_dataset "WebQA" "webqa" "webqa_llama_factory_filtered.json"
process_dataset "new_livevqa" "livevqa" "livevqa_llama_factory_filtered.json"
process_dataset "wikiart" "wiki_art" "wikiart_llama_factory_filtered.json"
process_dataset "wiki_en" "wiki_en" "wiki_en_llama_factory_filtered.json"
process_dataset "wiki_zh" "wiki_zh" "wiki_zh_llama_factory_filtered.json"

if [[ "${MODE}" == "check" ]]; then
  echo "[DONE] Validation passed; no files were changed."
else
  echo "[DONE] Agent SFT data links and derived JSON files are ready under ${DATA_DIR}"
fi
