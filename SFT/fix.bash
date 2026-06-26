#!/bin/bash
RAW_DATA=/home/jd/OpenSearch-VL-Search-VL-SFT-36K
SFT_DATA=/home/jd/OpenSearch-VL-main/SFT/data

# 数据集映射
declare -A MAP=(
    ["fvqa"]="new_fvqa"
    ["webqa"]="WebQA"
    ["livevqa"]="new_livevqa"
    ["palace"]="palace"
    ["wiki_art"]="wikiart"
    ["wiki_zh"]="wiki_zh"
    ["wiki_en"]="wiki_en"
)

echo "========================================="
echo "批量修复嵌套目录"
echo "========================================="

for raw_dir in "${!MAP[@]}"; do
    sft_dir="${MAP[$raw_dir]}"
    
    if [ -d "$RAW_DATA/$raw_dir/images/images" ]; then
        echo "修复: $raw_dir -> $sft_dir"
        cd "$SFT_DATA/$sft_dir"
        rm -f images
        ln -s "$RAW_DATA/$raw_dir/images/images" images
        echo "  ✅ 已修复"
    fi
done

echo ""
echo "✅ 完成"
