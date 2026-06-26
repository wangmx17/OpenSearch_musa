#!/bin/bash
# ============================================
# 脚本：创建图片目录的软链接
# ============================================

RAW_DATA=/home/jd/OpenSearch-VL-Search-VL-SFT-36K
SFT_DATA=/home/jd/OpenSearch-VL-main/SFT/data

echo "========================================="
echo "创建图片目录软链接"
echo "========================================="

cd "$SFT_DATA"

# 定义映射：目标目录 -> 源图片目录
declare -A LINK_MAP=(
    ["new_fvqa"]="$RAW_DATA/fvqa/images"
    ["palace"]="$RAW_DATA/palace/images"
    ["WebQA"]="$RAW_DATA/webqa/images"
    ["new_livevqa"]="$RAW_DATA/livevqa/images"
    ["wikiart"]="$RAW_DATA/wiki_art/images"
    ["wiki_zh"]="$RAW_DATA/wiki_zh/images"
    ["wiki_en"]="$RAW_DATA/wiki_en/images"
)

for target_dir in "${!LINK_MAP[@]}"; do
    source_dir="${LINK_MAP[$target_dir]}"
    
    # 检查源目录是否存在
    if [ -d "$source_dir" ]; then
        # 如果目标已存在，先删除
        [ -L "$SFT_DATA/$target_dir/images" ] && rm "$SFT_DATA/$target_dir/images"
        [ -d "$SFT_DATA/$target_dir/images" ] && rm -rf "$SFT_DATA/$target_dir/images"
        
        ln -s "$source_dir" "$SFT_DATA/$target_dir/images"
        echo "✅ $target_dir/images → $source_dir"
    else
        echo "⚠️  源目录不存在: $source_dir"
        # 创建空目录防止报错
        mkdir -p "$SFT_DATA/$target_dir/images"
        echo "  已创建空目录: $SFT_DATA/$target_dir/images（请手动放入图片）"
    fi
done

echo ""
echo "========================================="
echo "✅ 图片目录软链接创建完成"
echo "========================================="
