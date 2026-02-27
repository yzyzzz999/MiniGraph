#!/bin/bash
# DBpedia 知识图谱数据下载脚本
# 正确的下载地址格式

set -e

DATA_DIR="/autodl-fs/data/MiniGraph/data/raw"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=============================================="
echo "MiniGraph - DBpedia 数据下载"
echo "开始时间: $(date)"
echo "数据目录: $DATA_DIR"
echo "=============================================="
echo ""

# DBpedia 2022.09.01 版本
VERSION="2022.09.01"
BASE_URL="https://downloads.dbpedia.org/repo/dbpedia"

# 下载文件列表 (使用正确的路径)
declare -A FILES=(
    ["labels_lang=en.ttl.bz2"]="generic/labels/${VERSION}/labels_lang=en.ttl.bz2"
    ["instance-types_lang=en_specific.ttl.bz2"]="generic/instance-types/${VERSION}/instance-types_lang=en_specific.ttl.bz2"
    ["mappingbased-literals_lang=en.ttl.bz2"]="mappings/mappingbased-literals/${VERSION}/mappingbased-literals_lang=en.ttl.bz2"
    ["mappingbased-objects_lang=en.ttl.bz2"]="mappings/mappingbased-objects/${VERSION}/mappingbased-objects_lang=en.ttl.bz2"
    ["infobox-properties_lang=en.ttl.bz2"]="mappings/infobox-properties/${VERSION}/infobox-properties_lang=en.ttl.bz2"
)

echo "将要下载以下 DBpedia 文件:"
for name in "${!FILES[@]}"; do
    echo "  - $name"
done
echo ""

# 下载每个文件
for name in "${!FILES[@]}"; do
    path="${FILES[$name]}"
    url="${BASE_URL}/${path}"
    
    echo "----------------------------------------------"
    echo "[$(date +%H:%M:%S)] 开始下载: $name"
    echo "URL: $url"
    
    if [ -f "$name" ] && [ -s "$name" ]; then
        size=$(du -h "$name" | cut -f1)
        echo "文件已存在，跳过下载 (大小: $size)"
    else
        wget --progress=bar:force --timeout=300 -O "$name" "$url" 2>&1 && {
            size=$(du -h "$name" | cut -f1)
            echo "[$(date +%H:%M:%S)] 下载完成: $name (大小: $size)"
        } || {
            echo "[$(date +%H:%M:%S)] 下载失败: $name"
            rm -f "$name"
        }
    fi
    echo ""
done

echo "=============================================="
echo "下载任务结束"
echo "完成时间: $(date)"
echo ""
echo "数据文件列表:"
ls -lh "$DATA_DIR" | grep -v "^total"
echo ""
echo "总大小:"
du -sh "$DATA_DIR"
echo "=============================================="
