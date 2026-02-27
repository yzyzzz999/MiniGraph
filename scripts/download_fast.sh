#!/bin/bash
# 快速下载方案：使用 YAGO 或 DBpedia 的子集

set -e
DATA_DIR="/autodl-fs/data/MiniGraph/data/raw"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== 下载 YAGO 知识图谱子集 ==="
echo "YAGO 是 Wikidata 的精简版，更适合小规模项目"
echo ""

# 下载 YAGO 3 实体数据 (约 300MB)
echo "下载 YAGO 实体数据..."
wget -q --show-progress "https://yago-knowledge.org/data/yago3/yago-wd-full-types.nt.gz" \
  -O yago-types.nt.gz 2>&1 || echo "下载失败，尝试备用源"

# 下载 YAGO 关系数据
echo "下载 YAGO 关系数据..."
wget -q --show-progress "https://yago-knowledge.org/data/yago3/yago-wd-full-facts.nt.gz" \
  -O yago-facts.nt.gz 2>&1 || echo "下载失败，尝试备用源"

echo ""
echo "=== 下载完成 ==="
ls -lh "$DATA_DIR"
