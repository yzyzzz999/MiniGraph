#!/bin/bash
# 下载 Wikidata 精选子集

set -e

DATA_DIR="/autodl-fs/data/MiniGraph/data/raw"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== 开始下载 Wikidata 精选子集 ==="
echo "时间: $(date)"
echo ""

# 下载策略: 使用 Wikidata 的 JSON dump，但只下载特定实体类型
# 或者使用 SPARQL 查询获取精选数据

# 方案1: 下载 Wikidata 的 "truthy" 简化版 (约 30GB，太大)
# 方案2: 使用 Wikidata Query Service 分批获取

# 这里采用方案2: 分批获取热门实体

echo "步骤1: 获取热门人物数据..."
curl -s "https://query.wikidata.org/sparql?query=SELECT%20DISTINCT%20%3Fitem%20%3FitemLabel%20WHERE%20%7B%0A%20%20%3Fitem%20wdt%3AP31%20wd%3AQ5%20.%0A%20%20%3Fitem%20wdt%3AP166%20%3Faward%20.%0A%20%20SERVICE%20wikibase%3Alabel%20%7B%20bd%3AserviceParam%20wikibase%3Alanguage%20%22en%2Czh%22.%20%7D%0A%7D%20LIMIT%2010000&format=json" \
  -H "Accept: application/sparql-results+json" \
  -o persons.json || echo "人物数据下载可能需要代理"

echo "步骤2: 获取知名公司数据..."
curl -s "https://query.wikidata.org/sparql?query=SELECT%20DISTINCT%20%3Fitem%20%3FitemLabel%20WHERE%20%7B%0A%20%20%3Fitem%20wdt%3AP31%20wd%3AQ783794%20.%0A%20%20SERVICE%20wikibase%3Alabel%20%7B%20bd%3AserviceParam%20wikibase%3Alanguage%20%22en%2Czh%22.%20%7D%0A%7D%20LIMIT%205000&format=json" \
  -H "Accept: application/sparql-results+json" \
  -o companies.json || echo "公司数据下载可能需要代理"

echo ""
echo "=== 下载完成 ==="
echo "原始数据位置: $DATA_DIR"
ls -lh "$DATA_DIR"
