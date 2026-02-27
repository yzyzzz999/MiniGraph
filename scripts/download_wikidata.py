"""
Wikidata 数据下载脚本
下载精选子集用于知识图谱构建
"""
import os
import json
import requests
from pathlib import Path

DATA_DIR = Path("/autodl-fs/data/MiniGraph/data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Wikidata 实体类型
ENTITY_TYPES = {
    "humans": "Q5",           # 人物
    "companies": "Q783794",   # 公司
    "cities": "Q515",         # 城市
    "countries": "Q6256",     # 国家
    "movies": "Q11424",       # 电影
    "books": "Q571",          # 书籍
}

def download_via_sparql(entity_type, limit=5000):
    """通过 SPARQL 查询获取实体"""
    query = f"""
    SELECT DISTINCT ?item ?itemLabel WHERE {{
      ?item wdt:P31 wd:{entity_type} .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,zh". }}
    }} LIMIT {limit}
    """
    
    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/sparql-results+json"}
    params = {"query": query}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=120)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"下载 {entity_type} 失败: {e}")
        return None

def save_entities(data, filename):
    """保存实体数据"""
    filepath = DATA_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"保存到: {filepath} ({os.path.getsize(filepath) / 1024 / 1024:.2f} MB)")

def main():
    print("=== MiniGraph Wikidata 数据下载 ===")
    print(f"数据目录: {DATA_DIR}")
    print()
    
    for name, qid in ENTITY_TYPES.items():
        print(f"下载 {name} (类型: {qid})...")
        data = download_via_sparql(qid, limit=5000)
        if data:
            save_entities(data, f"{name}.json")
            if "results" in data and "bindings" in data["results"]:
                print(f"  获取到 {len(data['results']['bindings'])} 个实体")
        print()
    
    print("=== 下载完成 ===")
    print(f"数据文件位置: {DATA_DIR}")

if __name__ == "__main__":
    main()
