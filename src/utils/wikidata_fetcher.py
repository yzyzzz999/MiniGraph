"""
Wikidata API 数据获取模块
快速获取中文实体和关系
"""
import requests
import json
import time
from typing import List, Dict, Tuple
from pathlib import Path

class WikidataFetcher:
    """Wikidata 数据获取器"""
    
    def __init__(self):
        self.endpoint = "https://query.wikidata.org/sparql"
        self.headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": "MiniGraph/1.0"
        }
        
    def fetch_entities(self, entity_type: str, limit: int = 1000) -> List[Dict]:
        """
        获取指定类型的实体
        
        Args:
            entity_type: 实体类型 QID (如 Q5=人物, Q783794=公司)
            limit: 获取数量
        """
        query = f"""
        SELECT ?item ?itemLabel ?itemDescription WHERE {{
          ?item wdt:P31 wd:{entity_type} .
          ?item schema:description ?itemDescription .
          FILTER(LANG(?itemDescription) = "zh")
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "zh". }}
        }} LIMIT {limit}
        """
        
        try:
            response = requests.get(
                self.endpoint,
                headers=self.headers,
                params={"query": query},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("results", {}).get("bindings", []):
                results.append({
                    "id": item.get("item", {}).get("value", "").split("/")[-1],
                    "name": item.get("itemLabel", {}).get("value", ""),
                    "description": item.get("itemDescription", {}).get("value", ""),
                    "type": entity_type
                })
            return results
            
        except Exception as e:
            print(f"获取实体失败: {e}")
            return []
    
    def fetch_relations(self, entity_id: str) -> List[Dict]:
        """
        获取实体的关系
        
        Args:
            entity_id: 实体ID (如 Q42)
        """
        query = f"""
        SELECT ?property ?propertyLabel ?value ?valueLabel WHERE {{
          wd:{entity_id} ?p ?value .
          ?property wikibase:directClaim ?p .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "zh". }}
        }} LIMIT 50
        """
        
        try:
            response = requests.get(
                self.endpoint,
                headers=self.headers,
                params={"query": query},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("results", {}).get("bindings", []):
                prop = item.get("propertyLabel", {}).get("value", "")
                val = item.get("valueLabel", {}).get("value", "")
                if val and prop:
                    results.append({
                        "property": prop,
                        "value": val,
                        "value_id": item.get("value", {}).get("value", "").split("/")[-1]
                    })
            return results
            
        except Exception as e:
            print(f"获取关系失败: {e}")
            return []
    
    def fetch_batch(self, entity_types: Dict[str, str], limit_per_type: int = 500) -> Tuple[List, List]:
        """
        批量获取多种类型实体
        
        Returns:
            (entities, relations)
        """
        all_entities = []
        all_relations = []
        
        for type_name, qid in entity_types.items():
            print(f"获取 {type_name} 实体...")
            entities = self.fetch_entities(qid, limit_per_type)
            print(f"  获取到 {len(entities)} 个实体")
            all_entities.extend(entities)
            
            # 获取前100个实体的关系
            for entity in entities[:100]:
                if entity["id"]:
                    relations = self.fetch_relations(entity["id"])
                    for rel in relations:
                        all_relations.append({
                            "from": entity["name"],
                            "from_id": entity["id"],
                            "relation": rel["property"],
                            "to": rel["value"],
                            "to_id": rel.get("value_id", "")
                        })
                    time.sleep(0.1)  # 避免请求过快
        
        return all_entities, all_relations


def main():
    """测试获取数据"""
    fetcher = WikidataFetcher()
    
    # 定义要获取的实体类型
    entity_types = {
        "人物": "Q5",
        "公司": "Q783794",
        "城市": "Q515",
        "国家": "Q6256",
        "电影": "Q11424",
    }
    
    print("开始获取 Wikidata 数据...")
    entities, relations = fetcher.fetch_batch(entity_types, limit_per_type=200)
    
    print(f"\n总共获取:")
    print(f"  实体: {len(entities)}")
    print(f"  关系: {len(relations)}")
    
    # 保存数据
    output_dir = Path("/autodl-fs/data/MiniGraph/data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "wikidata_entities.json", "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "wikidata_relations.json", "w", encoding="utf-8") as f:
        json.dump(relations, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据已保存到:")
    print(f"  {output_dir}/wikidata_entities.json")
    print(f"  {output_dir}/wikidata_relations.json")


if __name__ == "__main__":
    main()
