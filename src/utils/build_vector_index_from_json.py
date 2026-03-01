"""
从 relations.json 构建向量索引（无需 Neo4j）
relations.json 包含实体名称和关系
"""
import json
import os
import sys
from collections import defaultdict

# 添加 src 目录到路径
sys.path.insert(0, '/autodl-fs/data/MiniGraph/src')

from vector_retriever_chinese import ChineseVectorRetriever

def build_entity_vectors_from_json():
    """从 relations.json 提取实体并构建向量索引"""
    
    print("从 relations.json 提取实体...")
    relations_path = '/autodl-fs/data/MiniGraph/data/processed/relations.json'
    
    with open(relations_path, 'r', encoding='utf-8') as f:
        relations = json.load(f)
    
    # 收集所有实体及其描述
    entity_desc = defaultdict(list)
    entity_categories = defaultdict(set)
    
    for rel in relations:
        from_entity = rel.get('from', '')
        to_entity = rel.get('to', '')
        relation = rel.get('relation', '')
        
        if from_entity:
            # 如果是描述关系，保存描述
            if relation in ['描述', '定义', '简介']:
                entity_desc[from_entity].append(to_entity)
            # 如果是标签关系
            elif relation in ['标签', '类别', '分类']:
                entity_categories[from_entity].add(to_entity)
    
    # 构建实体列表
    entities = []
    all_entity_names = set()
    for rel in relations:
        all_entity_names.add(rel.get('from', ''))
        all_entity_names.add(rel.get('to', ''))
    
    for name in all_entity_names:
        if name:
            desc = ' '.join(entity_desc.get(name, []))
            category = ' '.join(entity_categories.get(name, set()))
            entities.append({
                'name': name,
                'desc': desc,
                'category': category
            })
    
    print(f"加载了 {len(entities)} 个实体")
    
    # 构建向量索引
    retriever = ChineseVectorRetriever()
    retriever.build_index(entities, use_description=True)
    
    # 保存
    output_path = '/autodl-fs/data/MiniGraph/data/processed/vector_index_bge.json'
    retriever.save(output_path)
    
    print(f"\n向量索引已保存到: {output_path}")
    
    # 测试搜索
    print("\n测试中文搜索:")
    test_queries = ["社会主义", "人工智能", "北京大学", "唐朝"]
    for query in test_queries:
        print(f"\n查询: {query}")
        results = retriever.search(query, top_k=3)
        for r in results:
            print(f"  {r['name']}: {r['similarity']:.4f}")

if __name__ == '__main__':
    build_entity_vectors_from_json()
