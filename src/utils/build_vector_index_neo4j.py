"""
从 Neo4j 构建向量索引
"""
import sys
sys.path.insert(0, '/autodl-fs/data/MiniGraph/src')

from vector_retriever_chinese import ChineseVectorRetriever
from py2neo import Graph

def build_entity_vectors():
    print("从 Neo4j 加载实体...")
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
    
    # 获取所有实体（带描述）
    result = graph.run("""
        MATCH (e:Entity) 
        RETURN e.name as name, e.desc as desc, e.category as category
        LIMIT 10000
    """).data()
    
    entities = [{
        'name': r['name'],
        'desc': r['desc'] or '',
        'category': r['category'] or ''
    } for r in result]
    
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
    build_entity_vectors()
