"""
基于 Faiss 的 HNSW 向量检索
使用 faiss 的 IndexHNSWFlat 实现高效近似最近邻搜索
"""

import numpy as np
import faiss
from typing import List, Dict
import json


class FaissHNSWRetriever:
    """
    基于 Faiss HNSW 的向量检索器
    
    参数:
        dim: 向量维度
        M: 每个节点的最大连接数（默认 16）
        efConstruction: 构建时的搜索范围（默认 200）
        efSearch: 查询时的搜索范围（默认 50）
    """
    
    def __init__(self, dim: int = 1024, M: int = 16, efConstruction: int = 200, efSearch: int = 50):
        self.dim = dim
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        
        # Faiss 索引
        self.index = None
        
        # 实体名称映射
        self.entities: List[str] = []
        self.id_to_name: Dict[int, str] = {}
        self.name_to_id: Dict[str, int] = {}
    
    def build_index(self, entities: List[str], vectors: np.ndarray):
        """
        构建 HNSW 索引
        
        参数:
            entities: 实体名称列表
            vectors: 向量数组 (n_entities, dim)，必须归一化
        """
        print(f"构建 Faiss HNSW 索引: {len(entities)} 个实体, {vectors.shape[1]} 维")
        print(f"参数: M={self.M}, efConstruction={self.efConstruction}")
        
        self.entities = entities
        n_entities = len(entities)
        
        # 建立名称到 ID 的映射
        for i, name in enumerate(entities):
            self.id_to_name[i] = name
            self.name_to_id[name] = i
        
        # 确保向量是 float32 类型
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        
        # 创建 Faiss HNSW 索引（使用内积，因为向量已归一化，内积=余弦相似度）
        self.index = faiss.IndexHNSWFlat(self.dim, self.M)
        self.index.hnsw.efConstruction = self.efConstruction
        
        # 添加向量
        print(f"  添加 {n_entities} 个向量到索引...")
        self.index.add(vectors)
        
        # 设置查询参数
        self.index.hnsw.efSearch = self.efSearch
        
        print(f"Faiss HNSW 索引构建完成")
        print(f"  索引大小: {self.index.ntotal} 个向量")
    
    def search(self, query_vectors: np.ndarray, top_k: int = 10) -> List[List[Dict]]:
        """
        批量搜索最相似的实体
        
        参数:
            query_vectors: 查询向量数组 (n_queries, dim)
            top_k: 每个查询返回的结果数量
        
        返回:
            [[{'entity': name, 'similarity': score}, ...], ...]
        """
        if self.index is None:
            raise ValueError("索引未构建")
        
        # 确保是 float32
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)
        
        # 批量搜索
        distances, indices = self.index.search(query_vectors, top_k)
        
        # 转换结果
        results = []
        for dists, idxs in zip(distances, indices):
            query_results = []
            for dist, idx in zip(dists, idxs):
                if idx >= 0:  # Faiss 返回 -1 表示未找到
                    query_results.append({
                        'entity': self.id_to_name[idx],
                        'similarity': float(dist)  # 内积值（因为向量归一化，范围 [-1, 1]）
                    })
            results.append(query_results)
        
        return results
    
    def search_single(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        单个向量搜索
        
        参数:
            query_vector: 查询向量 (dim,)
            top_k: 返回结果数量
        
        返回:
            [{'entity': name, 'similarity': score}, ...]
        """
        # 添加 batch 维度
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        results = self.search(query_vector, top_k)
        return results[0] if results else []
    
    def save(self, filepath: str):
        """保存索引和实体映射"""
        # 保存 Faiss 索引
        faiss.write_index(self.index, filepath)
        
        # 保存实体映射
        meta_file = filepath + '.meta.json'
        with open(meta_file, 'w') as f:
            json.dump({
                'entities': self.entities,
                'id_to_name': self.id_to_name,
                'name_to_id': self.name_to_id,
                'dim': self.dim,
                'M': self.M,
                'efConstruction': self.efConstruction,
                'efSearch': self.efSearch
            }, f)
        
        print(f"索引已保存到: {filepath}")
        print(f"元数据已保存到: {meta_file}")
    
    def load(self, filepath: str):
        """加载索引和实体映射"""
        # 加载 Faiss 索引
        self.index = faiss.read_index(filepath)
        
        # 加载实体映射
        meta_file = filepath + '.meta.json'
        with open(meta_file, 'r') as f:
            data = json.load(f)
            self.entities = data['entities']
            self.id_to_name = {int(k): v for k, v in data['id_to_name'].items()}
            self.name_to_id = data['name_to_id']
            self.dim = data['dim']
            self.M = data['M']
            self.efConstruction = data['efConstruction']
            self.efSearch = data['efSearch']
        
        print(f"索引已加载: {self.index.ntotal} 个向量")


# 兼容性包装类
class HNSWVectorRetriever(FaissHNSWRetriever):
    """兼容性包装类"""
    pass


if __name__ == '__main__':
    print("测试 Faiss HNSW 实现...")
    
    # 创建测试数据
    np.random.seed(42)
    n = 10000
    dim = 1024
    vectors = np.random.randn(n, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    entities = [f"entity_{i}" for i in range(n)]
    
    # 构建索引
    import time
    start = time.time()
    retriever = FaissHNSWRetriever(dim=dim, M=16, efConstruction=200, efSearch=50)
    retriever.build_index(entities, vectors)
    build_time = time.time() - start
    print(f"构建时间: {build_time:.2f}s")
    
    # 测试查询
    query = vectors[0]
    start = time.time()
    results = retriever.search_single(query, top_k=10)
    query_time = (time.time() - start) * 1000
    print(f"查询时间: {query_time:.2f}ms")
    
    print(f"\n查询结果 (top 10):")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['entity']}: {r['similarity']:.4f}")
    
    print(f"\n验证: 第一个结果应该是 entity_0")
    print(f"实际: {results[0]['entity']}, 相似度: {results[0]['similarity']:.4f}")
