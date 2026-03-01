"""
统一的向量检索接口
支持多种后端：暴力搜索、Faiss HNSW
"""

import numpy as np
from typing import List, Dict, Optional
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from vector_retriever_chinese import ChineseVectorRetriever

# 可选导入 Faiss
try:
    from faiss_hnsw import FaissHNSWRetriever
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("警告: Faiss 未安装，将使用暴力搜索")


class UnifiedVectorRetriever:
    """
    统一向量检索器
    自动选择最优后端：Faiss HNSW（大规模）或 暴力搜索（小规模）
    """
    
    def __init__(self, 
                 model_name: str = "BAAI/bge-large-zh",
                 model_path: str = None,
                 use_faiss: bool = True,
                 faiss_threshold: int = 10000,  # 超过此数量使用 Faiss
                 faiss_M: int = 16,
                 faiss_efConstruction: int = 200,
                 faiss_efSearch: int = 50):
        """
        初始化统一检索器
        
        参数:
            model_name: Embedding 模型名称
            model_path: 本地模型路径
            use_faiss: 是否使用 Faiss HNSW
            faiss_threshold: 使用 Faiss 的实体数量阈值
            faiss_M: HNSW 参数 M
            faiss_efConstruction: HNSW 构建参数
            faiss_efSearch: HNSW 查询参数
        """
        self.encoder = ChineseVectorRetriever(model_name, model_path)
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_threshold = faiss_threshold
        
        self.faiss_params = {
            'M': faiss_M,
            'efConstruction': faiss_efConstruction,
            'efSearch': faiss_efSearch
        }
        
        self.entities: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.faiss_index: Optional[FaissHNSWRetriever] = None
        self.backend: str = "none"  # "brute_force" 或 "faiss"
    
    def build_index(self, entities: List[str], texts: List[str] = None, vectors: np.ndarray = None):
        """
        构建向量索引
        
        参数:
            entities: 实体名称列表
            texts: 实体描述文本列表（用于编码）
            vectors: 预计算的向量（可选）
        """
        self.entities = entities
        n_entities = len(entities)
        
        # 编码向量
        if vectors is None:
            if texts is None:
                raise ValueError("必须提供 texts 或 vectors")
            print(f"编码 {n_entities} 个实体...")
            vectors = self.encoder.encode(texts)
        
        self.vectors = vectors
        
        # 选择后端
        if self.use_faiss and n_entities >= self.faiss_threshold:
            print(f"实体数量 {n_entities} >= 阈值 {self.faiss_threshold}，使用 Faiss HNSW")
            self._build_faiss_index(entities, vectors)
        else:
            if self.use_faiss:
                print(f"实体数量 {n_entities} < 阈值 {self.faiss_threshold}，使用暴力搜索")
            else:
                print("使用暴力搜索（Faiss 未启用）")
            self.backend = "brute_force"
    
    def _build_faiss_index(self, entities: List[str], vectors: np.ndarray):
        """构建 Faiss HNSW 索引"""
        self.faiss_index = FaissHNSWRetriever(
            dim=vectors.shape[1],
            **self.faiss_params
        )
        self.faiss_index.build_index(entities, vectors)
        self.backend = "faiss"
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        搜索最相似的实体
        
        参数:
            query: 查询文本
            top_k: 返回结果数量
        
        返回:
            [{'entity': name, 'similarity': score, 'backend': backend}, ...]
        """
        # 编码查询
        query_vector = self.encoder.encode([query])[0]
        
        if self.backend == "faiss" and self.faiss_index is not None:
            # 使用 Faiss HNSW
            results = self.faiss_index.search_single(query_vector, top_k=top_k)
            for r in results:
                r['backend'] = 'faiss'
            return results
        else:
            # 使用暴力搜索
            return self._brute_force_search(query_vector, top_k)
    
    def _brute_force_search(self, query_vector: np.ndarray, top_k: int) -> List[Dict]:
        """暴力搜索"""
        # 计算相似度
        similarities = np.dot(self.vectors, query_vector)
        
        # 取 top k
        top_k_idx = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_k_idx:
            results.append({
                'entity': self.entities[idx],
                'similarity': float(similarities[idx]),
                'backend': 'brute_force'
            })
        
        return results
    
    def save(self, filepath: str):
        """保存索引"""
        import json
        
        # 保存配置
        config = {
            'entities': self.entities,
            'backend': self.backend,
            'faiss_params': self.faiss_params,
            'faiss_threshold': self.faiss_threshold
        }
        
        with open(filepath + '.config.json', 'w') as f:
            json.dump(config, f)
        
        # 保存向量
        np.save(filepath + '.vectors.npy', self.vectors)
        
        # 如果使用了 Faiss，保存 Faiss 索引
        if self.backend == "faiss" and self.faiss_index is not None:
            self.faiss_index.save(filepath + '.faiss')
        
        print(f"索引已保存到: {filepath}")
    
    def load(self, filepath: str, model_name: str = None, model_path: str = None):
        """加载索引"""
        import json
        
        # 加载配置
        with open(filepath + '.config.json', 'r') as f:
            config = json.load(f)
        
        self.entities = config['entities']
        self.backend = config['backend']
        self.faiss_params = config['faiss_params']
        self.faiss_threshold = config['faiss_threshold']
        
        # 加载向量
        self.vectors = np.load(filepath + '.vectors.npy')
        
        # 如果使用了 Faiss，加载 Faiss 索引
        if self.backend == "faiss":
            if model_name or model_path:
                self.encoder = ChineseVectorRetriever(model_name, model_path)
            self.faiss_index = FaissHNSWRetriever(**self.faiss_params)
            self.faiss_index.load(filepath + '.faiss')
        
        print(f"索引已加载: {len(self.entities)} 个实体，后端: {self.backend}")


# 兼容性：保持原有接口
class VectorRetriever(UnifiedVectorRetriever):
    """兼容性别名"""
    pass


if __name__ == '__main__':
    print("测试统一向量检索器...")
    
    # 创建测试数据
    entities = ["唐朝", "宋朝", "明朝", "李白", "杜甫", "苹果", "苹果公司"]
    texts = [
        "唐朝是中国历史上的强盛朝代",
        "宋朝是中国历史上的文化繁荣朝代",
        "明朝是中国历史上的最后一个汉族王朝",
        "李白是唐代著名诗人",
        "杜甫是唐代著名诗人",
        "苹果是一种水果",
        "苹果公司是一家科技公司"
    ]
    
    # 创建检索器
    retriever = UnifiedVectorRetriever(use_faiss=False)  # 先测试暴力搜索
    retriever.build_index(entities, texts)
    
    # 测试查询
    queries = ["唐朝的诗人", "科技公司", "水果"]
    
    for query in queries:
        print(f"\n查询: {query}")
        results = retriever.search(query, top_k=3)
        for i, r in enumerate(results):
            print(f"  {i+1}. {r['entity']}: {r['similarity']:.4f} ({r['backend']})")
