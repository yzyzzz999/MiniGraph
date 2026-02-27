"""
向量编码模块 - Milvus Lite 版本
无需 Docker，直接 pip 安装使用
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
import json

class VectorEncoder:
    """BERT 向量编码器"""
    
    def __init__(self, model_name: str = "bert-base-chinese", device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        print(f"加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.vector_dim = 768
        
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """编码文本为向量"""
        vectors = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**inputs)
                batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                vectors.append(batch_vectors)
        
        return np.vstack(vectors)


class MilvusLiteStore:
    """Milvus Lite 向量存储"""
    
    def __init__(self, collection_name: str = "entity_vectors", db_path: str = "./milvus.db"):
        try:
            from pymilvus import MilvusClient
            self.client = MilvusClient(db_path)
            self.collection_name = collection_name
            self._create_collection()
        except ImportError:
            print("Milvus Lite 未安装，使用备用方案")
            self.client = None
            self.vectors = {}
    
    def _create_collection(self):
        """创建集合"""
        if self.client is None:
            return
            
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=768,
            metric_type="COSINE"
        )
    
    def add_batch(self, entity_names: List[str], vectors: np.ndarray):
        """批量添加向量"""
        if self.client is None:
            # 备用：存到内存
            for name, vec in zip(entity_names, vectors):
                self.vectors[name] = vec.tolist()
            return
        
        data = [
            {"id": i, "vector": vec.tolist(), "name": name}
            for i, (name, vec) in enumerate(zip(entity_names, vectors))
        ]
        self.client.insert(self.collection_name, data)
    
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        """搜索相似向量"""
        if self.client is None:
            # 备用：简单线性搜索
            from sklearn.metrics.pairwise import cosine_similarity
            query = query_vector.reshape(1, -1)
            results = []
            for name, vec in self.vectors.items():
                sim = cosine_similarity(query, np.array(vec).reshape(1, -1))[0][0]
                results.append({"name": name, "distance": 1-sim, "similarity": float(sim)})
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector.tolist()],
            limit=top_k,
            output_fields=["name"]
        )
        return [{"name": r["entity"]["name"], "distance": r["distance"]} for r in results[0]]


def main():
    """测试"""
    print("测试向量编码...")
    
    # 测试数据
    texts = ["周杰伦", "昆凌", "阿里巴巴", "马云", "腾讯", "马化腾"]
    
    encoder = VectorEncoder()
    vectors = encoder.encode(texts)
    print(f"编码完成: {vectors.shape}")
    
    store = MilvusLiteStore()
    store.add_batch(texts, vectors)
    print("存储完成")
    
    # 测试搜索
    print("\n搜索相似:")
    results = store.search(vectors[0], top_k=3)
    for r in results:
        print(f"  {r['name']}: {r.get('similarity', 1-r['distance']):.4f}")


if __name__ == '__main__':
    main()
