"""
向量编码模块
使用 BERT 将实体编码为向量，存入 Milvus
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
                # 使用 [CLS] token 的向量
                batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                vectors.append(batch_vectors)
        
        return np.vstack(vectors)
    
    def encode_entities(self, entity_names: List[str]) -> np.ndarray:
        """批量编码实体名称"""
        return self.encode(entity_names)


class SimpleVectorStore:
    """简单的向量存储（用 JSON 文件，不依赖 Milvus）"""
    
    def __init__(self, save_path: str = "data/processed/vectors.json"):
        self.save_path = save_path
        self.vectors = {}
        
    def add(self, entity_name: str, vector: np.ndarray):
        """添加向量"""
        self.vectors[entity_name] = vector.tolist()
        
    def add_batch(self, entity_names: List[str], vectors: np.ndarray):
        """批量添加"""
        for name, vec in zip(entity_names, vectors):
            self.vectors[name] = vec.tolist()
            
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        """搜索相似向量（余弦相似度）"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        query_vector = query_vector.reshape(1, -1)
        results = []
        
        for name, vec in self.vectors.items():
            vec = np.array(vec).reshape(1, -1)
            similarity = cosine_similarity(query_vector, vec)[0][0]
            results.append({"name": name, "similarity": float(similarity)})
        
        # 排序并返回 top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def save(self):
        """保存到文件"""
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(self.vectors, f, ensure_ascii=False)
        print(f"向量已保存到: {self.save_path}")
        
    def load(self):
        """从文件加载"""
        import os
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r', encoding='utf-8') as f:
                self.vectors = json.load(f)
            print(f"向量已加载: {len(self.vectors)} 个")


def main():
    """测试编码"""
    # 读取实体
    print("读取实体...")
    with open('/autodl-fs/data/MiniGraph/data/processed/entities.json', 'r') as f:
        entities = json.load(f)
    
    entity_names = [e['name'] for e in entities[:1000]]  # 先编码前1000个
    print(f"将编码 {len(entity_names)} 个实体")
    
    # 编码
    encoder = VectorEncoder()
    vectors = encoder.encode_entities(entity_names)
    print(f"编码完成，向量形状: {vectors.shape}")
    
    # 存储
    store = SimpleVectorStore()
    store.add_batch(entity_names, vectors)
    store.save()
    
    # 测试搜索
    print("\n测试搜索:")
    query_vec = vectors[0]
    results = store.search(query_vec, top_k=5)
    for r in results:
        print(f"  {r['name']}: {r['similarity']:.4f}")


if __name__ == '__main__':
    main()
