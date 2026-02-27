"""
向量检索模块
使用 CodeBERT 编码实体，用 sklearn 做相似度搜索
（不依赖 Milvus，更轻量）
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import json
import os

class VectorRetriever:
    """向量检索器"""
    
    def __init__(self, model_path: str = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
        # 加载 CodeBERT（本地）
        if model_path and os.path.exists(model_path):
            print(f"加载本地模型: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to(self.device)
        else:
            # 使用缓存的 CodeBERT
            print("加载 CodeBERT...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/codebert-base",
                cache_dir="/root/.cache/huggingface/hub"
            )
            self.model = AutoModel.from_pretrained(
                "microsoft/codebert-base",
                cache_dir="/root/.cache/huggingface/hub"
            ).to(self.device)
        
        self.model.eval()
        self.entities = []
        self.vectors = None
    
    def encode(self, texts: list) -> np.ndarray:
        """编码文本为向量"""
        vectors = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = self.model(**inputs)
                # 使用 [CLS] token 的向量
                vec = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                vectors.append(vec[0])
        
        return np.array(vectors)
    
    def build_index(self, entities: list):
        """
        构建向量索引
        
        Args:
            entities: [{"name": "实体名", "desc": "描述"}, ...]
        """
        print(f"构建向量索引，共 {len(entities)} 个实体...")
        self.entities = entities
        
        # 编码实体名称
        texts = [e.get('name', '') for e in entities]
        self.vectors = self.encode(texts)
        
        print(f"索引构建完成，向量维度: {self.vectors.shape}")
    
    def search(self, query: str, top_k: int = 5) -> list:
        """
        向量相似度搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            [{"name": "实体名", "similarity": 0.95}, ...]
        """
        if self.vectors is None:
            return []
        
        # 编码查询
        query_vec = self.encode([query])
        
        # 计算相似度
        similarities = cosine_similarity(query_vec, self.vectors)[0]
        
        # 获取 top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'name': self.entities[idx].get('name', ''),
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def save(self, path: str):
        """保存索引"""
        data = {
            'entities': self.entities,
            'vectors': self.vectors.tolist() if self.vectors is not None else None
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"索引已保存: {path}")
    
    def load(self, path: str):
        """加载索引"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.entities = data['entities']
        self.vectors = np.array(data['vectors']) if data['vectors'] else None
        print(f"索引已加载: {len(self.entities)} 个实体")


def build_entity_vectors():
    """构建实体向量索引"""
    from py2neo import Graph
    
    print("从 Neo4j 加载实体...")
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
    
    # 获取所有实体
    result = graph.run("MATCH (e:Entity) RETURN e.name as name LIMIT 10000").data()
    entities = [{'name': r['name']} for r in result]
    
    print(f"加载了 {len(entities)} 个实体")
    
    # 构建向量索引
    retriever = VectorRetriever()
    retriever.build_index(entities)
    
    # 保存
    retriever.save('/autodl-fs/data/MiniGraph/data/processed/vector_index.json')
    
    # 测试搜索
    print("\n测试搜索:")
    results = retriever.search("社会主义", top_k=5)
    for r in results:
        print(f"  {r['name']}: {r['similarity']:.4f}")


if __name__ == '__main__':
    build_entity_vectors()
