"""
向量检索模块 (中文优化版)
使用 BAAI/bge-large-zh 替换 CodeBERT，提升中文语义理解
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import json
import os

class ChineseVectorRetriever:
    """中文优化向量检索器"""
    
    def __init__(self, model_name: str = "BAAI/bge-large-zh", model_path: str = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
        # 加载中文 embedding 模型
        if model_path and os.path.exists(model_path):
            print(f"加载本地模型: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to(self.device)
        else:
            print(f"加载中文 embedding 模型: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir="/root/.cache/huggingface/hub",
                local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir="/root/.cache/huggingface/hub",
                local_files_only=True
            ).to(self.device)
        
        self.model.eval()
        self.entities = []
        self.vectors = None
        self.model_name = model_name
    
    def encode(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """
        编码文本为向量
        BGE 模型需要添加 instruction 前缀用于检索任务
        """
        vectors = []
        
        # BGE 模型推荐的查询前缀
        instruction = "为这个句子生成表示以用于检索相关文章："
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # 添加 instruction 前缀（BGE 推荐做法）
                batch_with_instruction = [instruction + t for t in batch]
                
                inputs = self.tokenizer(
                    batch_with_instruction,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = self.model(**inputs)
                # 使用 [CLS] token 的向量
                batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                vectors.extend(batch_vectors)
        
        return np.array(vectors)
    
    def encode_entities(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """
        编码实体（不加 instruction 前缀）
        """
        vectors = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                inputs = self.tokenizer(
                    batch,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = self.model(**inputs)
                batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                vectors.extend(batch_vectors)
        
        return np.array(vectors)
    
    def build_index(self, entities: list, use_description: bool = True):
        """
        构建向量索引
        
        Args:
            entities: [{"name": "实体名", "desc": "描述", "category": "类别"}, ...]
            use_description: 是否使用描述信息增强编码
        """
        print(f"构建向量索引，共 {len(entities)} 个实体...")
        self.entities = entities
        
        # 构建编码文本（名称 + 描述）
        if use_description:
            texts = []
            for e in entities:
                name = e.get('name', '')
                desc = e.get('desc', '')
                if desc and desc != name:
                    texts.append(f"{name}：{desc}")
                else:
                    texts.append(name)
        else:
            texts = [e.get('name', '') for e in entities]
        
        self.vectors = self.encode_entities(texts)
        print(f"索引构建完成，向量维度: {self.vectors.shape}")
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> list:
        """
        向量相似度搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            threshold: 相似度阈值
            
        Returns:
            [{"name": "实体名", "similarity": 0.95, "entity": {...}}, ...]
        """
        if self.vectors is None:
            return []
        
        # 编码查询（加 instruction）
        query_vec = self.encode([query])
        
        # 计算相似度
        similarities = cosine_similarity(query_vec, self.vectors)[0]
        
        # 获取 top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim >= threshold:
                results.append({
                    'name': self.entities[idx].get('name', ''),
                    'similarity': sim,
                    'entity': self.entities[idx]
                })
        
        return results
    
    def save(self, path: str):
        """保存索引"""
        data = {
            'entities': self.entities,
            'vectors': self.vectors.tolist() if self.vectors is not None else None,
            'model_name': self.model_name
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
        self.model_name = data.get('model_name', 'unknown')
        print(f"索引已加载: {len(self.entities)} 个实体, 模型: {self.model_name}")


# 兼容旧版本的别名
VectorRetriever = ChineseVectorRetriever


def build_entity_vectors():
    """构建实体向量索引（从 Neo4j 加载）"""
    from py2neo import Graph
    
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
    output_dir = '/root/.openclaw/workspace/MiniGraph/data/processed'
    os.makedirs(output_dir, exist_ok=True)
    retriever.save(f'{output_dir}/vector_index_bge.json')
    
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
