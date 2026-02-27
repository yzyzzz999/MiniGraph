"""
向量编码模块
使用 BERT 将实体和文本编码为向量，存入 Milvus
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
        
        self.vector_dim = 768  # BERT-base 维度
        
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 文本列表
            batch_size: 批大小
            
        Returns:
            vectors: 向量数组 (N, 768)
        """
        vectors = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # 编码
                outputs = self.model(**inputs)
                
                # 使用 [CLS] token 的向量作为句子表示
                batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                vectors.append(batch_vectors)
        
        return np.vstack(vectors)
    
    def encode_entity(self, entity_name: str, entity_info: Dict = None) -> np.ndarray:
        """
        编码实体
        
        Args:
            entity_name: 实体名称
            entity_info: 实体额外信息
            
        Returns:
            vector: 向量 (768,)
        """
        # 构建描述文本
        text = entity_name
        if entity_info:
            if 'description' in entity_info:
                text += f"：{entity_info['description']}"
            elif 'type' in entity_info:
                text += f"（{entity_info['type']}）"
        
        vectors = self.encode([text])
        return vectors[0]
    
    def encode_entities_batch(self, entities: List[Dict]) -> np.ndarray:
        """
        批量编码实体
        
        Args:
            entities: [{"name": "...", "description": "..."}]
            
        Returns:
            vectors: 向量数组
        """
        texts = []
        for entity in entities:
            text = entity['name']
            if 'description' in entity:
                text += f"：{entity['description']}"
            texts.append(text)
        
        return self.encode(texts)


class MilvusStore:
    """Milvus 向量存储"""
    
    def __init__(self, collection_name: str = "entity_vectors", 
                 host: str = "localhost", port: str = "19530"):
        from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
        
        self.collection_name = collection_name
        
        # 连接 Milvus
        connections.connect(alias="default", host=host, port=port)
        
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="entity_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="entity_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        ]
        
        # 创建集合
        schema = CollectionSchema(fields, "Entity Vector Collection")
        self.collection = Collection(collection_name, schema)
        
        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="vector", index_params=index_params)
        
    def insert(self, entity_names: List[str], entity_types: List[str], 
               vectors: np.ndarray):
        """插入向量"""
        entities = [
            entity_names,
            entity_types,
            vectors.tolist()
        ]
        self.collection.insert(entities)
        self.collection.flush()
        
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        """搜索相似向量"""
        self.collection.load()
        
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["entity_name", "entity_type"]
        )
        
        # 格式化结果
        hits = []
        for result in results[0]:
            hits.append({
                'entity_name': result.entity.get('entity_name'),
                'entity_type': result.entity.get('entity_type'),
                'distance': result.distance
            })
        
        return hits
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'collection_name': self.collection_name,
            'num_entities': self.collection.num_entities
        }


def main():
    """测试"""
    # 测试编码
    encoder = VectorEncoder()
    
    texts = ["周杰伦", "昆凌", "阿里巴巴", "马云"]
    vectors = encoder.encode(texts)
    print(f"编码结果形状: {vectors.shape}")
    
    # 测试相似度
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(vectors)
    print(f"\n相似度矩阵:\n{sim_matrix}")


if __name__ == '__main__':
    main()
