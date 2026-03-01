"""
HNSW (Hierarchical Navigable Small World) 简化实现
基于论文: Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Set
import heapq
from dataclasses import dataclass
import json


@dataclass
class HNSWNode:
    """HNSW 节点"""
    id: int
    vector: np.ndarray
    neighbors: Dict[int, List[int]]  # layer -> neighbor_ids
    max_layer: int


class HNSWIndex:
    """
    简化版 HNSW 实现
    """
    
    def __init__(self, M: int = 16, ef_construction: int = 200, ef_search: int = 50):
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.mL = 1.0 / np.log(M)
        
        self.nodes: Dict[int, HNSWNode] = {}
        self.entry_point: int = None
        self.max_layer = 0
        self.dim = None
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算余弦相似度"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def _get_random_level(self) -> int:
        """随机选择节点层数"""
        level = 0
        while random.random() < self.mL and level < 16:
            level += 1
        return level
    
    def _search_layer_simple(self, query: np.ndarray, entry_point: int, ef: int, layer: int) -> List[Tuple[float, int]]:
        """
        简化版层搜索 - 使用 BFS
        """
        visited = {entry_point}
        candidates = [(self._cosine_similarity(query, self.nodes[entry_point].vector), entry_point)]
        results = [(self._cosine_similarity(query, self.nodes[entry_point].vector), entry_point)]
        
        while candidates:
            # 取出相似度最高的候选
            candidates.sort(reverse=True)
            curr_sim, curr_id = candidates.pop(0)
            
            # 检查是否终止
            if len(results) >= ef:
                results.sort(reverse=True)
                if curr_sim < results[ef-1][0]:
                    break
            
            # 遍历邻居
            node = self.nodes[curr_id]
            if layer not in node.neighbors:
                continue
                
            for neighbor_id in node.neighbors[layer]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    neighbor_sim = self._cosine_similarity(query, self.nodes[neighbor_id].vector)
                    candidates.append((neighbor_sim, neighbor_id))
                    results.append((neighbor_sim, neighbor_id))
        
        # 返回 top ef
        results.sort(reverse=True)
        return results[:ef]
    
    def _select_neighbors_simple(self, query: np.ndarray, candidates: List[Tuple[float, int]], M: int) -> List[int]:
        """选择 M 个最近邻居"""
        candidates = sorted(candidates, reverse=True)
        return [nid for _, nid in candidates[:M]]
    
    def add_item(self, item_id: int, vector: np.ndarray):
        """添加向量到索引"""
        if self.dim is None:
            self.dim = len(vector)
        
        level = self._get_random_level()
        
        new_node = HNSWNode(
            id=item_id,
            vector=vector,
            neighbors={},
            max_layer=level
        )
        self.nodes[item_id] = new_node
        
        if self.entry_point is None:
            self.entry_point = item_id
            self.max_layer = level
            return
        
        # 找到入口点
        curr_ep = self.entry_point
        
        # 在高于新节点层数的层，只找最近邻
        for layer in range(self.max_layer, level, -1):
            if layer in self.nodes[curr_ep].neighbors:
                results = self._search_layer_simple(vector, curr_ep, ef=1, layer=layer)
                if results:
                    curr_ep = results[0][1]
        
        # 在新节点层数及以下层，建立连接
        for layer in range(min(level, self.max_layer), -1, -1):
            results = self._search_layer_simple(vector, curr_ep, ef=self.ef_construction, layer=layer)
            
            neighbors = self._select_neighbors_simple(vector, results, self.M)
            new_node.neighbors[layer] = neighbors
            
            # 双向连接
            for neighbor_id in neighbors:
                neighbor = self.nodes[neighbor_id]
                if layer not in neighbor.neighbors:
                    neighbor.neighbors[layer] = []
                if item_id not in neighbor.neighbors[layer]:
                    neighbor.neighbors[layer].append(item_id)
                
                # 剪枝
                if len(neighbor.neighbors[layer]) > self.M:
                    neighbor_vec = neighbor.vector
                    all_neighbors = neighbor.neighbors[layer]
                    neighbor_dists = [(self._cosine_similarity(neighbor_vec, self.nodes[nid].vector), nid) 
                                     for nid in all_neighbors if nid in self.nodes]
                    neighbor_dists = sorted(neighbor_dists, reverse=True)
                    neighbor.neighbors[layer] = [nid for _, nid in neighbor_dists[:self.M]]
            
            if results:
                curr_ep = results[0][1]
        
        if level > self.max_layer:
            self.max_layer = level
            self.entry_point = item_id
    
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[float, int]]:
        """搜索 k 个最近邻"""
        if self.entry_point is None:
            return []
        
        curr_ep = self.entry_point
        
        # 从高层往低层搜索
        for layer in range(self.max_layer, 0, -1):
            if layer in self.nodes[curr_ep].neighbors and self.nodes[curr_ep].neighbors[layer]:
                results = self._search_layer_simple(query, curr_ep, ef=1, layer=layer)
                if results:
                    curr_ep = results[0][1]
        
        # 在第 0 层搜索
        results = self._search_layer_simple(query, curr_ep, ef=self.ef_search, layer=0)
        
        return results[:k]
    
    def save(self, filepath: str):
        """保存索引"""
        data = {
            'M': self.M,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'mL': self.mL,
            'max_layer': self.max_layer,
            'entry_point': self.entry_point,
            'dim': self.dim,
            'nodes': {}
        }
        
        for nid, node in self.nodes.items():
            data['nodes'][nid] = {
                'id': node.id,
                'vector': node.vector.tolist(),
                'neighbors': {str(k): v for k, v in node.neighbors.items()},
                'max_layer': node.max_layer
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath: str):
        """加载索引"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.M = data['M']
        self.ef_construction = data['ef_construction']
        self.ef_search = data['ef_search']
        self.mL = data['mL']
        self.max_layer = data['max_layer']
        self.entry_point = data['entry_point']
        self.dim = data['dim']
        
        self.nodes = {}
        for nid, node_data in data['nodes'].items():
            self.nodes[int(nid)] = HNSWNode(
                id=node_data['id'],
                vector=np.array(node_data['vector']),
                neighbors={int(k): v for k, v in node_data['neighbors'].items()},
                max_layer=node_data['max_layer']
            )


class HNSWVectorRetriever:
    """基于 HNSW 的向量检索器"""
    
    def __init__(self, M: int = 16, ef_construction: int = 200, ef_search: int = 50):
        self.index = HNSWIndex(M=M, ef_construction=ef_construction, ef_search=ef_search)
        self.entities: List[str] = []
        self.vectors: np.ndarray = None
        self.id_to_name: Dict[int, str] = {}
        self.name_to_id: Dict[str, int] = {}
    
    def build_index(self, entities: List[str], vectors: np.ndarray):
        """构建 HNSW 索引"""
        self.entities = entities
        self.vectors = vectors
        
        print(f"构建 HNSW 索引: {len(entities)} 个实体, {vectors.shape[1]} 维")
        print(f"参数: M={self.index.M}, ef_construction={self.index.ef_construction}")
        
        for i, name in enumerate(entities):
            self.id_to_name[i] = name
            self.name_to_id[name] = i
        
        for i in range(len(entities)):
            if i % 1000 == 0:
                print(f"  已添加 {i}/{len(entities)} 个向量")
            self.index.add_item(i, vectors[i])
        
        print(f"HNSW 索引构建完成")
        print(f"  最大层数: {self.index.max_layer}")
        print(f"  入口点: {self.index.entry_point}")
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """搜索最相似的实体"""
        results = self.index.search(query_vector, k=top_k)
        
        return [
            {
                'entity': self.id_to_name[nid],
                'similarity': float(sim)
            }
            for sim, nid in results
        ]
    
    def save(self, filepath: str):
        """保存索引"""
        self.index.save(filepath)
        entity_file = filepath.replace('.json', '_entities.json')
        with open(entity_file, 'w') as f:
            json.dump({
                'entities': self.entities,
                'id_to_name': self.id_to_name,
                'name_to_id': self.name_to_id
            }, f)
    
    def load(self, filepath: str):
        """加载索引"""
        self.index.load(filepath)
        entity_file = filepath.replace('.json', '_entities.json')
        with open(entity_file, 'r') as f:
            data = json.load(f)
            self.entities = data['entities']
            self.id_to_name = {int(k): v for k, v in data['id_to_name'].items()}
            self.name_to_id = data['name_to_id']


if __name__ == '__main__':
    print("测试 HNSW 实现...")
    
    np.random.seed(42)
    n = 1000
    dim = 128
    vectors = np.random.randn(n, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    entities = [f"entity_{i}" for i in range(n)]
    
    retriever = HNSWVectorRetriever(M=16, ef_construction=100, ef_search=50)
    retriever.build_index(entities, vectors)
    
    # 测试查询
    query = vectors[0]
    results = retriever.search(query, top_k=10)
    
    print(f"\n查询结果 (top 10):")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['entity']}: {r['similarity']:.4f}")
    
    print(f"\n验证: 第一个结果应该是 entity_0")
    print(f"实际: {results[0]['entity']}, 相似度: {results[0]['similarity']:.4f}")
