"""
HNSW (Hierarchical Navigable Small World) 自实现
用于高效近似最近邻搜索
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
    # 每层连接的邻居: {layer: [neighbor_id, ...]}
    neighbors: Dict[int, List[int]]
    # 节点出现的最高层
    max_layer: int


class HNSWIndex:
    """
    HNSW 索引实现
    
    参数:
        M: 每个节点的最大连接数
        ef_construction: 构建时的搜索范围
        ef_search: 查询时的搜索范围
        mL: 层数因子 (1/ln(M))
    """
    
    def __init__(self, M: int = 16, ef_construction: int = 200, ef_search: int = 50):
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.mL = 1.0 / np.log(M)
        
        # 存储所有节点
        self.nodes: Dict[int, HNSWNode] = {}
        
        # 入口点（最高层的节点）
        self.entry_point: int = None
        
        # 当前最大层数
        self.max_layer = 0
        
        # 向量维度
        self.dim = None
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算余弦相似度"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def _get_random_level(self) -> int:
        """
        随机选择节点层数
        使用指数分布，越高层概率越低
        """
        level = 0
        while random.random() < self.mL and level < 16:
            level += 1
        return level
    
    def _search_layer(self, query: np.ndarray, 
                      entry_point: int, 
                      ef: int, 
                      layer: int) -> List[Tuple[float, int]]:
        """
        在指定层搜索最近邻
        
        返回: [(similarity, node_id), ...] 按相似度降序
        """
        # 访问过的节点
        visited: Set[int] = {entry_point}
        
        # 候选集: 用最大堆（Python 只有最小堆，所以存负值）
        # (负相似度, node_id)
        entry_sim = self._cosine_similarity(query, self.nodes[entry_point].vector)
        candidates = [(-entry_sim, entry_point)]
        
        # 结果集: 同样用最大堆，保持 ef 个最佳结果
        results = [(-entry_sim, entry_point)]
        
        while candidates:
            # 取出相似度最大的候选
            neg_sim, curr_id = heapq.heappop(candidates)
            curr_sim = -neg_sim
            
            # 如果当前相似度小于结果集中最差的，停止
            if len(results) >= ef:
                worst_sim = -results[0][0]  # 最大堆顶是最差的
                if curr_sim < worst_sim:
                    break
            
            # 遍历当前节点的邻居
            node = self.nodes[curr_id]
            if layer not in node.neighbors:
                continue
                
            for neighbor_id in node.neighbors[layer]:
                if neighbor_id in visited:
                    continue
                    
                visited.add(neighbor_id)
                neighbor_sim = self._cosine_similarity(query, self.nodes[neighbor_id].vector)
                
                # 加入候选集继续探索
                heapq.heappush(candidates, (-neighbor_sim, neighbor_id))
                
                # 加入结果集
                heapq.heappush(results, (-neighbor_sim, neighbor_id))
                
                # 保持结果集大小为 ef
                if len(results) > ef:
                    heapq.heappop(results)  # 移除最差的
        
        # 转换为列表并排序（按相似度降序）
        final_results = sorted([(-sim, nid) for sim, nid in results], reverse=True)
        return final_results
    
    def _select_neighbors(self, query: np.ndarray, 
                          candidates: List[Tuple[float, int]], 
                          M: int) -> List[int]:
        """
        从候选集中选择 M 个邻居
        使用简单启发式：选择距离最近的 M 个
        """
        # 按距离排序，取前 M 个
        candidates = sorted(candidates, reverse=True)
        return [nid for _, nid in candidates[:M]]
    
    def add_item(self, item_id: int, vector: np.ndarray):
        """
        添加一个向量到索引
        """
        if self.dim is None:
            self.dim = len(vector)
        
        # 随机选择层数
        level = self._get_random_level()
        
        # 创建新节点
        new_node = HNSWNode(
            id=item_id,
            vector=vector,
            neighbors={},
            max_layer=level
        )
        self.nodes[item_id] = new_node
        
        # 如果是第一个节点
        if self.entry_point is None:
            self.entry_point = item_id
            self.max_layer = level
            return
        
        # 从最高层开始搜索
        curr_ep = self.entry_point
        
        # 对于高于新节点层数的层，只找最近邻作为入口
        for layer in range(self.max_layer, level, -1):
            results = self._search_layer(vector, curr_ep, ef=1, layer=layer)
            if results:
                curr_ep = results[0][1]
        
        # 在新节点层数及以下层，建立连接
        for layer in range(min(level, self.max_layer), -1, -1):
            # 搜索 ef_construction 个最近邻
            results = self._search_layer(vector, curr_ep, ef=self.ef_construction, layer=layer)
            
            # 选择 M 个邻居
            neighbors = self._select_neighbors(vector, results, self.M)
            new_node.neighbors[layer] = neighbors
            
            # 双向连接：更新邻居的连接
            for neighbor_id in neighbors:
                neighbor = self.nodes[neighbor_id]
                if layer not in neighbor.neighbors:
                    neighbor.neighbors[layer] = []
                
                # 添加反向连接
                if item_id not in neighbor.neighbors[layer]:
                    neighbor.neighbors[layer].append(item_id)
                
                # 如果邻居连接数超过 M，需要剪枝
                if len(neighbor.neighbors[layer]) > self.M:
                    # 重新选择邻居：保留距离最近的 M 个
                    neighbor_vec = neighbor.vector
                    all_neighbors = neighbor.neighbors[layer]
                    
                    # 计算到所有邻居的距离
                    neighbor_dists = []
                    for nid in all_neighbors:
                        if nid in self.nodes:
                            dist = self._cosine_similarity(neighbor_vec, self.nodes[nid].vector)
                            neighbor_dists.append((dist, nid))
                    
                    # 保留最近的 M 个
                    neighbor_dists = sorted(neighbor_dists, reverse=True)
                    neighbor.neighbors[layer] = [nid for _, nid in neighbor_dists[:self.M]]
            
            # 更新入口点为当前层最近邻
            if results:
                curr_ep = results[0][1]
        
        # 更新最大层数
        if level > self.max_layer:
            self.max_layer = level
            self.entry_point = item_id
    
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[float, int]]:
        """
        搜索 k 个最近邻
        
        返回: [(similarity, node_id), ...] 按相似度降序
        """
        if self.entry_point is None:
            return []
        
        curr_ep = self.entry_point
        
        # 从最高层开始，每层找到最近邻作为下一层入口
        # 注意：只搜索节点存在的层
        for layer in range(self.max_layer, 0, -1):
            # 检查入口点是否在该层有连接
            if layer in self.nodes[curr_ep].neighbors and self.nodes[curr_ep].neighbors[layer]:
                results = self._search_layer(query, curr_ep, ef=1, layer=layer)
                if results:
                    curr_ep = results[0][1]
        
        # 在第 0 层搜索 ef_search 个候选
        results = self._search_layer(query, curr_ep, ef=self.ef_search, layer=0)
        
        # 返回 top k
        return results[:k]
    
    def save(self, filepath: str):
        """保存索引到文件"""
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
                'neighbors': node.neighbors,
                'max_layer': node.max_layer
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath: str):
        """从文件加载索引"""
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
    """
    基于 HNSW 的向量检索器
    替代原来的暴力搜索
    """
    
    def __init__(self, M: int = 16, ef_construction: int = 200, ef_search: int = 50):
        self.index = HNSWIndex(M=M, ef_construction=ef_construction, ef_search=ef_search)
        self.entities: List[str] = []
        self.vectors: np.ndarray = None
        self.id_to_name: Dict[int, str] = {}
        self.name_to_id: Dict[str, int] = {}
    
    def build_index(self, entities: List[str], vectors: np.ndarray):
        """
        构建 HNSW 索引
        
        参数:
            entities: 实体名称列表
            vectors: 向量数组 (n_entities, dim)
        """
        self.entities = entities
        self.vectors = vectors
        
        print(f"构建 HNSW 索引: {len(entities)} 个实体, {vectors.shape[1]} 维")
        print(f"参数: M={self.index.M}, ef_construction={self.index.ef_construction}")
        
        # 建立名称到 ID 的映射
        for i, name in enumerate(entities):
            self.id_to_name[i] = name
            self.name_to_id[name] = i
        
        # 逐个添加向量
        for i in range(len(entities)):
            if i % 1000 == 0:
                print(f"  已添加 {i}/{len(entities)} 个向量")
            self.index.add_item(i, vectors[i])
        
        print(f"HNSW 索引构建完成")
        print(f"  最大层数: {self.index.max_layer}")
        print(f"  入口点: {self.index.entry_point}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        搜索最相似的实体
        
        参数:
            query: 查询文本（需要先编码成向量）
            top_k: 返回结果数量
        
        返回:
            [{'entity': name, 'similarity': score}, ...]
        """
        # 注意：这里假设 query 已经是向量
        # 实际使用时需要先用 embedding 模型编码
        if isinstance(query, str):
            raise ValueError("query 必须是向量，请先使用 embedding 模型编码")
        
        results = self.index.search(query, k=top_k)
        
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
        # 同时保存实体列表
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
        # 加载实体列表
        entity_file = filepath.replace('.json', '_entities.json')
        with open(entity_file, 'r') as f:
            data = json.load(f)
            self.entities = data['entities']
            self.id_to_name = {int(k): v for k, v in data['id_to_name'].items()}
            self.name_to_id = data['name_to_id']


if __name__ == '__main__':
    # 简单测试
    print("测试 HNSW 实现...")
    
    # 创建随机向量
    np.random.seed(42)
    n = 1000
    dim = 128
    vectors = np.random.randn(n, dim).astype(np.float32)
    
    # 归一化
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    entities = [f"entity_{i}" for i in range(n)]
    
    # 构建索引
    retriever = HNSWVectorRetriever(M=16, ef_construction=100, ef_search=50)
    retriever.build_index(entities, vectors)
    
    # 测试查询
    query = vectors[0]  # 用第一个向量作为查询
    results = retriever.search(query, top_k=10)
    
    print(f"\n查询结果 (top 10):")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['entity']}: {r['similarity']:.4f}")
    
    # 验证第一个结果应该是自己
    print(f"\n验证: 第一个结果应该是 entity_0")
    print(f"实际: {results[0]['entity']}, 相似度: {results[0]['similarity']:.4f}")
