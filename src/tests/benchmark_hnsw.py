#!/usr/bin/env python3
"""
HNSW 性能对比测试
对比自实现 HNSW vs 暴力搜索
"""

import numpy as np
import time
import json
from typing import List, Tuple
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from retriever.hnsw_index import HNSWVectorRetriever


def brute_force_search(vectors: np.ndarray, query: np.ndarray, k: int = 10) -> List[Tuple[float, int]]:
    """暴力搜索（余弦相似度）"""
    # 计算所有相似度
    similarities = np.dot(vectors, query)
    
    # 取 top k
    top_k_idx = np.argsort(similarities)[::-1][:k]
    
    return [(float(similarities[i]), int(i)) for i in top_k_idx]


def recall_at_k(hnsw_results: List[Tuple[float, int]], 
                brute_results: List[Tuple[float, int]], 
                k: int = 10) -> float:
    """计算召回率"""
    hnsw_set = set([nid for _, nid in hnsw_results[:k]])
    brute_set = set([nid for _, nid in brute_results[:k]])
    
    intersection = hnsw_set & brute_set
    return len(intersection) / len(brute_set)


def benchmark(n_entities: int = 10000, dim: int = 1024, n_queries: int = 100):
    """
    性能对比测试
    
    参数:
        n_entities: 实体数量
        dim: 向量维度
        n_queries: 查询次数
    """
    print("=" * 60)
    print(f"HNSW 性能对比测试")
    print(f"实体数: {n_entities}, 维度: {dim}, 查询数: {n_queries}")
    print("=" * 60)
    
    # 生成随机向量（模拟 BGE 向量）
    print("\n1. 生成测试数据...")
    np.random.seed(42)
    vectors = np.random.randn(n_entities, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    entities = [f"entity_{i}" for i in range(n_entities)]
    
    # 生成查询向量
    query_indices = np.random.choice(n_entities, n_queries, replace=False)
    queries = [vectors[i] for i in query_indices]
    
    # 测试不同配置
    configs = [
        {'M': 8, 'ef_construction': 100, 'ef_search': 20, 'name': 'HNSW-Fast'},
        {'M': 16, 'ef_construction': 200, 'ef_search': 50, 'name': 'HNSW-Balanced'},
        {'M': 32, 'ef_construction': 400, 'ef_search': 100, 'name': 'HNSW-Accurate'},
    ]
    
    results = []
    
    # 暴力搜索基准
    print("\n2. 暴力搜索基准...")
    brute_times = []
    brute_results_all = []
    
    for i, query in enumerate(queries):
        start = time.time()
        brute_res = brute_force_search(vectors, query, k=10)
        brute_times.append(time.time() - start)
        brute_results_all.append(brute_res)
    
    brute_avg_time = np.mean(brute_times) * 1000  # ms
    print(f"   平均查询时间: {brute_avg_time:.2f} ms")
    
    results.append({
        'method': 'Brute Force',
        'build_time': 0,
        'query_time': brute_avg_time,
        'recall@1': 1.0,
        'recall@5': 1.0,
        'recall@10': 1.0
    })
    
    # 测试 HNSW
    for config in configs:
        print(f"\n3. 测试 {config['name']}...")
        print(f"   参数: M={config['M']}, ef_construction={config['ef_construction']}, ef_search={config['ef_search']}")
        
        # 构建索引
        start = time.time()
        retriever = HNSWVectorRetriever(
            M=config['M'],
            ef_construction=config['ef_construction'],
            ef_search=config['ef_search']
        )
        retriever.build_index(entities, vectors)
        build_time = time.time() - start
        
        print(f"   构建时间: {build_time:.2f} s")
        print(f"   最大层数: {retriever.index.max_layer}")
        
        # 查询测试
        hnsw_times = []
        hnsw_results_all = []
        
        for query in queries:
            start = time.time()
            hnsw_res = retriever.index.search(query, k=10)
            hnsw_times.append(time.time() - start)
            hnsw_results_all.append(hnsw_res)
        
        hnsw_avg_time = np.mean(hnsw_times) * 1000  # ms
        print(f"   平均查询时间: {hnsw_avg_time:.2f} ms")
        
        # 计算召回率
        recalls_at_1 = []
        recalls_at_5 = []
        recalls_at_10 = []
        
        for hnsw_res, brute_res in zip(hnsw_results_all, brute_results_all):
            recalls_at_1.append(recall_at_k(hnsw_res, brute_res, k=1))
            recalls_at_5.append(recall_at_k(hnsw_res, brute_res, k=5))
            recalls_at_10.append(recall_at_k(hnsw_res, brute_res, k=10))
        
        recall_1 = np.mean(recalls_at_1)
        recall_5 = np.mean(recalls_at_5)
        recall_10 = np.mean(recalls_at_10)
        
        print(f"   召回率@1: {recall_1:.2%}")
        print(f"   召回率@5: {recall_5:.2%}")
        print(f"   召回率@10: {recall_10:.2%}")
        
        # 加速比
        speedup = brute_avg_time / hnsw_avg_time
        print(f"   加速比: {speedup:.1f}x")
        
        results.append({
            'method': config['name'],
            'params': f"M={config['M']},ef={config['ef_search']}",
            'build_time': round(build_time, 2),
            'query_time': round(hnsw_avg_time, 2),
            'recall@1': round(recall_1, 4),
            'recall@5': round(recall_5, 4),
            'recall@10': round(recall_10, 4),
            'speedup': round(speedup, 1)
        })
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("汇总结果")
    print("=" * 60)
    print(f"{'Method':<20} {'Build(s)':<10} {'Query(ms)':<12} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'Speedup':<8}")
    print("-" * 80)
    
    for r in results:
        if 'params' in r:
            method = f"{r['method']} ({r['params']})"
        else:
            method = r['method']
        print(f"{method:<20} {r['build_time']:<10} {r['query_time']:<12} "
              f"{r['recall@1']:<8.2%} {r['recall@5']:<8.2%} {r['recall@10']:<8.2%} "
              f"{r.get('speedup', '-'):<8}")
    
    # 保存结果
    with open('hnsw_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n结果已保存到: hnsw_benchmark_results.json")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='HNSW 性能对比测试')
    parser.add_argument('--n', type=int, default=10000, help='实体数量')
    parser.add_argument('--dim', type=int, default=1024, help='向量维度')
    parser.add_argument('--queries', type=int, default=100, help='查询次数')
    
    args = parser.parse_args()
    
    benchmark(n_entities=args.n, dim=args.dim, n_queries=args.queries)
