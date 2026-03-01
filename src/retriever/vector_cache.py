"""
向量检索缓存模块 - 简化版
只缓存 entities 和 vectors，模型正常加载
"""
import os
import pickle
import hashlib
import json

CACHE_DIR = '/autodl-fs/data/MiniGraph/data/cache'

def get_cache_path(index_path):
    """生成缓存文件路径"""
    if not os.path.exists(index_path):
        return None
    index_mtime = os.path.getmtime(index_path)
    index_size = os.path.getsize(index_path)
    cache_key = f"{index_path}_{index_mtime}_{index_size}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"vec_cache_{cache_hash}.pkl")

def save_vector_cache(index_path, entities, vectors, model_name):
    """保存向量缓存"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = get_cache_path(index_path)
    if not cache_path:
        return None
    
    cache_data = {
        'entities': entities,
        'vectors': vectors,
        'model_name': model_name,
        'index_path': index_path
    }
    
    print(f"[缓存] 保存向量缓存: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    # 元信息
    meta_path = cache_path + '.meta'
    with open(meta_path, 'w') as f:
        json.dump({
            'model_name': model_name,
            'entity_count': len(entities),
            'vector_shape': list(vectors.shape) if vectors is not None else None
        }, f, indent=2)
    
    print(f"[缓存] 保存完成: {len(entities)} 个实体")
    return cache_path

def load_vector_cache(index_path):
    """加载向量缓存"""
    cache_path = get_cache_path(index_path)
    if not cache_path or not os.path.exists(cache_path):
        return None
    
    print(f"[缓存] 加载向量缓存: {cache_path}")
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
    
    print(f"[缓存] 加载完成: {len(cache_data.get('entities', []))} 个实体")
    return cache_data

def vector_cache_exists(index_path):
    """检查缓存是否存在"""
    cache_path = get_cache_path(index_path)
    return cache_path and os.path.exists(cache_path)
