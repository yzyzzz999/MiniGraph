"""
MiniGraph API 服务 - 优化版
1. 优化的 RAG 提示词模板
2. 实体关系链式检索
3. 向量索引缓存加速
4. 异步处理支持
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from py2neo import Graph
import sys
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.append('/autodl-fs/data/MiniGraph/src')
from llm.llm_client_openai import get_llm_client
from retriever.vector_retriever_chinese import ChineseVectorRetriever
from retriever.vector_cache import vector_cache_exists, load_vector_cache, save_vector_cache

app = Flask(__name__)
CORS(app)

# ============ API 监控 ============

# 请求计数器
request_count = 0
request_count_lock = Lock()

@app.before_request
def before_request():
    """请求开始时间"""
    request.start_time = time.time()
    global request_count
    with request_count_lock:
        request_count += 1

@app.route('/health', endpoint='health_check')
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'services': {
            'neo4j': graph is not None,
            'llm': llm is not None,
            'vector_index': retriever.vectors is not None
        }
    })

@app.route('/metrics', endpoint='api_metrics')
def api_metrics():
    """API 指标接口"""
    global request_count
    return jsonify({
        'total_requests': request_count,
        'vector_index_size': len(retriever.entities) if retriever.entities else 0,
        'llm_cache_size': len(llm.cache) if llm else 0,
        'neo4j_connected': graph is not None
    })

# 线程池用于异步任务
executor = ThreadPoolExecutor(max_workers=4)

# 异步任务结果存储（内存中，重启丢失）
task_results = {}
task_lock = Lock()

def store_task_result(task_id, result):
    """存储任务结果"""
    with task_lock:
        task_results[task_id] = {
            'status': 'completed',
            'result': result,
            'timestamp': time.time()
        }

def get_task_result(task_id):
    """获取任务结果"""
    with task_lock:
        return task_results.get(task_id)

# 连接 Neo4j
print("连接 Neo4j...")
try:
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
    print("Neo4j 连接成功!")
except Exception as e:
    print(f"Neo4j 连接失败: {e}")
    graph = None

# 初始化 LLM
print("初始化 LLM...")
try:
    llm = get_llm_client()
    print("LLM 初始化完成!")
except Exception as e:
    print(f"LLM 初始化失败: {e}")
    llm = None

# 初始化向量检索（带缓存加速）
print("初始化向量检索...")

retriever = ChineseVectorRetriever()
vector_index_paths = [
    '/autodl-fs/data/MiniGraph/data/processed/vector_index_bge_full.json',
    '/root/.openclaw/workspace/MiniGraph/data/processed/vector_index_bge_full.json',
]

vector_index_loaded = False
index_path = None
for path in vector_index_paths:
    if os.path.exists(path):
        index_path = path
        break

if index_path:
    model_name = "BAAI/bge-large-zh"
    try:
        # 尝试加载向量缓存
        if vector_cache_exists(index_path):
            cache_data = load_vector_cache(index_path)
            if cache_data:
                retriever.entities = cache_data['entities']
                retriever.vectors = cache_data['vectors']
                retriever.model_name = cache_data['model_name']
                print(f"向量数据缓存加载完成: {len(retriever.entities)} 个实体")
                vector_index_loaded = True
        
        # 没有缓存，从 JSON 加载
        if not vector_index_loaded:
            print("从 JSON 加载向量索引...")
            retriever.load(index_path)
            print(f"向量索引已加载: {len(retriever.entities)} 个实体")
            # 保存向量缓存供下次使用
            save_vector_cache(index_path, retriever.entities, 
                            retriever.vectors, retriever.model_name)
            vector_index_loaded = True
            
    except Exception as e:
        print(f"加载失败: {e}")
        try:
            retriever.load(index_path)
            print(f"向量索引已加载: {len(retriever.entities)} 个实体")
            vector_index_loaded = True
        except Exception as e2:
            print(f"回退加载也失败: {e2}")

if not vector_index_loaded:
    print("警告: 向量索引不存在")


# ============ 优化的 RAG 提示词模板 ============

RAG_PROMPT_TEMPLATE = """你是一个知识图谱问答助手。请基于提供的知识图谱信息回答问题。

【知识图谱信息】
{context}

【用户问题】
{question}

【回答规则】
1. **优先使用**知识图谱中的信息回答
2. **关系推理**：如果图谱中有相关实体但没有直接关系，可以基于常识进行合理推理，但必须明确标注
3. **明确标注**信息来源：
   - 来自知识图谱：保留相关度分数，引用具体实体
   - 来自推理/常识补充：用"（补充/推理）"标注
4. **禁止编造**知识图谱中没有的具体细节（如精确年份、数字等）
5. 回答控制在200字以内，信息丰富但不过度推测

【示例1-事实题】
知识图谱信息：
- 唐朝[中国历史朝代]（相关度0.70）

问题：唐朝的开国皇帝是谁？
回答：唐朝的开国皇帝是李渊（知识图谱中虽未直接列出李渊，但结合历史常识补充）。

【示例2-关系推理题】
知识图谱信息：
- 李白[唐代诗人]（相关度0.80）
- 杜甫[唐代诗人]（相关度0.78）
- 唐朝[中国历史朝代]（相关度0.75）

问题：李白和杜甫有什么关系？
回答：李白和杜甫都是唐朝著名诗人（知识图谱，相关度0.80/0.78），二人生活在同一时代（推理），被合称"李杜"。

请回答："""

MULTIHOP_PROMPT_TEMPLATE = """你是一个知识图谱推理专家。请基于多跳查询结果，分析并回答用户问题。

【查询路径】
{paths}

【用户问题】
{question}

【推理要求】
1. 分析查询路径中的实体关系链
2. 找出与问题相关的关键信息
3. 给出逻辑清晰的答案
4. 如果路径不足以回答问题，说明需要更多信息

请分析并回答："""


# ============ 实体关系链式检索 ============

def get_entity_relations_chain(entity_name: str, max_depth: int = 2, max_branches: int = 3) -> dict:
    """
    获取实体的关系链（多跳探索）
    
    Args:
        entity_name: 起始实体名称
        max_depth: 最大探索深度
        max_branches: 每层最大分支数
        
    Returns:
        {
            'entity': entity_name,
            'chains': [
                {'path': ['A', 'rel1', 'B', 'rel2', 'C'], 'depth': 2},
                ...
            ]
        }
    """
    if not graph:
        return {'entity': entity_name, 'chains': []}
    
    chains = []
    visited = {entity_name}
    
    # BFS 探索关系链
    queue = [(entity_name, [], 0)]  # (当前实体, 路径, 深度)
    
    while queue:
        current, path, depth = queue.pop(0)
        
        if depth >= max_depth:
            continue
        
        try:
            # 查询当前实体的关系（支持 Value 类型节点）
            relations = graph.run("""
                MATCH (e:Entity {name: $name})-[r]->(v)
                RETURN type(r) as rel, v.name as target, v.desc as target_desc
                LIMIT $limit
            """, name=current, limit=max_branches).data()
            
            for rel in relations:
                target = rel['target']
                if target and target not in visited:
                    visited.add(target)
                    new_path = path + [current, rel['rel'], target]
                    chains.append({
                        'path': new_path,
                        'depth': depth + 1,
                        'target_desc': rel.get('target_desc', '')
                    })
                    # 只继续探索 Entity 类型的节点
                    if target:
                        queue.append((target, new_path, depth + 1))
                    
        except Exception as e:
            print(f"查询 {current} 的关系失败: {e}")
    
    return {'entity': entity_name, 'chains': chains}


def get_entity_with_relations(entity_name: str, relation_limit: int = 10) -> dict:
    """
    获取实体及其关系（增强版）
    
    Returns:
        {
            'name': entity_name,
            'desc': description,
            'relations': [{'rel': '关系名', 'target': '目标实体', 'target_desc': '描述'}, ...],
            'reverse_relations': [{'rel': '关系名', 'source': '源实体'}, ...]
        }
    """
    if not graph:
        return {'name': entity_name}
    
    try:
        # 实体基本信息
        entity_info = graph.run("""
            MATCH (e:Entity {name: $name})
            RETURN e.name as name, e.desc as desc, e.category as category
        """, name=entity_name).data()
        
        if not entity_info:
            return {'name': entity_name}
        
        entity = entity_info[0]
        
        # 出边关系
        relations = graph.run("""
            MATCH (e:Entity {name: $name})-[r]->(v:Entity)
            RETURN type(r) as rel, v.name as target, v.desc as target_desc
            LIMIT $limit
        """, name=entity_name, limit=relation_limit).data()
        
        # 入边关系
        reverse_relations = graph.run("""
            MATCH (v:Entity)-[r]->(e:Entity {name: $name})
            RETURN type(r) as rel, v.name as source
            LIMIT $limit
        """, name=entity_name, limit=relation_limit).data()
        
        return {
            'name': entity['name'],
            'desc': entity.get('desc', ''),
            'category': entity.get('category', ''),
            'relations': relations,
            'reverse_relations': reverse_relations
        }
        
    except Exception as e:
        print(f"获取实体 {entity_name} 信息失败: {e}")
        return {'name': entity_name}


def build_enhanced_context(entities: list, use_chain: bool = True) -> str:
    """
    构建增强的上下文（包含关系链）
    
    Args:
        entities: 实体列表 [{'name': '...', 'similarity': 0.9}, ...]
        use_chain: 是否使用关系链
        
    Returns:
        格式化的上下文文本
    """
    context_parts = []
    
    for i, entity_info in enumerate(entities[:5], 1):  # 最多5个实体
        name = entity_info.get('name') or entity_info.get('entity', {}).get('name')
        similarity = entity_info.get('similarity', 0)
        
        if not name:
            continue
        
        # 获取实体详细信息
        entity_detail = get_entity_with_relations(name, relation_limit=5)
        
        part = f"\n【实体{i}】{name}"
        if similarity > 0:
            part += f" (相关度: {similarity:.2f})"
        
        if entity_detail.get('desc'):
            part += f"\n描述: {entity_detail['desc'][:100]}"
        
        if entity_detail.get('category'):
            part += f"\n类别: {entity_detail['category']}"
        
        # 添加直接关系
        if entity_detail.get('relations'):
            part += "\n关系:"
            for rel in entity_detail['relations'][:3]:
                part += f"\n  - {rel['rel']} → {rel['target']}"
        
        context_parts.append(part)
        
        # 添加关系链（如果启用）
        if use_chain and i <= 2:  # 只对前2个实体添加关系链
            chain_info = get_entity_relations_chain(name, max_depth=2, max_branches=2)
            if chain_info['chains']:
                context_parts.append(f"  关系链:")
                for chain in chain_info['chains'][:2]:
                    path_str = ' → '.join(chain['path'])
                    context_parts.append(f"    {path_str}")
    
    return '\n'.join(context_parts)


# ============ API 路由 ============

@app.route('/')
def index():
    """首页"""
    return jsonify({
        'name': 'MiniGraph API (优化版 - RAG + 关系链)',
        'version': '3.1.0',
        'features': [
            '优化的 RAG 提示词模板',
            '实体关系链式检索',
            '增强的上下文构建',
            '中文向量检索 (BGE-large-zh)'
        ],
        'endpoints': [
            '/query - 优化的自然语言问答',
            '/query_enhanced - 增强版问答（含关系链）',
            '/entity_chain/<name> - 获取实体关系链',
            '/entity/<name> - 实体详情（增强版）',
            '/search?q=xxx - 搜索实体',
            '/vector_search?q=xxx - 向量相似度搜索',
        ]
    })


@app.route('/search')
def search():
    """搜索实体（关键词 + 向量混合）"""
    query = request.args.get('q', '')
    top_k = int(request.args.get('k', 10))
    
    if not query:
        return jsonify({'error': '缺少查询参数 q'}), 400
    
    results = []
    seen = set()
    
    # 1. 向量搜索（优先）
    if retriever.vectors is not None:
        vector_results = retriever.search(query, top_k=top_k)
        for r in vector_results:
            name = r['name']
            if name not in seen:
                seen.add(name)
                results.append({
                    'name': name,
                    'source': 'vector',
                    'similarity': r['similarity'],
                    'entity': r.get('entity', {})
                })
    
    # 2. 关键词搜索（补充）
    if graph:
        try:
            keyword_results = graph.run(
                "MATCH (e:Entity) WHERE e.name CONTAINS $query RETURN e.name as name LIMIT $limit",
                query=query, limit=top_k
            ).data()
            
            for r in keyword_results:
                name = r['name']
                if name not in seen:
                    seen.add(name)
                    results.append({'name': name, 'source': 'keyword'})
        except Exception as e:
            print(f"关键词搜索失败: {e}")
    
    return jsonify({
        'query': query,
        'count': len(results),
        'results': results
    })


@app.route('/vector_search')
def vector_search():
    """纯向量相似度搜索"""
    query = request.args.get('q', '')
    top_k = int(request.args.get('k', 5))
    threshold = float(request.args.get('threshold', 0.0))
    
    if not query:
        return jsonify({'error': '缺少查询参数 q'}), 400
    
    if retriever.vectors is None:
        return jsonify({'error': '向量索引未加载'}), 500
    
    results = retriever.search(query, top_k=top_k, threshold=threshold)
    
    return jsonify({
        'query': query,
        'model': retriever.model_name if hasattr(retriever, 'model_name') else 'unknown',
        'count': len(results),
        'results': results
    })


@app.route('/entity/<name>')
def get_entity(name):
    """获取实体详情（增强版）"""
    entity_detail = get_entity_with_relations(name, relation_limit=20)
    
    if not entity_detail.get('desc') and not entity_detail.get('relations'):
        return jsonify({'error': '实体不存在或无法获取信息'}), 404
    
    return jsonify(entity_detail)


@app.route('/entity_chain/<name>')
def get_entity_chain(name):
    """获取实体关系链"""
    max_depth = int(request.args.get('depth', 2))
    max_branches = int(request.args.get('branches', 3))
    
    chain_info = get_entity_relations_chain(name, max_depth=max_depth, max_branches=max_branches)
    
    return jsonify(chain_info)


@app.route('/batch/entities', methods=['POST'])
def batch_get_entities():
    """批量获取实体详情
    
    Request Body:
    {
        "names": ["北京大学", "唐朝", "人工智能"],
        "include_relations": true
    }
    """
    data = request.get_json()
    if not data or 'names' not in data:
        return jsonify({'error': '缺少 names 参数'}), 400
    
    names = data['names']
    include_relations = data.get('include_relations', True)
    
    if not isinstance(names, list):
        return jsonify({'error': 'names 必须是数组'}), 400
    
    results = []
    errors = []
    
    for name in names:
        try:
            if include_relations:
                entity_detail = get_entity_with_relations(name, relation_limit=10)
            else:
                # 只获取基本信息
                entity_detail = {'name': name}
                if graph:
                    result = graph.run("""
                        MATCH (e:Entity {name: $name})
                        RETURN e.desc as desc, e.category as category
                    """, name=name).data()
                    if result:
                        entity_detail['desc'] = result[0].get('desc')
                        entity_detail['category'] = result[0].get('category')
            
            results.append(entity_detail)
        except Exception as e:
            errors.append({'name': name, 'error': str(e)})
    
    return jsonify({
        'count': len(results),
        'results': results,
        'errors': errors if errors else None
    })


@app.route('/batch/vector_search', methods=['POST'])
def batch_vector_search():
    """批量向量相似度搜索
    
    Request Body:
    {
        "queries": ["人工智能", "唐朝", "北京大学"],
        "top_k": 3,
        "threshold": 0.5
    }
    """
    data = request.get_json()
    if not data or 'queries' not in data:
        return jsonify({'error': '缺少 queries 参数'}), 400
    
    queries = data['queries']
    top_k = int(data.get('top_k', 5))
    threshold = float(data.get('threshold', 0.0))
    
    if not isinstance(queries, list):
        return jsonify({'error': 'queries 必须是数组'}), 400
    
    if retriever.vectors is None:
        return jsonify({'error': '向量索引未加载'}), 500
    
    results = []
    for query in queries:
        try:
            search_results = retriever.search(query, top_k=top_k, threshold=threshold)
            results.append({
                'query': query,
                'count': len(search_results),
                'results': search_results
            })
        except Exception as e:
            results.append({
                'query': query,
                'error': str(e)
            })
    
    return jsonify({
        'count': len(queries),
        'model': retriever.model_name if hasattr(retriever, 'model_name') else 'unknown',
        'results': results
    })


@app.route('/query', methods=['POST'])
def query():
    """
    自然语言问答（优化的 RAG）
    """
    if not llm:
        return jsonify({'error': 'LLM 未初始化'}), 500
    
    data = request.get_json()
    question = data.get('question', '')
    use_enhanced = data.get('enhanced', False)
    
    if not question:
        return jsonify({'error': '问题不能为空'}), 400
    
    # 1. 向量检索相关实体
    context_entities = []
    if retriever.vectors is not None:
        vector_results = retriever.search(question, top_k=5)
        context_entities = vector_results
    
    # 2. 构建上下文
    if use_enhanced:
        context_text = build_enhanced_context(context_entities, use_chain=True)
    else:
        context_text = build_enhanced_context(context_entities, use_chain=False)
    
    # 3. 使用优化的提示词模板生成回答
    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context_text,
        question=question
    )
    
    answer = llm.generate(prompt, use_cache=True)
    
    return jsonify({
        'question': question,
        'answer': answer,
        'context': context_entities,
        'source': 'llm_rag_enhanced' if use_enhanced else 'llm_rag',
        'model': retriever.model_name if hasattr(retriever, 'model_name') else 'unknown'
    })


@app.route('/query_enhanced', methods=['POST'])
def query_enhanced():
    """
    增强版自然语言问答（自动启用关系链）
    """
    data = request.get_json()
    data['enhanced'] = True
    return query()


@app.route('/multihop', methods=['POST'])
def multihop():
    """
    多跳推理查询（优化版 - 速度优化）
    """
    import time
    start_time = time.time()
    
    if not llm:
        return jsonify({'error': 'LLM 未初始化'}), 500
    
    data = request.get_json()
    question = data.get('question', '')
    max_hops = data.get('max_hops', 2)  # 默认2跳，减少探索时间
    
    if not question:
        return jsonify({'error': '问题不能为空'}), 400
    
    # 1. 向量检索起始实体
    start_entities = []
    if retriever.vectors is not None:
        vector_results = retriever.search(question, top_k=2)  # 减少检索数量
        start_entities = vector_results
    
    # 2. 探索关系链
    paths = []
    for entity_info in start_entities[:1]:  # 只探索第1个实体，减少时间
        name = entity_info.get('name') or entity_info.get('entity', {}).get('name')
        if name:
            chain_info = get_entity_relations_chain(name, max_depth=max_hops, max_branches=2)
            if chain_info['chains']:
                paths.append({
                    'start_entity': name,
                    'chains': chain_info['chains'][:3]  # 最多3条链
                })
    
    # 3. 如果没有找到路径，直接返回提示，不调用 LLM
    if not paths:
        elapsed = time.time() - start_time
        return jsonify({
            'question': question,
            'max_hops': max_hops,
            'paths': [],
            'answer': '未找到相关关系链，无法回答。请尝试其他问题或增加知识图谱数据。',
            'time_ms': round(elapsed * 1000, 2)
        })
    
    # 4. 构建路径描述（简化版）
    paths_text = ""
    for i, path_info in enumerate(paths, 1):
        paths_text += f"\n路径{i} ({path_info['start_entity']}):\n"
        for chain in path_info['chains']:
            path_str = ' → '.join(chain['path'])
            paths_text += f"  {path_str}\n"
    
    # 5. 使用简化的提示词模板
    simple_prompt = f"""基于以下知识图谱路径回答问题：

{paths_text}

问题：{question}

请简洁回答（100字以内）："""
    
    # 6. 调用 LLM（添加超时保护）
    try:
        answer = llm.generate(simple_prompt)
        # 限制回答长度
        if len(answer) > 300:
            answer = answer[:300] + "..."
    except Exception as e:
        answer = f"推理失败: {str(e)}"
    
    elapsed = time.time() - start_time
    
    return jsonify({
        'question': question,
        'max_hops': max_hops,
        'paths': paths,
        'answer': answer,
        'time_ms': round(elapsed * 1000, 2)
    })


@app.route('/entity_linking', methods=['POST'])
def entity_linking():
    """
    实体链接/消歧接口
    
    Request Body:
    {
        "mention": "苹果",
        "context": "苹果公司发布了新款iPhone",
        "top_k": 3
    }
    """
    data = request.get_json()
    mention = data.get('mention', '')
    context = data.get('context', '')
    top_k = int(data.get('top_k', 5))
    
    if not mention:
        return jsonify({'error': '缺少 mention 参数'}), 400
    
    if retriever.vectors is None:
        return jsonify({'error': '向量索引未加载'}), 500
    
    # 1. 向量检索候选实体
    candidates = retriever.search(mention, top_k=top_k * 2)  # 检索更多候选
    
    if not candidates:
        return jsonify({
            'mention': mention,
            'context': context,
            'candidates': [],
            'message': '未找到候选实体'
        })
    
    # 2. 如果有上下文，计算上下文相关性
    results = []
    for cand in candidates:
        entity_name = cand.get('name') or cand.get('entity', {}).get('name')
        similarity = cand.get('similarity', 0)
        
        # 基础分数：向量相似度
        score = similarity
        
        # 如果有上下文，计算上下文相关性
        context_score = 0
        if context and entity_name:
            # 简单规则：实体名或描述在上下文中出现则加分
            entity_lower = entity_name.lower()
            context_lower = context.lower()
            
            # 完全匹配
            if mention.lower() in entity_lower or entity_lower in mention.lower():
                context_score += 0.1
            
            # 获取实体详情
            try:
                if graph:
                    entity_info = graph.run("""
                        MATCH (e:Entity {name: $name})
                        RETURN e.desc as desc, e.category as category
                    """, name=entity_name).data()
                    
                    if entity_info:
                        desc = entity_info[0].get('desc', '') or ''
                        category = entity_info[0].get('category', '') or ''
                        
                        # 描述或类别在上下文中出现
                        if desc and any(word in context_lower for word in desc.lower().split()[:5]):
                            context_score += 0.05
                        if category and category.lower() in context_lower:
                            context_score += 0.08
            except:
                pass
        
        final_score = min(score + context_score, 1.0)  # 最高1.0
        
        results.append({
            'entity': entity_name,
            'score': round(final_score, 4),
            'vector_similarity': round(similarity, 4),
            'context_bonus': round(context_score, 4),
            'description': cand.get('entity', {}).get('desc', '')
        })
    
    # 按分数排序
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return jsonify({
        'mention': mention,
        'context': context,
        'candidates': results[:top_k],
        'best_match': results[0] if results else None
    })


# ============ 异步处理接口 ============

@app.route('/async/query', methods=['POST'])
def async_query():
    """
    异步查询接口 - 提交任务并立即返回任务ID
    
    Request Body: 同 /query 接口
    Response: {"task_id": "xxx", "status": "pending"}
    """
    data = request.get_json()
    task_id = str(uuid.uuid4())
    
    # 存储初始状态
    with task_lock:
        task_results[task_id] = {
            'status': 'pending',
            'created_at': time.time()
        }
    
    # 在线程池中执行查询
    def run_query():
        try:
            # 模拟同步调用
            result = process_query_sync(data)
            store_task_result(task_id, result)
        except Exception as e:
            with task_lock:
                task_results[task_id] = {
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': time.time()
                }
    
    executor.submit(run_query)
    
    return jsonify({
        'task_id': task_id,
        'status': 'pending',
        'check_url': f'/async/result/{task_id}'
    })


@app.route('/async/result/<task_id>', methods=['GET'])
def get_async_result(task_id):
    """获取异步任务结果"""
    result = get_task_result(task_id)
    
    if not result:
        return jsonify({'error': '任务不存在'}), 404
    
    return jsonify(result)


def process_query_sync(data):
    """同步处理查询（用于线程池）"""
    question = data.get('question', '')
    use_enhanced = data.get('enhanced', False)
    
    if not question:
        return {'error': '问题不能为空'}
    
    if not llm:
        return {'error': 'LLM 未初始化'}
    
    # 1. 向量检索相关实体
    context_entities = []
    if retriever.vectors is not None:
        vector_results = retriever.search(question, top_k=5)
        context_entities = vector_results
    
    # 2. 构建上下文（使用正确的函数）
    if use_enhanced:
        context_text = build_enhanced_context(context_entities, use_chain=True)
    else:
        context_text = build_enhanced_context(context_entities, use_chain=False)
    
    # 3. 生成回答
    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context_text,
        question=question
    )
    
    answer = llm.generate(prompt, use_cache=True)
    
    return {
        'question': question,
        'answer': answer,
        'context': context_entities,
        'source': 'llm_rag_enhanced' if use_enhanced else 'llm_rag'
    }


# 配置静态文件目录（用于前端界面）
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'web')
if os.path.exists(static_dir):
    app.static_folder = static_dir
    print(f"静态文件目录: {static_dir}")

@app.route('/web')
def web_interface():
    """前端界面"""
    return app.send_static_file('index.html')


def main():
    print("=" * 60)
    print("启动 MiniGraph API (优化版 - RAG + 关系链)")
    print("=" * 60)
    print(f"Embedding 模型: {retriever.model_name if hasattr(retriever, 'model_name') else 'unknown'}")
    print(f"向量索引: {len(retriever.entities)} 个实体")
    print("访问: http://localhost:5000")
    print("前端界面: http://localhost:5000/web")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()
