"""
MiniGraph API 服务 - 优化版
1. 优化的 RAG 提示词模板
2. 实体关系链式检索
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from py2neo import Graph
import sys
import os

sys.path.append('/autodl-fs/data/MiniGraph/src')
from llm_client_openai import get_llm_client
from vector_retriever_chinese import ChineseVectorRetriever

app = Flask(__name__)
CORS(app)

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

# 初始化向量检索
print("初始化向量检索...")
retriever = ChineseVectorRetriever()
vector_index_paths = [
    '/autodl-fs/data/MiniGraph/data/processed/vector_index_bge.json',
    '/root/.openclaw/workspace/MiniGraph/data/processed/vector_index_bge.json',
]

vector_index_loaded = False
for path in vector_index_paths:
    if os.path.exists(path):
        try:
            retriever.load(path)
            print(f"向量索引已加载: {len(retriever.entities)} 个实体")
            vector_index_loaded = True
            break
        except Exception as e:
            print(f"加载 {path} 失败: {e}")

if not vector_index_loaded:
    print("警告: 向量索引不存在")


# ============ 优化的 RAG 提示词模板 ============

RAG_PROMPT_TEMPLATE = """你是一个专业的知识图谱问答助手。请基于提供的知识图谱信息，准确、简洁地回答用户问题。

【知识图谱信息】
{context}

【用户问题】
{question}

【回答要求】
1. 基于上述知识图谱信息回答，不要编造
2. 如果信息不足，明确说明"根据现有信息无法回答"
3. 回答要简洁准确，控制在200字以内
4. 可以适当引用知识图谱中的具体实体和关系

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
            # 查询当前实体的关系
            relations = graph.run("""
                MATCH (e:Entity {name: $name})-[r]->(v:Entity)
                RETURN type(r) as rel, v.name as target, v.desc as target_desc
                LIMIT $limit
            """, name=current, limit=max_branches).data()
            
            for rel in relations:
                target = rel['target']
                if target not in visited:
                    visited.add(target)
                    new_path = path + [current, rel['rel'], target]
                    chains.append({
                        'path': new_path,
                        'depth': depth + 1,
                        'target_desc': rel.get('target_desc', '')
                    })
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
            '/stats - 统计信息'
        ]
    })


@app.route('/stats')
def stats():
    """统计信息"""
    try:
        entity_count = graph.run("MATCH (e:Entity) RETURN count(e) as c").data()[0]['c'] if graph else 0
        relation_count = graph.run("MATCH ()-[r]->() RETURN count(r) as c").data()[0]['c'] if graph else 0
    except:
        entity_count = 0
        relation_count = 0
    
    cache_size = len(llm.cache) if llm else 0
    vector_count = len(retriever.entities) if retriever.entities else 0
    
    return jsonify({
        'entity_count': entity_count,
        'relation_count': relation_count,
        'vector_index_count': vector_count,
        'llm_cache_size': cache_size,
        'embedding_model': retriever.model_name if hasattr(retriever, 'model_name') else 'unknown',
        'status': 'running'
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
    多跳推理查询（优化版）
    """
    if not llm:
        return jsonify({'error': 'LLM 未初始化'}), 500
    
    data = request.get_json()
    question = data.get('question', '')
    max_hops = data.get('max_hops', 3)
    
    if not question:
        return jsonify({'error': '问题不能为空'}), 400
    
    # 1. 向量检索起始实体
    start_entities = []
    if retriever.vectors is not None:
        vector_results = retriever.search(question, top_k=3)
        start_entities = vector_results
    
    # 2. 探索关系链
    paths = []
    for entity_info in start_entities[:2]:  # 只探索前2个实体
        name = entity_info.get('name') or entity_info.get('entity', {}).get('name')
        if name:
            chain_info = get_entity_relations_chain(name, max_depth=max_hops, max_branches=2)
            if chain_info['chains']:
                paths.append({
                    'start_entity': name,
                    'chains': chain_info['chains']
                })
    
    # 3. 构建路径描述
    paths_text = ""
    for i, path_info in enumerate(paths, 1):
        paths_text += f"\n路径{i} (从 {path_info['start_entity']} 开始):\n"
        for chain in path_info['chains'][:3]:  # 最多3条链
            path_str = ' → '.join(chain['path'])
            paths_text += f"  {path_str}\n"
    
    # 4. 使用多跳提示词模板
    prompt = MULTIHOP_PROMPT_TEMPLATE.format(
        paths=paths_text,
        question=question
    )
    
    answer = llm.generate(prompt)
    
    return jsonify({
        'question': question,
        'max_hops': max_hops,
        'paths': paths,
        'answer': answer
    })


def main():
    print("=" * 60)
    print("启动 MiniGraph API (优化版 - RAG + 关系链)")
    print("=" * 60)
    print(f"Embedding 模型: {retriever.model_name if hasattr(retriever, 'model_name') else 'unknown'}")
    print(f"向量索引: {len(retriever.entities)} 个实体")
    print("访问: http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()
