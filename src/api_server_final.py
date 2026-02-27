"""
MiniGraph API 服务 - LLM + RAG + 向量检索版本
支持自然语言问答、多跳推理、向量相似度搜索
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from py2neo import Graph
import sys
import os

sys.path.append('/autodl-fs/data/MiniGraph/src')
from llm_client_openai import get_llm_client
from vector_retriever import VectorRetriever

app = Flask(__name__)
CORS(app)

# 连接 Neo4j
print("连接 Neo4j...")
graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
print("Neo4j 连接成功!")

# 初始化 LLM
print("初始化 LLM...")
llm = get_llm_client()
print("LLM 初始化完成!")

# 初始化向量检索
print("初始化向量检索...")
retriever = VectorRetriever()
vector_index_path = '/autodl-fs/data/MiniGraph/data/processed/vector_index.json'
if os.path.exists(vector_index_path):
    retriever.load(vector_index_path)
    print(f"向量索引已加载: {len(retriever.entities)} 个实体")
else:
    print("警告: 向量索引不存在，请先运行 vector_retriever.py")

@app.route('/')
def index():
    """首页"""
    return jsonify({
        'name': 'MiniGraph API (LLM + RAG + 向量检索)',
        'version': '3.0.0',
        'features': [
            '自然语言问答 (LLM + RAG + 向量检索)',
            '实体搜索 (关键词 + 向量)',
            '实体详情',
            '多跳推理'
        ],
        'endpoints': [
            '/query - 自然语言问答',
            '/entity/<name> - 实体详情',
            '/search?q=xxx - 搜索实体',
            '/vector_search?q=xxx - 向量相似度搜索',
            '/stats - 统计信息',
            '/chat - 直接对话 LLM'
        ]
    })

@app.route('/stats')
def stats():
    """统计信息"""
    entity_count = graph.run("MATCH (e:Entity) RETURN count(e) as c").data()[0]['c']
    relation_count = graph.run("MATCH ()-[r]->() RETURN count(r) as c").data()[0]['c']
    cache_size = len(llm.cache)
    vector_count = len(retriever.entities) if retriever.entities else 0
    
    return jsonify({
        'entity_count': entity_count,
        'relation_count': relation_count,
        'vector_index_count': vector_count,
        'llm_cache_size': cache_size,
        'status': 'running'
    })

@app.route('/search')
def search():
    """搜索实体（关键词 + 向量）"""
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': '缺少查询参数 q'}), 400
    
    # 1. 关键词搜索
    keyword_results = graph.run(
        "MATCH (e:Entity) WHERE e.name CONTAINS $query RETURN e.name as name LIMIT 10",
        query=query
    ).data()
    
    # 2. 向量搜索
    vector_results = []
    if retriever.vectors is not None:
        vector_results = retriever.search(query, top_k=10)
    
    # 合并结果（去重）
    seen = set()
    combined = []
    
    for r in keyword_results:
        name = r['name']
        if name not in seen:
            seen.add(name)
            combined.append({'name': name, 'source': 'keyword'})
    
    for r in vector_results:
        name = r['name']
        if name not in seen:
            seen.add(name)
            combined.append({'name': name, 'source': 'vector', 'similarity': r['similarity']})
    
    return jsonify({
        'query': query,
        'count': len(combined),
        'results': combined[:20]
    })

@app.route('/vector_search')
def vector_search():
    """纯向量相似度搜索"""
    query = request.args.get('q', '')
    top_k = int(request.args.get('k', 5))
    
    if not query:
        return jsonify({'error': '缺少查询参数 q'}), 400
    
    if retriever.vectors is None:
        return jsonify({'error': '向量索引未加载'}), 500
    
    results = retriever.search(query, top_k=top_k)
    
    return jsonify({
        'query': query,
        'count': len(results),
        'results': results
    })

@app.route('/entity/<name>')
def get_entity(name):
    """获取实体详情"""
    result = graph.run(
        "MATCH (e:Entity {name: $name}) RETURN e",
        name=name
    ).data()
    
    if not result:
        return jsonify({'error': '实体不存在'}), 404
    
    entity = dict(result[0]['e'])
    
    # 查询关系
    relations = graph.run(
        "MATCH (e:Entity {name: $name})-[r]->(v) RETURN type(r) as rel, v.name as value LIMIT 50",
        name=name
    ).data()
    
    return jsonify({
        'entity': entity,
        'relations': relations
    })

@app.route('/query', methods=['POST'])
def query():
    """
    自然语言问答 (LLM + RAG + 向量检索)
    """
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': '问题不能为空'}), 400
    
    # 1. 向量检索相关实体
    context_entities = []
    if retriever.vectors is not None:
        vector_results = retriever.search(question, top_k=5)
        for r in vector_results:
            # 获取实体详情
            entity_data = graph.run(
                "MATCH (e:Entity {name: $name}) RETURN e.name as name, e.描述 as desc LIMIT 1",
                name=r['name']
            ).data()
            if entity_data:
                context_entities.append(entity_data[0])
    
    # 2. 构建上下文
    context_text = ""
    for e in context_entities:
        desc = e.get('desc', '')
        name = e.get('name', '')
        if desc:
            context_text += f"- {name}: {desc}\n"
        else:
            context_text += f"- {name}\n"
    
    # 3. 用 LLM 生成回答
    prompt = f"""基于以下知识图谱信息回答问题：

知识图谱信息：
{context_text}

用户问题：{question}

请根据知识图谱信息回答问题。如果信息不足，请说明。"""

    answer = llm.generate(prompt, use_cache=True)
    
    return jsonify({
        'question': question,
        'answer': answer,
        'context': context_entities,
        'source': 'llm_rag_vector'
    })

@app.route('/chat', methods=['POST'])
def chat():
    """直接对话 LLM"""
    data = request.get_json()
    message = data.get('message', '')
    system = data.get('system', '你是一个知识图谱助手')
    
    if not message:
        return jsonify({'error': '消息不能为空'}), 400
    
    response = llm.generate(message, system=system)
    
    return jsonify({
        'message': message,
        'response': response
    })

@app.route('/multihop', methods=['POST'])
def multihop():
    """多跳推理查询"""
    data = request.get_json()
    question = data.get('question', '')
    max_hops = data.get('max_hops', 3)
    
    if not question:
        return jsonify({'error': '问题不能为空'}), 400
    
    # 多跳推理步骤
    steps = []
    
    # Step 1: 规划查询
    plan_prompt = f"""分析以下问题，规划查询步骤（最多{max_hops}步）：

问题：{question}

请输出查询步骤，格式：
1. 查询[实体A]
2. 查找[实体A]的[关系]
3. ..."""

    plan = llm.generate(plan_prompt)
    steps.append({'step': '规划', 'result': plan})
    
    # Step 2: 向量检索起始实体
    if retriever.vectors is not None:
        vector_results = retriever.search(question, top_k=3)
        if vector_results:
            entity_name = vector_results[0]['name']
            relations = graph.run(
                "MATCH (e:Entity {name: $name})-[r]->(v) RETURN type(r) as rel, v.name as target LIMIT 10",
                name=entity_name
            ).data()
            
            context = f"实体：{entity_name}\n关系：" + ", ".join([f"{r['rel']}->{r['target']}" for r in relations[:5]])
            steps.append({'step': f'向量检索实体：{entity_name}', 'result': context})
    
    # Step 3: 综合回答
    final_prompt = f"""基于以下查询步骤回答问题：

问题：{question}

查询步骤：
"""
    for s in steps:
        final_prompt += f"\n{s['step']}:\n{s['result']}\n"
    
    final_prompt += f"\n请综合以上信息，回答：{question}"
    
    answer = llm.generate(final_prompt)
    steps.append({'step': '综合回答', 'result': answer})
    
    return jsonify({
        'question': question,
        'max_hops': max_hops,
        'steps': steps,
        'answer': answer
    })

def main():
    print("=" * 60)
    print("启动 MiniGraph API (LLM + RAG + 向量检索)")
    print("=" * 60)
    print("访问: http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()
