"""
MiniGraph API 服务 - LLM + RAG 版本
支持自然语言问答、多跳推理
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from py2neo import Graph
import sys
import os

sys.path.append('/autodl-fs/data/MiniGraph/src')
from llm_client_openai import get_llm_client

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

@app.route('/')
def index():
    """首页"""
    return jsonify({
        'name': 'MiniGraph API (LLM + RAG)',
        'version': '2.0.0',
        'features': [
            '自然语言问答 (LLM + RAG)',
            '实体搜索',
            '实体详情',
            '多跳推理'
        ],
        'endpoints': [
            '/query - 自然语言问答',
            '/entity/<name> - 实体详情',
            '/search?q=xxx - 搜索实体',
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
    
    return jsonify({
        'entity_count': entity_count,
        'relation_count': relation_count,
        'llm_cache_size': cache_size,
        'status': 'running'
    })

@app.route('/search')
def search():
    """搜索实体"""
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': '缺少查询参数 q'}), 400
    
    results = graph.run(
        "MATCH (e:Entity) WHERE e.name CONTAINS $query RETURN e.name as name LIMIT 20",
        query=query
    ).data()
    
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
    自然语言问答 (LLM + RAG)
    
    Request:
        {"question": "什么是社会主义？"}
    
    Response:
        {
            "question": "什么是社会主义？",
            "answer": "...",
            "context": [...],
            "source": "llm_rag"
        }
    """
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': '问题不能为空'}), 400
    
    # 1. 从图数据库检索相关实体
    keywords = question[:10]  # 简单提取关键词
    context_entities = graph.run(
        "MATCH (e:Entity) WHERE e.name CONTAINS $keyword RETURN e.name as name, e.描述 as desc LIMIT 5",
        keyword=keywords
    ).data()
    
    # 2. 构建上下文
    context_text = ""
    for e in context_entities:
        desc = e.get('desc', '')
        if desc:
            context_text += f"- {e['name']}: {desc}\n"
    
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
        'source': 'llm_rag'
    })

@app.route('/chat', methods=['POST'])
def chat():
    """
    直接对话 LLM
    
    Request:
        {"message": "你好", "system": "你是一个知识图谱助手"}
    """
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
    """
    多跳推理查询
    
    Request:
        {"question": "社会主义和资本主义有什么区别？", "max_hops": 3}
    """
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
    
    # Step 2: 执行查询（简化版）
    # 提取第一个实体进行查询
    entities = graph.run(
        "MATCH (e:Entity) WHERE e.name CONTAINS $q RETURN e.name as name LIMIT 3",
        q=question[:5]
    ).data()
    
    if entities:
        entity_name = entities[0]['name']
        relations = graph.run(
            "MATCH (e:Entity {name: $name})-[r]->(v) RETURN type(r) as rel, v.name as target LIMIT 10",
            name=entity_name
        ).data()
        
        context = f"实体：{entity_name}\n关系：" + ", ".join([f"{r['rel']}->{r['target']}" for r in relations[:5]])
        steps.append({'step': f'查询实体：{entity_name}', 'result': context})
    
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
    print("=" * 50)
    print("启动 MiniGraph API (LLM + RAG)")
    print("=" * 50)
    print("访问: http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()
