"""
MiniGraph API 服务 - Neo4j 版本
直接查询 Neo4j 数据库
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from py2neo import Graph
import sys

app = Flask(__name__)
CORS(app)

# 连接 Neo4j
print("连接 Neo4j...")
graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
print("连接成功!")

@app.route('/')
def index():
    """首页"""
    return jsonify({
        'name': 'MiniGraph API',
        'version': '1.0.0',
        'endpoints': [
            '/query - 自然语言查询',
            '/entity/<name> - 实体详情',
            '/search?q=xxx - 搜索实体',
            '/stats - 统计信息'
        ]
    })

@app.route('/stats')
def stats():
    """统计信息"""
    entity_count = graph.run("MATCH (e:Entity) RETURN count(e) as c").data()[0]['c']
    relation_count = graph.run("MATCH ()-[r]->() RETURN count(r) as c").data()[0]['c']
    
    return jsonify({
        'entity_count': entity_count,
        'relation_count': relation_count,
        'status': 'running'
    })

@app.route('/entity/<name>')
def get_entity(name):
    """获取实体详情"""
    # 查询实体
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

@app.route('/search')
def search():
    """搜索实体"""
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': '缺少查询参数 q'}), 400
    
    # 模糊搜索
    results = graph.run(
        "MATCH (e:Entity) WHERE e.name CONTAINS $query RETURN e.name as name LIMIT 20",
        query=query
    ).data()
    
    return jsonify({
        'query': query,
        'count': len(results),
        'results': results
    })

@app.route('/query', methods=['POST'])
def query():
    """
    自然语言查询
    示例: {"question": "周杰伦的妻子是谁？"}
    """
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': '问题不能为空'}), 400
    
    response = {
        'question': question,
        'answer': None,
        'reasoning': []
    }
    
    # 简单提取实体（匹配问题中的实体名）
    # 先获取所有实体名进行匹配
    entities_in_db = graph.run(
        "MATCH (e:Entity) RETURN e.name as name LIMIT 1000"
    ).data()
    
    found_entity = None
    for e in entities_in_db:
        if e['name'] in question:
            found_entity = e['name']
            break
    
    if found_entity:
        response['reasoning'].append(f"找到实体: {found_entity}")
        
        # 查找关系
        relations = graph.run(
            "MATCH (e:Entity {name: $name})-[r]->(v) RETURN type(r) as rel, v.name as value",
            name=found_entity
        ).data()
        
        # 根据问题关键词匹配关系
        if '妻子' in question or '配偶' in question:
            spouse = [r for r in relations if '妻子' in r['rel'] or '配偶' in r['rel']]
            if spouse:
                response['answer'] = spouse[0]['value']
                response['reasoning'].append(f"关系: {spouse[0]['rel']}")
        
        response['entity'] = found_entity
        response['relations'] = relations[:10]
    else:
        response['reasoning'].append("未找到匹配实体")
    
    return jsonify(response)

def main():
    print("启动 MiniGraph API (Neo4j 版本)...")
    print("访问: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()
