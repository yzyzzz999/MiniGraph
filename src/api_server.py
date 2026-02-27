"""
MiniGraph API 服务
提供知识图谱查询接口
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import sys
sys.path.append('/autodl-fs/data/MiniGraph/src')

app = Flask(__name__)
CORS(app)  # 允许跨域

# 加载数据
print("加载知识图谱数据...")
try:
    with open('/autodl-fs/data/MiniGraph/data/processed/entities.json', 'r', encoding='utf-8') as f:
        entities = json.load(f)
    with open('/autodl-fs/data/MiniGraph/data/processed/relations.json', 'r', encoding='utf-8') as f:
        relations = json.load(f)
    print(f"加载完成: {len(entities)} 实体, {len(relations)} 关系")
except:
    entities = []
    relations = []
    print("数据文件不存在，请先运行数据导入")

# 构建实体索引
entity_dict = {e['name']: e for e in entities}

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
    return jsonify({
        'entity_count': len(entities),
        'relation_count': len(relations),
        'status': 'running'
    })

@app.route('/entity/<name>')
def get_entity(name):
    """获取实体详情"""
    entity = entity_dict.get(name)
    if not entity:
        return jsonify({'error': '实体不存在'}), 404
    
    # 查找相关关系
    entity_relations = [
        r for r in relations 
        if r['from'] == name or r['to'] == name
    ]
    
    return jsonify({
        'entity': entity,
        'relations': entity_relations[:50]  # 最多返回50条
    })

@app.route('/search')
def search():
    """搜索实体"""
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': '缺少查询参数 q'}), 400
    
    # 简单模糊匹配
    results = [
        e for e in entities 
        if query.lower() in e['name'].lower()
    ][:20]  # 最多返回20条
    
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
    
    # 简单规则匹配（后续可接入 Agent）
    response = {
        'question': question,
        'answer': None,
        'reasoning': []
    }
    
    # 提取实体（简单匹配）
    found_entities = []
    for e in entities[:1000]:  # 只检查前1000个
        if e['name'] in question:
            found_entities.append(e)
    
    if found_entities:
        entity = found_entities[0]
        response['reasoning'].append(f"找到实体: {entity['name']}")
        
        # 查找关系
        entity_rels = [r for r in relations if r['from'] == entity['name']]
        
        if '妻子' in question or '配偶' in question:
            spouse_rels = [r for r in entity_rels if '妻子' in r['relation'] or '配偶' in r['relation']]
            if spouse_rels:
                response['answer'] = spouse_rels[0]['to']
                response['reasoning'].append(f"关系: {spouse_rels[0]['relation']}")
        
        response['entity'] = entity
        response['relations'] = entity_rels[:10]
    
    return jsonify(response)

@app.route('/relations/<name>')
def get_relations(name):
    """获取实体的所有关系"""
    entity_rels = [
        r for r in relations 
        if r['from'] == name or r['to'] == name
    ]
    
    return jsonify({
        'entity': name,
        'count': len(entity_rels),
        'relations': entity_rels
    })

def main():
    print("启动 MiniGraph API...")
    print("访问: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()
