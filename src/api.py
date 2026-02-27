"""
查询 API 模块
提供 RESTful API 接口进行知识图谱查询
"""
from flask import Flask, request, jsonify
from typing import Dict, List
import json
import sys
sys.path.append('/autodl-fs/data/MiniGraph/src')

from agents.multi_agent import MultiAgentSystem

app = Flask(__name__)

# 初始化多 Agent 系统
agent_system = MultiAgentSystem()

@app.route('/')
def index():
    """首页"""
    return jsonify({
        'name': 'MiniGraph API',
        'version': '1.0.0',
        'endpoints': [
            '/query - 知识图谱查询',
            '/entity/<name> - 实体详情',
            '/search - 向量搜索'
        ]
    })

@app.route('/query', methods=['POST'])
def query():
    """
    知识图谱查询
    
    Request:
        {
            "question": "周杰伦的妻子是谁？"
        }
    
    Response:
        {
            "response": "昆凌",
            "confidence": 0.9,
            "paths": [...]
        }
    """
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': '问题不能为空'}), 400
    
    try:
        result = agent_system.process(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/entity/<name>', methods=['GET'])
def get_entity(name: str):
    """
    获取实体详情
    
    Response:
        {
            "name": "周杰伦",
            "type": "Person",
            "attributes": {...}
        }
    """
    # TODO: 从 Neo4j 查询实体
    return jsonify({
        'name': name,
        'message': '功能开发中'
    })

@app.route('/search', methods=['POST'])
def vector_search():
    """
    向量相似度搜索
    
    Request:
        {
            "text": "歌手",
            "top_k": 10
        }
    """
    data = request.get_json()
    text = data.get('text', '')
    top_k = data.get('top_k', 10)
    
    # TODO: 向量搜索
    return jsonify({
        'query': text,
        'top_k': top_k,
        'results': [],
        'message': '功能开发中'
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """获取系统统计信息"""
    return jsonify({
        'entities': 0,  # TODO: 从 Neo4j 获取
        'relations': 0,
        'vectors': 0
    })

def main():
    """启动服务"""
    print("启动 MiniGraph API 服务...")
    print("访问: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()
