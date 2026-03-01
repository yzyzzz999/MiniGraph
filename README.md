# MiniGraph - 中文知识图谱问答系统

基于 Neo4j + LLM + 向量检索的中文知识图谱问答系统，支持实体搜索、关系查询、自然语言问答和多跳推理。

## 🎯 项目特点

- **中文优化**: 基于 CN-DBpedia/OwnThink 中文知识图谱，使用 BGE-large-zh 中文 Embedding 模型
- **RAG 问答**: 结合知识图谱和 LLM 的检索增强生成问答
- **多跳推理**: 支持多跳关系推理查询
- **向量检索**: 基于 BGE 的语义相似度搜索
- **RESTful API**: 提供标准 HTTP 接口

## 🏗️ 技术栈

- **图数据库**: Neo4j 5.x
- **后端框架**: Flask + Flask-CORS
- **LLM**: 通义千问 (qwen-plus) via OpenAI API
- **向量模型**: BAAI/bge-large-zh (中文优化)
- **向量索引**: 25,916 实体，1024 维

## 📁 项目结构

```
MiniGraph/
├── data/
│   ├── raw/                    # 原始数据
│   ├── processed/              # 处理后数据
│   └── cache/                  # 向量缓存
├── src/
│   ├── main.py                 # 主 API 服务
│   ├── llm/
│   │   └── llm_client_openai.py    # LLM 客户端
│   ├── retriever/
│   │   ├── vector_retriever_chinese.py  # 中文向量检索
│   │   └── vector_cache.py      # 向量缓存
│   ├── utils/
│   │   ├── parser.py           # 数据解析
│   │   ├── neo4j_importer.py   # Neo4j 导入
│   │   └── build_full_vector_index.py  # 构建向量索引
│   └── tests/
│       ├── hallucination_test_report.md  # 幻觉测试报告
│       └── multihop_test_report.md       # 多跳推理测试报告
├── logs/                       # 日志文件
└── README.md
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- Neo4j 5.x
- CUDA (可选，用于加速向量检索)

### 2. 安装依赖

```bash
pip install py2neo flask flask-cors openai numpy torch transformers
```

### 3. 配置环境变量

```bash
export LLM_API_KEY="your-api-key"
```

### 4. 启动 Neo4j

```bash
# 下载并启动 Neo4j
wget https://dist.neo4j.org/neo4j-community-5.26.21-unix.tar.gz
tar -xzf neo4j-community-5.26.21-unix.tar.gz
cd neo4j-community-5.26.21
./bin/neo4j start
```

### 5. 启动 API 服务

```bash
cd /autodl-fs/data/MiniGraph
export LLM_API_KEY="your-api-key"
python src/main.py
```

服务启动后访问: http://localhost:5000

## 📡 API 接口文档

### 基础接口

#### 首页
```
GET /
```

#### 健康检查
```
GET /health
```

#### API 指标
```
GET /metrics
```

### 实体查询

#### 搜索实体
```
GET /search?q=关键词
```

示例:
```bash
curl http://localhost:5000/search?q=唐朝
```

#### 实体详情
```
GET /entity/<实体名>
```

示例:
```bash
curl http://localhost:5000/entity/唐朝
```

#### 实体关系链
```
GET /entity_chain/<实体名>?depth=2&branches=3
```

示例:
```bash
curl http://localhost:5000/entity_chain/唐朝?depth=2
```

### 向量检索

#### 向量相似度搜索
```
GET /vector_search?q=查询&top_k=5
```

示例:
```bash
curl http://localhost:5000/vector_search?q=人工智能&top_k=5
```

### RAG 问答

#### 基础问答
```
POST /query
Content-Type: application/json

{"question": "问题"}
```

示例:
```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "唐朝的开国皇帝是谁？"}'
```

#### 增强版问答（含关系链）
```
POST /query_enhanced
Content-Type: application/json

{"question": "问题"}
```

### 多跳推理

```
POST /multihop
Content-Type: application/json

{"question": "问题", "max_hops": 2}
```

示例:
```bash
curl -X POST http://localhost:5000/multihop \
  -H "Content-Type: application/json" \
  -d '{"question": "唐朝的开国皇帝是谁？", "max_hops": 2}'
```

### 批量接口

#### 批量获取实体
```
POST /batch/entities
Content-Type: application/json

{"names": ["实体1", "实体2"], "include_relations": true}
```

#### 批量向量搜索
```
POST /batch/vector_search
Content-Type: application/json

{"queries": ["查询1", "查询2"], "top_k": 3}
```

### 实体消歧

```
POST /entity_linking
Content-Type: application/json

{"mention": "苹果", "context": "苹果公司发布了新款iPhone", "top_k": 3}
```

### 异步处理

#### 提交异步任务
```
POST /async/query
Content-Type: application/json

{"question": "问题"}
```

#### 获取任务结果
```
GET /async/result/<task_id>
```

## 📊 数据规模

- **实体数量**: 25,916
- **关系数量**: 约 200 万
- **向量维度**: 1024 (BGE-large-zh)
- **数据来源**: CN-DBpedia/OwnThink 中文百科

## 🔧 配置说明

### Neo4j 连接配置

编辑 `src/main.py`:
```python
graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
```

### LLM API 配置

设置环境变量:
```bash
export LLM_API_KEY="your-api-key"
```

或修改代码中的默认值。

### 向量索引配置

编辑 `src/main.py` 中的 `vector_index_paths`:
```python
vector_index_paths = [
    '/autodl-fs/data/MiniGraph/data/processed/vector_index_bge_full.json',
]
```

## 🧪 测试

### 幻觉测试

```bash
# 测试具体年份（可能编造）
curl -X POST http://localhost:5000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"唐朝是哪一年建立的？"}'

# 测试精确数字（可能编造）
curl -X POST http://localhost:5000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"唐朝有多少人口？"}'
```

### 多跳推理测试

```bash
# 测试多跳推理
curl -X POST http://localhost:5000/multihop \
  -H 'Content-Type: application/json' \
  -d '{"question":"唐朝的开国皇帝是谁？", "max_hops": 2}'
```

## 📝 日志查看

```bash
# 查看 API 日志
tail -f /tmp/api_server.log
```

## 🚀 部署指南

### AutoDL 服务器部署

1. **进入项目目录**
```bash
cd /autodl-fs/data/MiniGraph
```

2. **启动 Neo4j**（如未启动）
```bash
/root/autodl-fs/neo4j-community-5.26.21/bin/neo4j start
```

3. **启动 API 服务**
```bash
export LLM_API_KEY="your-api-key"
python src/main.py
```

4. **服务访问地址**
- 本地: http://localhost:5000
- 外网: http://<服务器IP>:5000

### 外网访问代理（Clash）

如需访问 GitHub/Hugging Face:
```bash
# 启动代理
cd /root/clash && sh start.sh

# 停止代理
cd /root/clash && sh stop.sh
```

## 📈 性能指标

- **向量检索**: 使用 BGE-large-zh，1024 维
- **向量索引**: 25,916 实体，约 535MB
- **启动时间**: 约 15 秒（含向量索引加载）
- **平均响应时间**: 
  - 实体查询: < 100ms
  - RAG 问答: 1-3s
  - 多跳推理: 1-2s

## 🧠 RAG 提示词优化

系统使用优化的 RAG 提示词模板，特点：
- 明确标注信息来源（知识图谱 vs 常识补充）
- 保留知识图谱中的相关度分数
- 禁止编造具体细节（年份、数字等）
- 合理补充常识并标注

详见测试报告：
- `tests/hallucination_test_report.md` - 幻觉测试报告
- `tests/multihop_test_report.md` - 多跳推理测试报告

## 🤝 贡献

欢迎提交 Issue 和 PR!

## 📄 许可证

MIT License

## 🙏 致谢

- [CN-DBpedia](https://github.com/GeneralZh/ownthink) - 中文知识图谱数据
- [Neo4j](https://neo4j.com/) - 图数据库
- [Flask](https://flask.palletsprojects.com/) - Web 框架
- [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh) - 中文 Embedding 模型
