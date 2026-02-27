# MiniGraph - 轻量级中文知识图谱问答系统

基于 Neo4j + Flask 的中文知识图谱问答系统，支持实体搜索、关系查询和自然语言问答。

## 🎯 项目特点

- **轻量级**: 无需复杂配置，快速部署
- **中文支持**: 基于 CN-DBpedia/OwnThink 中文知识图谱
- **实时查询**: 直接查询 Neo4j 图数据库
- **RESTful API**: 提供标准 HTTP 接口

## 🏗️ 技术栈

- **图数据库**: Neo4j 5.x
- **后端框架**: Flask + Flask-CORS
- **图数据**: CN-DBpedia/OwnThink (中文百科)
- **向量检索**: Milvus Lite / sklearn (可选)

## 📁 项目结构

```
MiniGraph/
├── data/
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后数据
├── src/
│   ├── api_server_neo4j.py   # Neo4j API 服务
│   ├── utils/
│   │   ├── parser.py         # 数据解析
│   │   ├── neo4j_importer.py # Neo4j 导入
│   │   └── vector_encoder_milvus.py  # 向量编码
│   ├── models/
│   │   └── schema.py         # 图谱 Schema
│   └── agents/
│       └── multi_agent.py    # 多 Agent 框架
├── logs/                 # 日志文件
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install py2neo flask flask-cors
```

### 2. 启动 Neo4j

```bash
# 下载并启动 Neo4j
wget https://dist.neo4j.org/neo4j-community-5.26.21-unix.tar.gz
tar -xzf neo4j-community-5.26.21-unix.tar.gz
cd neo4j-community-5.26.21
./bin/neo4j start
```

修改配置允许外部访问:
```bash
# conf/neo4j.conf
server.default_listen_address=0.0.0.0
```

### 3. 导入数据

```bash
cd /autodl-fs/data/MiniGraph
python import_to_neo4j.py
```

### 4. 启动 API 服务

```bash
python src/api_server_neo4j.py
```

服务启动后访问: http://localhost:5000

## 📡 API 接口

### 首页
```
GET /
```

### 统计信息
```
GET /stats
```

### 搜索实体
```
GET /search?q=关键词
```

示例:
```bash
curl http://localhost:5000/search?q=社会主义
```

### 实体详情
```
GET /entity/<实体名>
```

示例:
```bash
curl http://localhost:5000/entity/社会主义荣辱观
```

### 自然语言查询
```
POST /query
Content-Type: application/json

{"question": "社会主义是什么？"}
```

示例:
```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "社会主义是什么？"}'
```

## 📊 数据规模

- **实体数量**: 8.4 万+
- **关系数量**: 10 万+
- **数据来源**: CN-DBpedia/OwnThink 中文百科

## 🔧 配置说明

### Neo4j 连接配置

编辑 `src/api_server_neo4j.py`:
```python
graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
```

### 修改监听端口

```python
app.run(host='0.0.0.0', port=5000)
```

## 📝 日志查看

```bash
# API 日志
tail -f logs/api.log

# 导入日志
tail -f logs/import_data.log
```

## 🤝 贡献

欢迎提交 Issue 和 PR!

## 📄 许可证

MIT License

## 🙏 致谢

- [CN-DBpedia](https://github.com/GeneralZh/ownthink) - 中文知识图谱数据
- [Neo4j](https://neo4j.com/) - 图数据库
- [Flask](https://flask.palletsprojects.com/) - Web 框架
