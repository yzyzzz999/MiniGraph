# HEARTBEAT.md - MiniGraph 项目状态追踪

> 本文件由 Kimi 自动维护，每天第一次 heartbeat 时读取。
> 用户也可随时查看/修改。

---

## 📌 当前阶段

**Phase 2: LLM + RAG + 向量检索** — ✅ 已完成

---

## ✅ 已完成阶段

| 阶段 | 状态 | 关键成果 |
|------|------|----------|
| Phase 1 | ✅ | Neo4j 图数据库搭建、CN-DBpedia 数据导入、基础 API |
| Phase 2 | ✅ | LLM 接入、RAG 问答、向量检索、多跳推理 |

---

## 🎉 Phase 2 完成总结

### 已完成任务
- [x] LLM 客户端 (`llm_client_openai.py`)
  - 支持通义千问 API
  - 支持缓存和限流保护
- [x] 向量检索模块 (`vector_retriever.py`)
  - CodeBERT 编码实体
  - sklearn 相似度计算
- [x] 完整 API 服务 (`api_server_final.py`)
  - `/query` - LLM + RAG 问答
  - `/vector_search` - 向量相似度搜索
  - `/search` - 混合搜索（关键词 + 向量）
  - `/multihop` - 多跳推理
  - `/chat` - 直接对话 LLM
- [x] 代码提交到 GitHub

### 技术栈
| 组件 | 版本/型号 |
|------|----------|
| LLM | 通义千问 (qwen-plus) |
| Embedding | CodeBERT |
| 向量检索 | sklearn cosine_similarity |
| 图数据库 | Neo4j |
| API 框架 | Flask |

---

## 🔄 明天任务 (Phase 3)

### 目标：优化向量检索，提升问答质量

**待完成：**
- [ ] 替换 CodeBERT 为中文 embedding 模型
  - 候选: `BAAI/bge-large-zh` 或 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- [ ] 重新构建向量索引
- [ ] 测试向量检索准确性
- [ ] 优化 RAG 提示词模板
- [ ] 添加实体关系链式检索

---

## 📂 关键路径

| 类型 | 路径/地址 |
|------|----------|
| 项目根目录 (服务器) | `/autodl-fs/data/MiniGraph/` |
| LLM 客户端 | `src/llm_client_openai.py` |
| 向量检索 | `src/vector_retriever.py` |
| API 服务 | `src/api_server_final.py` |
| 向量索引 | `data/processed/vector_index.json` |
| GitHub 仓库 | https://github.com/yzyzzz999/mini-graph-rag |

---

## 🛠️ 启动命令

```bash
cd /autodl-fs/data/MiniGraph
export LLM_API_KEY="sk-581f4dde4cf643d2b6af556d5c592be7"
python src/api_server_final.py
```

---

## 📝 已知问题

1. **向量检索准确性低**
   - CodeBERT 是针对代码训练的，对中文语义理解不好
   - 需要替换为中文 embedding 模型

2. **RAG 上下文质量**
   - 当前只检索实体名称，未利用实体描述和关系
   - 需要改进上下文构建逻辑

---

## 🔗 相关记忆文件

- 长期记忆: `/root/.openclaw/workspace/MEMORY.md`
- 每日记忆: `/root/.openclaw/workspace/memory/2026-02-28.md`

---

## 📅 最近更新

- 2026-02-28 00:40: Phase 2 完成，LLM + RAG + 向量检索模块上线
- 2026-02-27 23:00: 开始 Phase 2，实现 LLM 接入
- 2026-02-26: Phase 1 完成，Neo4j + CN-DBpedia 数据导入

---

*最后更新: 2026-02-28 00:40*
