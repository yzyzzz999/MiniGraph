"""
LLM 接入模块
支持 Kimi API 和 OpenAI API
"""
import os
import json
import requests
from typing import List, Dict, Optional

class LLMClient:
    """LLM 客户端"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "kimi-coding/k2p5"):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or "https://api.moonshot.cn/v1"
        self.model = model
        
    def chat(self, messages: List[Dict], temperature: float = 0.7) -> str:
        """
        调用 LLM 进行对话
        
        Args:
            messages: [{"role": "user", "content": "..."}]
            temperature: 温度参数
            
        Returns:
            LLM 回复文本
        """
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            return ""
    
    def generate(self, prompt: str, system: str = None) -> str:
        """简化调用方式"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages)


class RAGSystem:
    """RAG 系统：检索增强生成"""
    
    def __init__(self, llm_client: LLMClient, neo4j_graph=None, vector_store=None):
        self.llm = llm_client
        self.graph = neo4j_graph
        self.vector_store = vector_store
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        检索相关文档/实体
        
        1. 向量相似度搜索
        2. 关键词匹配
        3. 图数据库查询
        """
        results = []
        
        # 1. 向量检索（如果有）
        if self.vector_store:
            try:
                from vector_encoder_v2 import VectorEncoder
                encoder = VectorEncoder(model_name="microsoft/codebert-base")
                query_vec = encoder.encode([query])[0]
                vec_results = self.vector_store.search(query_vec, top_k=top_k)
                results.extend([{"source": "vector", **r} for r in vec_results])
            except:
                pass
        
        # 2. 图数据库检索
        if self.graph:
            try:
                # 模糊匹配实体名
                graph_results = self.graph.run(
                    "MATCH (e:Entity) WHERE e.name CONTAINS $query RETURN e.name as name LIMIT $limit",
                    query=query, limit=top_k
                ).data()
                results.extend([{"source": "graph", **r} for r in graph_results])
            except:
                pass
        
        return results[:top_k]
    
    def generate(self, query: str, context: List[Dict] = None) -> str:
        """
        生成回答
        
        Args:
            query: 用户问题
            context: 检索到的上下文
        """
        # 构建 prompt
        context_text = ""
        if context:
            context_text = "\n".join([
                f"- {c.get('name', '')}: {c.get('description', '')}"
                for c in context
            ])
        
        prompt = f"""基于以下信息回答问题：

上下文信息：
{context_text}

用户问题：{query}

请根据上下文信息回答问题。如果上下文不足以回答，请说明。"""

        return self.llm.generate(prompt)
    
    def query(self, question: str) -> Dict:
        """
        完整的 RAG 流程
        
        Returns:
            {
                "question": 问题,
                "answer": 回答,
                "context": 检索到的上下文
            }
        """
        # 1. 检索
        context = self.retrieve(question)
        
        # 2. 生成
        answer = self.generate(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "context": context
        }


class MultiHopReasoning:
    """多跳推理系统"""
    
    def __init__(self, llm_client: LLMClient, rag_system: RAGSystem):
        self.llm = llm_client
        self.rag = rag_system
        
    def plan(self, question: str) -> List[str]:
        """
        LLM 规划查询步骤
        
        Returns:
            查询步骤列表
        """
        prompt = f"""分析以下问题，规划查询步骤：

问题：{question}

请输出查询步骤，每行一个步骤，格式如下：
1. 查询[实体A]的基本信息
2. 查找[实体A]的[关系]关系
3. 查询[实体B]的[属性]

查询步骤："""

        response = self.llm.generate(prompt)
        steps = [s.strip() for s in response.split('\n') if s.strip()]
        return steps
    
    def execute(self, question: str, max_hops: int = 3) -> Dict:
        """
        执行多跳推理
        
        Args:
            question: 问题
            max_hops: 最大跳数
            
        Returns:
            {
                "question": 问题,
                "steps": 执行步骤,
                "answer": 最终答案
            }
        """
        # 1. 规划查询步骤
        steps = self.plan(question)
        steps = steps[:max_hops]  # 限制跳数
        
        # 2. 逐步执行
        context_history = []
        for i, step in enumerate(steps):
            print(f"执行步骤 {i+1}: {step}")
            
            # 执行单步查询
            result = self.rag.query(step)
            context_history.append({
                "step": step,
                "result": result
            })
        
        # 3. 综合生成答案
        final_prompt = f"""基于以下查询步骤和结果，回答问题：

问题：{question}

查询步骤：
"""
        for h in context_history:
            final_prompt += f"\n步骤：{h['step']}\n结果：{h['result']['answer']}\n"
        
        final_prompt += f"\n请综合以上信息，回答原始问题：{question}"
        
        answer = self.llm.generate(final_prompt)
        
        return {
            "question": question,
            "steps": steps,
            "context_history": context_history,
            "answer": answer
        }


# MCP 工具定义
MCP_TOOLS = {
    "search_entity": {
        "description": "搜索知识图谱中的实体",
        "parameters": {
            "query": "搜索关键词"
        }
    },
    "get_entity_relations": {
        "description": "获取实体的关系",
        "parameters": {
            "entity_name": "实体名称"
        }
    },
    "vector_search": {
        "description": "向量相似度搜索",
        "parameters": {
            "text": "查询文本"
        }
    }
}


class MCPClient:
    """MCP 客户端 - 工具调用"""
    
    def __init__(self, llm_client: LLMClient, tools: Dict = None):
        self.llm = llm_client
        self.tools = tools or MCP_TOOLS
        
    def call_tool(self, tool_name: str, parameters: Dict) -> Dict:
        """调用工具"""
        # 这里实现具体的工具调用逻辑
        # 实际使用时需要连接 Neo4j、Milvus 等
        return {
            "tool": tool_name,
            "parameters": parameters,
            "result": "工具执行结果"
        }
    
    def chat_with_tools(self, messages: List[Dict]) -> str:
        """
        支持工具调用的对话
        
        1. LLM 判断是否需要调用工具
        2. 执行工具
        3. 返回结果给 LLM
        4. LLM 生成最终回答
        """
        # 简化版：直接调用 LLM
        # 完整版需要解析 LLM 的工具调用请求
        return self.llm.chat(messages)


if __name__ == "__main__":
    # 测试
    llm = LLMClient(api_key="your-api-key")
    
    # 测试 LLM
    response = llm.generate("你好，请介绍一下知识图谱")
    print("LLM 回复:", response)
    
    # 测试 RAG
    rag = RAGSystem(llm)
    result = rag.query("什么是知识图谱？")
    print("RAG 结果:", result)
    
    # 测试多跳
    hop = MultiHopReasoning(llm, rag)
    result = hop.execute("周杰伦的妻子是谁？")
    print("多跳结果:", result)
