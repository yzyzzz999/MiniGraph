"""
多 Agent 协同框架
用于知识图谱问答系统
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import json

class BaseAgent(ABC):
    """Agent 基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.memory = []
    
    @abstractmethod
    def process(self, input_data: Dict) -> Dict:
        """处理输入，返回结果"""
        pass
    
    def log(self, message: str):
        """记录日志"""
        self.memory.append({"agent": self.name, "message": message})
        print(f"[{self.name}] {message}")


class EntityRecognitionAgent(BaseAgent):
    """
    Agent 1: 实体识别
    从用户问题中提取实体
    """
    
    def __init__(self):
        super().__init__("EntityRecognition")
    
    def process(self, input_data: Dict) -> Dict:
        """
        输入: {"question": "周杰伦的妻子是谁？"}
        输出: {"entities": [{"name": "周杰伦", "type": "Person", "mention": "周杰伦"}]}
        """
        question = input_data.get("question", "")
        self.log(f"处理问题: {question}")
        
        # TODO: 使用 NER 模型识别实体
        # 这里先用简单的规则演示
        entities = []
        
        # 模拟识别结果
        if "周杰伦" in question:
            entities.append({"name": "周杰伦", "type": "Person", "mention": "周杰伦"})
        if "妻子" in question or "配偶" in question:
            entities.append({"name": "配偶", "type": "Relation", "mention": "妻子"})
        
        self.log(f"识别到 {len(entities)} 个实体")
        
        return {
            "entities": entities,
            "original_question": question
        }


class QueryUnderstandingAgent(BaseAgent):
    """
    Agent 2: 查询理解
    理解用户意图，构建查询计划
    """
    
    def __init__(self):
        super().__init__("QueryUnderstanding")
    
    def process(self, input_data: Dict) -> Dict:
        """
        输入: {"entities": [...], "original_question": "..."}
        输出: {"query_plan": {"intent": "find_relation", "subject": "...", "relation": "..."}}
        """
        entities = input_data.get("entities", [])
        question = input_data.get("original_question", "")
        
        self.log(f"分析查询意图")
        
        # 识别查询意图
        intent = "unknown"
        if "谁" in question or "是谁" in question:
            intent = "find_entity"
        elif "妻子" in question or "丈夫" in question or "配偶" in question:
            intent = "find_relation"
        elif "哪里" in question or "地点" in question:
            intent = "find_location"
        
        # 构建查询计划
        query_plan = {
            "intent": intent,
            "entities": entities,
            " hops": 1  # 单跳查询
        }
        
        self.log(f"查询意图: {intent}")
        
        return {
            "query_plan": query_plan,
            "original_question": question
        }


class GraphReasoningAgent(BaseAgent):
    """
    Agent 3: 图谱推理
    在 Neo4j 中执行查询，获取答案
    """
    
    def __init__(self, neo4j_driver=None):
        super().__init__("GraphReasoning")
        self.driver = neo4j_driver
    
    def process(self, input_data: Dict) -> Dict:
        """
        输入: {"query_plan": {...}}
        输出: {"answer": "...", "paths": [...]}
        """
        query_plan = input_data.get("query_plan", {})
        intent = query_plan.get("intent", "unknown")
        entities = query_plan.get("entities", [])
        
        self.log(f"执行图谱查询，意图: {intent}")
        
        # TODO: 根据意图生成 Cypher 查询
        # 这里模拟查询结果
        answer = None
        paths = []
        
        if intent == "find_relation":
            # 查找关系
            for entity in entities:
                if entity.get("type") == "Person":
                    # 模拟查询结果
                    answer = "昆凌"
                    paths = [
                        {"from": "周杰伦", "relation": "配偶", "to": "昆凌"}
                    ]
                    break
        
        self.log(f"查询完成，找到答案: {answer}")
        
        return {
            "answer": answer,
            "paths": paths,
            "query_plan": query_plan
        }


class AnswerGenerationAgent(BaseAgent):
    """
    Agent 4: 答案生成
    整合结果，生成自然语言回答
    """
    
    def __init__(self):
        super().__init__("AnswerGeneration")
    
    def process(self, input_data: Dict) -> Dict:
        """
        输入: {"answer": "...", "paths": [...]}
        输出: {"response": "...", "confidence": 0.95}
        """
        answer = input_data.get("answer")
        paths = input_data.get("paths", [])
        question = input_data.get("original_question", "")
        
        self.log("生成自然语言回答")
        
        if answer:
            # 生成回答
            if paths:
                path_str = " → ".join([f"{p['from']}-{p['relation']}-{p['to']}" for p in paths])
                response = f"{answer}。推理路径: {path_str}"
            else:
                response = f"{answer}"
            confidence = 0.9
        else:
            response = "抱歉，我没有找到相关信息。"
            confidence = 0.0
        
        self.log(f"回答: {response}")
        
        return {
            "response": response,
            "confidence": confidence,
            "answer": answer,
            "paths": paths
        }


class MultiAgentSystem:
    """多 Agent 协同系统"""
    
    def __init__(self):
        self.agents = {
            "entity_recognition": EntityRecognitionAgent(),
            "query_understanding": QueryUnderstandingAgent(),
            "graph_reasoning": GraphReasoningAgent(),
            "answer_generation": AnswerGenerationAgent(),
        }
    
    def process(self, question: str) -> Dict:
        """处理用户问题"""
        print(f"\n{'='*50}")
        print(f"用户问题: {question}")
        print(f"{'='*50}\n")
        
        # Step 1: 实体识别
        result = self.agents["entity_recognition"].process({"question": question})
        
        # Step 2: 查询理解
        result = self.agents["query_understanding"].process(result)
        
        # Step 3: 图谱推理
        result = self.agents["graph_reasoning"].process(result)
        result["original_question"] = question
        
        # Step 4: 答案生成
        result = self.agents["answer_generation"].process(result)
        
        print(f"\n{'='*50}")
        print(f"最终回答: {result['response']}")
        print(f"{'='*50}\n")
        
        return result


if __name__ == '__main__':
    # 测试
    system = MultiAgentSystem()
    
    test_questions = [
        "周杰伦的妻子是谁？",
        "阿里巴巴的创始人是谁？",
    ]
    
    for q in test_questions:
        system.process(q)
