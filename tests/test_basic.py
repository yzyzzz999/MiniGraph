"""
测试用例
"""
import unittest
import sys
sys.path.append('/autodl-fs/data/MiniGraph/src')

class TestParser(unittest.TestCase):
    """测试数据解析器"""
    
    def test_parse_triple(self):
        """测试三元组解析"""
        from utils.parser import CNDbpediaParser
        # TODO: 添加测试数据
        self.assertTrue(True)

class TestSchema(unittest.TestCase):
    """测试 Schema 定义"""
    
    def test_entity_types(self):
        """测试实体类型"""
        from models.schema import ENTITY_TYPES
        self.assertIn('Person', ENTITY_TYPES)
        self.assertIn('Organization', ENTITY_TYPES)

class TestAgents(unittest.TestCase):
    """测试 Agent 框架"""
    
    def test_entity_recognition(self):
        """测试实体识别 Agent"""
        from agents.multi_agent import EntityRecognitionAgent
        agent = EntityRecognitionAgent()
        result = agent.process({'question': '周杰伦的妻子是谁？'})
        self.assertIn('entities', result)

if __name__ == '__main__':
    unittest.main()
