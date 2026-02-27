"""
Neo4j 数据导入模块
将 CN-DBpedia 三元组导入 Neo4j 图数据库
"""
from py2neo import Graph, Node, Relationship, NodeMatcher
from typing import Iterator, Dict, List, Tuple
from tqdm import tqdm
import json

class Neo4jImporter:
    """Neo4j 数据导入器"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", password: str = "password"):
        self.graph = Graph(uri, auth=(user, password))
        self.matcher = NodeMatcher(self.graph)
        self.batch_size = 1000
        
    def create_indexes(self):
        """创建索引"""
        print("创建索引...")
        self.graph.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
        self.graph.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")
        print("索引创建完成")
        
    def clear_database(self):
        """清空数据库"""
        print("清空数据库...")
        self.graph.run("MATCH (n) DETACH DELETE n")
        print("数据库已清空")
        
    def import_entities(self, entities: Dict[str, Dict]):
        """
        导入实体
        entities: {entity_name: {'type': 'Person', 'attributes': {...}}}
        """
        print(f"导入 {len(entities)} 个实体...")
        
        tx = self.graph.begin()
        count = 0
        
        for entity_name, data in tqdm(entities.items()):
            # 创建实体节点
            entity_type = data.get('type', 'Entity')
            node = Node(entity_type, name=entity_name)
            
            # 添加属性
            for attr, values in data.get('attributes', {}).items():
                if isinstance(values, list) and len(values) == 1:
                    node[attr] = values[0]
                else:
                    node[attr] = values
            
            tx.create(node)
            count += 1
            
            # 批量提交
            if count % self.batch_size == 0:
                tx.commit()
                tx = self.graph.begin()
        
        tx.commit()
        print(f"实体导入完成: {count}")
        
    def import_relations(self, relations: List[Dict]):
        """
        导入关系
        relations: [{'from': 'A', 'to': 'B', 'type': '配偶'}]
        """
        print(f"导入 {len(relations)} 个关系...")
        
        tx = self.graph.begin()
        count = 0
        
        for rel in tqdm(relations):
            from_name = rel['from']
            to_name = rel['to']
            rel_type = rel['type']
            
            # 查找节点
            from_node = self.matcher.match(name=from_name).first()
            to_node = self.matcher.match(name=to_name).first()
            
            if from_node and to_node:
                # 创建关系
                relationship = Relationship(from_node, rel_type, to_node)
                tx.create(relationship)
                count += 1
                
                # 批量提交
                if count % self.batch_size == 0:
                    tx.commit()
                    tx = self.graph.begin()
        
        tx.commit()
        print(f"关系导入完成: {count}")
        
    def query_entity(self, name: str) -> Dict:
        """查询实体"""
        node = self.matcher.match(name=name).first()
        if node:
            return dict(node)
        return None
        
    def query_relation(self, from_name: str, relation_type: str = None) -> List[Dict]:
        """查询关系"""
        if relation_type:
            query = """
            MATCH (e:Entity {name: $name})-[r:$rel_type]->(target)
            RETURN target.name as target, type(r) as relation
            """
            result = self.graph.run(query, name=from_name, rel_type=relation_type)
        else:
            query = """
            MATCH (e:Entity {name: $name})-[r]->(target)
            RETURN target.name as target, type(r) as relation
            """
            result = self.graph.run(query, name=from_name)
        
        return [dict(record) for record in result]
        
    def get_stats(self) -> Dict:
        """获取数据库统计"""
        node_count = self.graph.run("MATCH (n) RETURN count(n) as count").data()[0]['count']
        rel_count = self.graph.run("MATCH ()-[r]->() RETURN count(r) as count").data()[0]['count']
        
        return {
            'nodes': node_count,
            'relationships': rel_count
        }


def main():
    """测试导入"""
    importer = Neo4jImporter()
    
    # 创建索引
    importer.create_indexes()
    
    # 测试数据
    test_entities = {
        "周杰伦": {
            "type": "Person",
            "attributes": {
                "职业": ["歌手", "演员"],
                "国籍": ["中国"]
            }
        },
        "昆凌": {
            "type": "Person", 
            "attributes": {
                "职业": ["模特", "演员"],
                "国籍": ["中国"]
            }
        }
    }
    
    test_relations = [
        {"from": "周杰伦", "to": "昆凌", "type": "配偶"}
    ]
    
    # 导入
    importer.import_entities(test_entities)
    importer.import_relations(test_relations)
    
    # 查询
    stats = importer.get_stats()
    print(f"\n数据库统计: {stats}")
    
    result = importer.query_entity("周杰伦")
    print(f"\n查询结果: {result}")


if __name__ == '__main__':
    main()
