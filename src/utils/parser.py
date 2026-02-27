"""
CN-DBpedia 数据解析模块
解析 baiketriples.txt 文件
"""
import json
from pathlib import Path
from typing import Iterator, Dict, Tuple
from collections import defaultdict

class CNDbpediaParser:
    """CN-DBpedia 数据解析器"""
    
    def __init__(self, data_file: str):
        self.data_file = Path(data_file)
        self.stats = defaultdict(int)
    
    def parse_triples(self) -> Iterator[Dict[str, str]]:
        """
        解析三元组数据
        格式: 实体\t属性\t值\n
        """
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 3:
                    entity = parts[0]
                    attribute = parts[1]
                    value = '\t'.join(parts[2:])  # 值可能包含tab
                    
                    self.stats['total_triples'] += 1
                    self.stats[f'attr_{attribute}'] += 1
                    
                    yield {
                        'entity': entity,
                        'attribute': attribute,
                        'value': value,
                        'line_num': line_num
                    }
                else:
                    self.stats['invalid_lines'] += 1
    
    def get_entity_types(self) -> Dict[str, int]:
        """获取实体类型分布"""
        type_counts = defaultdict(int)
        for triple in self.parse_triples():
            if triple['attribute'] == '类型':
                type_counts[triple['value']] += 1
        return dict(type_counts)
    
    def get_relation_types(self) -> Dict[str, int]:
        """获取关系类型分布"""
        return dict(self.stats)
    
    def build_entity_dict(self) -> Dict[str, Dict]:
        """
        构建实体字典
        每个实体包含所有属性和值
        """
        entities = defaultdict(lambda: {'name': '', 'attributes': {}})
        
        for triple in self.parse_triples():
            entity_name = triple['entity']
            attr = triple['attribute']
            value = triple['value']
            
            entities[entity_name]['name'] = entity_name
            
            if attr not in entities[entity_name]['attributes']:
                entities[entity_name]['attributes'][attr] = []
            entities[entity_name]['attributes'][attr].append(value)
        
        return dict(entities)
    
    def print_stats(self):
        """打印统计信息"""
        print(f"总三元组数: {self.stats['total_triples']}")
        print(f"无效行数: {self.stats.get('invalid_lines', 0)}")
        print("\n属性分布 (Top 20):")
        sorted_attrs = sorted(
            [(k, v) for k, v in self.stats.items() if k.startswith('attr_')],
            key=lambda x: x[1],
            reverse=True
        )[:20]
        for attr, count in sorted_attrs:
            print(f"  {attr.replace('attr_', '')}: {count}")


def main():
    """测试解析器"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python parser.py <baiketriples.txt>")
        return
    
    data_file = sys.argv[1]
    parser = CNDbpediaParser(data_file)
    
    print("开始解析 CN-DBpedia 数据...")
    print(f"文件: {data_file}")
    print()
    
    # 解析前 10000 条测试
    count = 0
    for triple in parser.parse_triples():
        count += 1
        if count <= 5:
            print(f"[{count}] {triple['entity']} | {triple['attribute']} | {triple['value'][:50]}")
        if count >= 10000:
            break
    
    print()
    parser.print_stats()


if __name__ == '__main__':
    main()
