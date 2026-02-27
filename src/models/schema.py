"""
知识图谱 Schema 定义
CN-DBpedia 知识图谱结构
"""

# 实体类型定义
ENTITY_TYPES = {
    "Person": {
        "description": "人物",
        "attributes": ["姓名", "出生日期", "出生地", "职业", "国籍", "毕业院校", "代表作品"],
        "relations": ["配偶", "子女", "父母", "所属机构", "好友"]
    },
    "Organization": {
        "description": "组织/机构",
        "attributes": ["名称", "成立时间", "总部地点", "创始人", "类型", "行业"],
        "relations": ["子公司", "母公司", "竞争对手", "合作伙伴", "CEO"]
    },
    "Location": {
        "description": "地点",
        "attributes": ["名称", "所属国家", "人口", "面积", "时区", "气候"],
        "relations": ["所属省份", "相邻城市", "包含景点"]
    },
    "Work": {
        "description": "作品",
        "attributes": ["名称", "创作时间", "作者", "类型", "语言", "获奖情况"],
        "relations": ["作者", "改编作品", "续集", "前传"]
    },
    "Event": {
        "description": "事件",
        "attributes": ["名称", "发生时间", "地点", "参与人物", "结果", "影响"],
        "relations": ["参与人物", "发生地点", "相关事件"]
    },
    "Concept": {
        "description": "概念/学科",
        "attributes": ["名称", "定义", "所属学科", "创立者", "应用领域"],
        "relations": ["子概念", "父概念", "相关概念"]
    }
}

# 关系类型定义
RELATION_TYPES = {
    # 人物关系
    "配偶": {"domain": "Person", "range": "Person"},
    "子女": {"domain": "Person", "range": "Person"},
    "父母": {"domain": "Person", "range": "Person"},
    "好友": {"domain": "Person", "range": "Person"},
    
    # 组织关系
    "CEO": {"domain": "Organization", "range": "Person"},
    "创始人": {"domain": "Organization", "range": "Person"},
    "子公司": {"domain": "Organization", "range": "Organization"},
    "母公司": {"domain": "Organization", "range": "Organization"},
    
    # 地点关系
    "所属国家": {"domain": "Location", "range": "Location"},
    "所属省份": {"domain": "Location", "range": "Location"},
    "相邻城市": {"domain": "Location", "range": "Location"},
    
    # 作品关系
    "作者": {"domain": "Work", "range": "Person"},
    "导演": {"domain": "Work", "range": "Person"},
    "主演": {"domain": "Work", "range": "Person"},
    
    # 通用关系
    "类型": {"domain": "Entity", "range": "Concept"},
    "所属机构": {"domain": "Person", "range": "Organization"},
}

# CN-DBpedia 属性到 Schema 的映射
ATTRIBUTE_MAPPING = {
    # 人物属性
    "中文名": "姓名",
    "外文名": "外文名",
    "出生日期": "出生日期",
    "出生地": "出生地",
    "职业": "职业",
    "国籍": "国籍",
    "毕业院校": "毕业院校",
    "代表作品": "代表作品",
    "主要成就": "主要成就",
    
    # 组织属性
    "成立时间": "成立时间",
    "总部地点": "总部地点",
    "创始人": "创始人",
    "经营范围": "经营范围",
    
    # 地点属性
    "人口": "人口",
    "面积": "面积",
    "气候": "气候",
    
    # 作品属性
    "创作时间": "创作时间",
    "作者": "作者",
    "类型": "类型",
    "语言": "语言",
}

# Neo4j 图数据库 Schema
NEO4J_SCHEMA = {
    "node_labels": ["Person", "Organization", "Location", "Work", "Event", "Concept", "Entity"],
    "edge_types": list(RELATION_TYPES.keys()),
    "indexes": [
        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    ],
    "constraints": [
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    ]
}

# Milvus 向量数据库 Schema
MILVUS_SCHEMA = {
    "collection_name": "entity_vectors",
    "dimension": 768,  # BERT-base 向量维度
    "fields": [
        {"name": "id", "dtype": "INT64", "is_primary": True},
        {"name": "entity_name", "dtype": "VARCHAR", "max_length": 512},
        {"name": "entity_type", "dtype": "VARCHAR", "max_length": 64},
        {"name": "vector", "dtype": "FLOAT_VECTOR", "dim": 768},
        {"name": "description", "dtype": "VARCHAR", "max_length": 2048},
    ],
    "index_params": {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
}

if __name__ == '__main__':
    print("=== CN-DBpedia 知识图谱 Schema ===")
    print()
    print("实体类型:")
    for entity_type, info in ENTITY_TYPES.items():
        print(f"  {entity_type}: {info['description']}")
    print()
    print("关系类型数量:", len(RELATION_TYPES))
    print("属性映射数量:", len(ATTRIBUTE_MAPPING))
