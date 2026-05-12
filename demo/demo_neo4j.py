# query_neo4j.py
"""
查询 Neo4j 中的实体和关系
"""

from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# Neo4j 配置
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://172.20.48.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "rag")


class Neo4jQuery:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        print(f"✅ 连接 Neo4j: {NEO4J_URI}")

    def close(self):
        self.driver.close()
        print("✅ 连接已关闭")

    def query_entities(self, limit=50):
        """查询所有实体"""
        query = """
        MATCH (e:Entity) 
        RETURN e.name AS name, e.type AS type, e.frequency AS frequency
        ORDER BY e.frequency DESC
        LIMIT $limit
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, limit=limit)
            entities = []
            for record in result:
                entities.append({
                    "name": record["name"],
                    "type": record["type"],
                    "frequency": record["frequency"]
                })
            return entities

    def query_entity_by_name(self, name):
        """查询指定名称的实体"""
        query = """
        MATCH (e:Entity {name: $name}) 
        RETURN e.name AS name, e.type AS type, e.frequency AS frequency, e.summary AS summary
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, name=name).single()
            if result:
                return {
                    "name": result["name"],
                    "type": result["type"],
                    "frequency": result["frequency"],
                    "summary": result["summary"]
                }
            return None

    def query_entity_relations(self, name):
        """查询实体的所有关系"""
        query = """
        MATCH (e:Entity {name: $name})-[r]-(n:Entity)
        RETURN e.name AS source, 
               n.name AS target, 
               type(r) AS relation_type,
               r.weight AS weight,
               r.evidence AS evidence
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, name=name)
            relations = []
            for record in result:
                relations.append({
                    "source": record["source"],
                    "target": record["target"],
                    "relation_type": record["relation_type"],
                    "weight": record["weight"],
                    "evidence": record["evidence"][:200] if record["evidence"] else ""
                })
            return relations

    def query_entities_by_type(self, entity_type, limit=50):
        """按类型查询实体"""
        query = """
        MATCH (e:Entity {type: $type})
        RETURN e.name AS name, e.type AS type, e.frequency AS frequency
        ORDER BY e.frequency DESC
        LIMIT $limit
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, type=entity_type, limit=limit)
            entities = []
            for record in result:
                entities.append({
                    "name": record["name"],
                    "type": record["type"],
                    "frequency": record["frequency"]
                })
            return entities

    def fuzzy_search_entity(self, keyword):
        """模糊搜索实体（包含关键词）"""
        query = """
        MATCH (e:Entity)
        WHERE e.name CONTAINS $keyword
        RETURN e.name AS name, e.type AS type, e.frequency AS frequency
        ORDER BY e.frequency DESC
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, keyword=keyword)
            entities = []
            for record in result:
                entities.append({
                    "name": record["name"],
                    "type": record["type"],
                    "frequency": record["frequency"]
                })
            return entities

    def get_statistics(self):
        """获取图统计信息"""
        query = """
        MATCH (e:Entity)
        WITH count(e) AS entity_count
        MATCH ()-[r:RELATION]->()
        WITH entity_count, count(r) AS relation_count
        MATCH (c:Community)
        RETURN entity_count, relation_count, count(c) AS community_count
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query).single()
            return {
                "entity_count": result["entity_count"] if result else 0,
                "relation_count": result["relation_count"] if result else 0,
                "community_count": result["community_count"] if result else 0
            }


def main():
    print("=" * 60)
    print("Neo4j 数据查询")
    print("=" * 60)

    q = Neo4jQuery()

    try:
        # 1. 获取统计信息
        print("\n📊 1. 图统计信息")
        print("-" * 40)
        stats = q.get_statistics()
        print(f"   实体数量: {stats['entity_count']}")
        print(f"   关系数量: {stats['relation_count']}")
        print(f"   社区数量: {stats['community_count']}")

        # 2. 查询所有实体（前20个）
        print("\n📊 2. 所有实体（按频率降序，前20个）")
        print("-" * 40)
        entities = q.query_entities(20)
        for i, e in enumerate(entities, 1):
            print(f"   {i:2d}. {e['name']} ({e['type']}) - 频率: {e['frequency']}")

        # 3. 模糊搜索包含"云创"的实体
        print("\n🔍 3. 模糊搜索包含 '云创' 的实体")
        print("-" * 40)
        results = q.fuzzy_search_entity("云创")
        if results:
            for e in results:
                print(f"   - {e['name']} ({e['type']}) - 频率: {e['frequency']}")
        else:
            print("   ❌ 未找到包含 '云创' 的实体")

        # 4. 精确查询"云创科技"
        print("\n🔍 4. 精确查询实体: '云创科技'")
        print("-" * 40)
        entity = q.query_entity_by_name("云创科技")
        if entity:
            print(f"   名称: {entity['name']}")
            print(f"   类型: {entity['type']}")
            print(f"   频率: {entity['frequency']}")
            print(f"   摘要: {entity['summary'][:100]}...")
        else:
            print("   ❌ 未找到实体 '云创科技'")

            # 尝试模糊搜索其他名称
            print("\n   🔍 尝试搜索其他可能的名称...")
            for keyword in ["云创", "科技", "公司"]:
                results = q.fuzzy_search_entity(keyword)
                if results:
                    print(f"   包含 '{keyword}' 的实体:")
                    for e in results[:5]:
                        print(f"     - {e['name']} ({e['type']})")

        # 5. 查询云创科技的关系（如果存在）
        print("\n🔗 5. 查询 '云创科技' 的关系")
        print("-" * 40)
        relations = q.query_entity_relations("云创科技")
        if relations:
            for r in relations:
                print(f"   {r['source']} -[{r['relation_type']}]-> {r['target']} (权重: {r['weight']})")
        else:
            print("   ❌ 未找到 '云创科技' 的关系")

        # 6. 按类型统计（可选）
        print("\n📊 6. 按类型统计实体数量")
        print("-" * 40)
        type_counts = {}
        all_entities = q.query_entities(200)
        for e in all_entities:
            t = e['type']
            type_counts[t] = type_counts.get(t, 0) + 1

        for t, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {t}: {count} 个")

        # 7. 查看所有实体名称（用于调试）
        print("\n📋 7. 所有实体名称列表（前30个）")
        print("-" * 40)
        for i, e in enumerate(entities[:30], 1):
            print(f"   {i:2d}. {e['name']}")

    except Exception as e:
        print(f"\n❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        q.close()


if __name__ == "__main__":
    main()