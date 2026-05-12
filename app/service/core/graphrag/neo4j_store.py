"""
Neo4j 图存储 - 唯一图存储后端
"""

import os
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase, AsyncGraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.error("neo4j 模块未安装，GraphRAG 功能不可用。请运行: pip install neo4j")


def get_neo4j_config() -> Dict[str, Any]:
    """从环境变量获取 Neo4j 配置"""
    return {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", ""),
        "database": os.getenv("NEO4J_DATABASE", "neo4j"),
        "max_connection_pool_size": int(os.getenv("NEO4J_MAX_POOL_SIZE", "50")),
        "connection_timeout": int(os.getenv("NEO4J_CONNECTION_TIMEOUT", "30")),
    }


@dataclass
class StoredEntity:
    """存储的实体"""
    name: str
    entity_type: str
    frequency: int = 1
    summary: str = ""
    metadata: Dict[str, Any] = None
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "name": self.name,
            "type": self.entity_type,
            "frequency": self.frequency,
            "summary": self.summary,
            "metadata": self.metadata or {}
        }


@dataclass
class StoredRelation:
    """存储的关系"""
    source: str
    target: str
    relation_type: str
    weight: float = 1.0
    evidence: str = ""
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "evidence": self.evidence[:200] if self.evidence else ""
        }


class Neo4jStore:
    """
    Neo4j 图存储 - 唯一后端
    用于存储实体、关系和社区
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j 模块未安装，GraphRAG 功能不可用")

        self._config = get_neo4j_config()
        self._driver = None
        self._async_driver = None
        self._connected = False

        self._connect()

        logger.info(f"Neo4jStore 初始化完成: uri={self._config['uri']}, database={self._config['database']}")

    def _connect(self):
        """建立 Neo4j 连接"""
        try:
            self._driver = GraphDatabase.driver(
                self._config["uri"],
                auth=(self._config["user"], self._config["password"]),
                max_connection_pool_size=self._config["max_connection_pool_size"],
                connection_timeout=self._config["connection_timeout"]
            )
            # 验证连接
            with self._driver.session(database=self._config["database"]) as session:
                session.run("RETURN 1")
            self._connected = True
            logger.info(f"Neo4j 连接成功: {self._config['uri']}")
        except Exception as e:
            logger.error(f"Neo4j 连接失败: {e}")
            raise

    async def _get_async_driver(self):
        """获取异步驱动"""
        if self._async_driver is None:
            self._async_driver = AsyncGraphDatabase.driver(
                self._config["uri"],
                auth=(self._config["user"], self._config["password"]),
                max_connection_pool_size=self._config["max_connection_pool_size"],
                connection_timeout=self._config["connection_timeout"]
            )
        return self._async_driver

    def _get_session(self):
        """获取同步会话"""
        if not self._connected:
            self._connect()
        return self._driver.session(database=self._config["database"])

    async def _get_async_session(self):
        """获取异步会话"""
        driver = await self._get_async_driver()
        return driver.session(database=self._config["database"])

    # ========== 索引初始化 ==========

    def init_constraints_and_indexes(self):
        """初始化约束和索引（启动时调用一次）"""
        with self._get_session() as session:
            # 实体约束和索引
            try:
                session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            except Exception as e:
                logger.debug(f"实体约束可能已存在: {e}")

            try:
                session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")
            except Exception as e:
                logger.debug(f"实体类型索引可能已存在: {e}")

            try:
                session.run("CREATE INDEX entity_frequency IF NOT EXISTS FOR (e:Entity) ON (e.frequency)")
            except Exception as e:
                logger.debug(f"实体频率索引可能已存在: {e}")

            # 关系索引
            try:
                session.run("CREATE INDEX relation_type IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.type)")
            except Exception as e:
                logger.debug(f"关系类型索引可能已存在: {e}")

            # 社区索引
            try:
                session.run("CREATE INDEX community_id IF NOT EXISTS FOR (c:Community) ON (c.community_id)")
            except Exception as e:
                logger.debug(f"社区ID索引可能已存在: {e}")

            logger.info("Neo4j 约束和索引初始化完成")

    # ========== 实体操作 ==========

    def save_entity(self, entity: StoredEntity) -> bool:
        """保存单个实体"""
        query = """
        MERGE (e:Entity {name: $name})
        SET e.type = $type,
            e.frequency = COALESCE(e.frequency, 0) + $frequency,
            e.summary = $summary,
            e.updated_at = datetime()
        RETURN e
        """
        try:
            with self._get_session() as session:
                session.run(query, {
                    "name": entity.name,
                    "type": entity.entity_type,
                    "frequency": entity.frequency,
                    "summary": entity.summary
                })
            return True
        except Exception as e:
            logger.error(f"保存实体失败 {entity.name}: {e}")
            return False

    def save_entities_batch(self, entities: List[StoredEntity]) -> int:
        """批量保存实体"""
        if not entities:
            return 0

        query = """
        UNWIND $entities AS entity
        MERGE (e:Entity {name: entity.name})
        SET e.type = entity.type,
            e.frequency = COALESCE(e.frequency, 0) + entity.frequency,
            e.summary = entity.summary,
            e.updated_at = datetime()
        RETURN count(e)
        """
        try:
            with self._get_session() as session:
                result = session.run(query, {
                    "entities": [
                        {
                            "name": e.name,
                            "type": e.entity_type,
                            "frequency": e.frequency,
                            "summary": e.summary
                        }
                        for e in entities
                    ]
                })
                return result.single()[0]
        except Exception as e:
            logger.error(f"批量保存实体失败: {e}")
            return 0

    def get_entity(self, name: str) -> Optional[StoredEntity]:
        """获取实体"""
        query = """
        MATCH (e:Entity {name: $name})
        RETURN e.name AS name, e.type AS type, e.frequency AS frequency, e.summary AS summary
        """
        try:
            with self._get_session() as session:
                result = session.run(query, {"name": name}).single()
                if result:
                    return StoredEntity(
                        name=result["name"],
                        entity_type=result["type"],
                        frequency=result["frequency"],
                        summary=result["summary"] or ""
                    )
            return None
        except Exception as e:
            logger.error(f"获取实体失败 {name}: {e}")
            return None

    def get_all_entities(self, limit: int = 1000) -> List[StoredEntity]:
        """获取所有实体"""
        query = """
        MATCH (e:Entity)
        RETURN e.name AS name, e.type AS type, e.frequency AS frequency, e.summary AS summary
        ORDER BY e.frequency DESC
        LIMIT $limit
        """
        try:
            with self._get_session() as session:
                results = session.run(query, {"limit": limit})
                return [
                    StoredEntity(
                        name=r["name"],
                        entity_type=r["type"],
                        frequency=r["frequency"],
                        summary=r["summary"] or ""
                    )
                    for r in results
                ]
        except Exception as e:
            logger.error(f"获取所有实体失败: {e}")
            return []

    def delete_entity(self, name: str) -> bool:
        """删除实体及关联关系"""
        query = """
        MATCH (e:Entity {name: $name})
        DETACH DELETE e
        """
        try:
            with self._get_session() as session:
                session.run(query, {"name": name})
            return True
        except Exception as e:
            logger.error(f"删除实体失败 {name}: {e}")
            return False

    # ========== 关系操作 ==========

    def save_relation(self, relation: StoredRelation) -> bool:
        """保存单个关系"""
        query = """
        MATCH (source:Entity {name: $source})
        MATCH (target:Entity {name: $target})
        MERGE (source)-[r:RELATION {type: $relation_type}]->(target)
        SET r.weight = COALESCE(r.weight, 0) + $weight,
            r.evidence = $evidence,
            r.updated_at = datetime()
        RETURN r
        """
        try:
            with self._get_session() as session:
                session.run(query, {
                    "source": relation.source,
                    "target": relation.target,
                    "relation_type": relation.relation_type,
                    "weight": relation.weight,
                    "evidence": relation.evidence
                })
            return True
        except Exception as e:
            logger.error(f"保存关系失败 {relation.source}->{relation.target}: {e}")
            return False

    def save_relations_batch(self, relations: List[StoredRelation]) -> int:
        """批量保存关系"""
        if not relations:
            return 0

        query = """
        UNWIND $relations AS rel
        MATCH (source:Entity {name: rel.source})
        MATCH (target:Entity {name: rel.target})
        MERGE (source)-[r:RELATION {type: rel.relation_type}]->(target)
        SET r.weight = COALESCE(r.weight, 0) + rel.weight,
            r.evidence = rel.evidence,
            r.updated_at = datetime()
        RETURN count(r)
        """
        try:
            with self._get_session() as session:
                result = session.run(query, {
                    "relations": [
                        {
                            "source": r.source,
                            "target": r.target,
                            "relation_type": r.relation_type,
                            "weight": r.weight,
                            "evidence": r.evidence
                        }
                        for r in relations
                    ]
                })
                return result.single()[0] if result else 0
        except Exception as e:
            logger.error(f"批量保存关系失败: {e}")
            return 0

    def get_relations(self, source: str = None, target: str = None) -> List[StoredRelation]:
        """获取关系"""
        query_parts = ["MATCH (source:Entity)-[r:RELATION]->(target:Entity)"]
        params = {}

        if source:
            query_parts.append("WHERE source.name = $source")
            params["source"] = source
        if target:
            query_parts.append("WHERE target.name = $target")
            params["target"] = target

        query_parts.append("RETURN source.name AS source, target.name AS target, r.type AS relation_type, r.weight AS weight, r.evidence AS evidence")

        try:
            with self._get_session() as session:
                results = session.run(" ".join(query_parts), params)
                return [
                    StoredRelation(
                        source=r["source"],
                        target=r["target"],
                        relation_type=r["relation_type"],
                        weight=r["weight"],
                        evidence=r["evidence"] or ""
                    )
                    for r in results
                ]
        except Exception as e:
            logger.error(f"获取关系失败: {e}")
            return []

    # ========== 图查询操作 ==========

    def get_neighbors(self, entity_name: str, max_depth: int = 2) -> List[Tuple[str, int, str]]:
        """
        获取实体的邻居（广度优先）
        Returns: [(entity_name, depth, relation_type), ...]
        """
        # 修复：使用 Cypher 的变量长度路径，不使用 dynamic parameter for max_depth
        query = f"""
        MATCH path = (start:Entity {{name: $name}})-[*1..{max_depth}]-(neighbor:Entity)
        WHERE start <> neighbor
        WITH neighbor, length(path) AS depth
        OPTIONAL MATCH (start)-[r:RELATION]-(neighbor)
        RETURN DISTINCT neighbor.name AS name, depth, r.type AS relation_type
        ORDER BY depth, name
        """
        try:
            with self._get_session() as session:
                results = session.run(query, {"name": entity_name})
                return [(r["name"], r["depth"], r.get("relation_type") or "RELATED_TO") for r in results]
        except Exception as e:
            logger.error(f"获取邻居失败 {entity_name}: {e}")
            return []

    def get_path_between(self, source: str, target: str, max_depth: int = 3) -> List[Dict]:
        """获取两个实体之间的路径"""
        # 修复：使用 f-string 嵌入 max_depth
        query = f"""
        MATCH path = shortestPath((source:Entity {{name: $source}})-[*1..{max_depth}]-(target:Entity {{name: $target}}))
        RETURN [node in nodes(path) | node.name] AS nodes,
               [rel in relationships(path) | rel.type] AS relations,
               length(path) AS length
        """
        try:
            with self._get_session() as session:
                result = session.run(query, {"source": source, "target": target}).single()
                if result:
                    return [{
                        "nodes": result["nodes"],
                        "relations": result["relations"],
                        "length": result["length"]
                    }]
            return []
        except Exception as e:
            logger.error(f"获取路径失败 {source}->{target}: {e}")
            return []

    def find_related_entities(self, entity_names: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
        """
        找到与一组实体相关的其他实体
        Returns: [(entity_name, relevance_score), ...]
        """
        if not entity_names:
            return []

        query = """
        MATCH (e:Entity)
        WHERE e.name IN $entities
        WITH collect(e) AS start_entities
        MATCH (start)-[r:RELATION]-(related:Entity)
        WHERE start IN start_entities AND related.name NOT IN $entities
        RETURN related.name AS name, 
               sum(r.weight) AS relevance,
               count(r) AS connection_count
        ORDER BY relevance DESC, connection_count DESC
        LIMIT $top_k
        """
        try:
            with self._get_session() as session:
                results = session.run(query, {
                    "entities": entity_names,
                    "top_k": top_k
                })
                return [(r["name"], r["relevance"]) for r in results]
        except Exception as e:
            logger.error(f"查找相关实体失败: {e}")
            return []

    def get_entity_graph(self, limit: int = 500) -> Dict[str, Set[str]]:
        """获取完整实体关系图"""
        query = """
        MATCH (source:Entity)-[r:RELATION]->(target:Entity)
        RETURN source.name AS source, target.name AS target
        LIMIT $limit
        """
        try:
            graph: Dict[str, Set[str]] = {}
            with self._get_session() as session:
                results = session.run(query, {"limit": limit})
                for r in results:
                    source = r["source"]
                    target = r["target"]
                    if source not in graph:
                        graph[source] = set()
                    graph[source].add(target)
            return graph
        except Exception as e:
            logger.error(f"获取实体关系图失败: {e}")
            return {}

    # ========== 社区操作 ==========

    def save_community(self, community_id: int, entities: List[str],
                       summary: str = "", keywords: List[str] = None,
                       level: int = 0, density: float = 0.0):
        """保存社区"""
        query = """
        MERGE (c:Community {community_id: $community_id})
        SET c.entities = $entities,
            c.summary = $summary,
            c.keywords = $keywords,
            c.level = $level,
            c.density = $density,
            c.updated_at = datetime()
        """
        try:
            with self._get_session() as session:
                session.run(query, {
                    "community_id": community_id,
                    "entities": entities,
                    "summary": summary,
                    "keywords": keywords or [],
                    "level": level,
                    "density": density
                })
        except Exception as e:
            logger.error(f"保存社区失败 {community_id}: {e}")

    def get_community(self, community_id: int) -> Optional[Dict]:
        """获取社区"""
        query = """
        MATCH (c:Community {community_id: $community_id})
        RETURN c.community_id AS community_id,
               c.entities AS entities,
               c.summary AS summary,
               c.keywords AS keywords,
               c.level AS level,
               c.density AS density
        """
        try:
            with self._get_session() as session:
                result = session.run(query, {"community_id": community_id}).single()
                if result:
                    return {
                        "community_id": result["community_id"],
                        "entities": result["entities"],
                        "summary": result["summary"],
                        "keywords": result["keywords"],
                        "level": result["level"],
                        "density": result["density"]
                    }
            return None
        except Exception as e:
            logger.error(f"获取社区失败 {community_id}: {e}")
            return None

    def get_all_communities(self) -> List[Dict]:
        """获取所有社区"""
        query = """
        MATCH (c:Community)
        RETURN c.community_id AS community_id,
               c.entities AS entities,
               c.summary AS summary,
               c.keywords AS keywords,
               c.level AS level,
               c.density AS density
        ORDER BY c.level, c.community_id
        """
        try:
            with self._get_session() as session:
                return [dict(r) for r in session.run(query)]
        except Exception as e:
            logger.error(f"获取所有社区失败: {e}")
            return []

    def delete_graph(self):
        """删除所有图谱数据"""
        query = """
        MATCH (n)
        DETACH DELETE n
        """
        try:
            with self._get_session() as session:
                session.run(query)
            logger.info("Neo4j 图数据已清空")
        except Exception as e:
            logger.error(f"清空图数据失败: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取图统计信息"""
        query = """
        MATCH (e:Entity)
        WITH count(e) AS entity_count
        MATCH ()-[r:RELATION]->()
        WITH entity_count, count(r) AS relation_count
        MATCH (c:Community)
        RETURN entity_count, relation_count, count(c) AS community_count
        """
        try:
            with self._get_session() as session:
                result = session.run(query).single()
                return {
                    "entity_count": result["entity_count"] if result else 0,
                    "relation_count": result["relation_count"] if result else 0,
                    "community_count": result["community_count"] if result else 0,
                    "connected": self._connected
                }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e)}

    def close(self):
        """关闭连接"""
        if self._driver:
            self._driver.close()
            self._connected = False
            logger.info("Neo4j 连接已关闭")


# 全局单例
_neo4j_store = None


def get_neo4j_store() -> Neo4jStore:
    """获取 Neo4j 存储实例"""
    global _neo4j_store
    if _neo4j_store is None:
        _neo4j_store = Neo4jStore()
    return _neo4j_store


__all__ = ['Neo4jStore', 'get_neo4j_store', 'StoredEntity', 'StoredRelation', 'get_neo4j_config']