# app/service/core/graphrag/neo4j_store.py
"""Neo4j 图存储"""

import os
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("neo4j 模块未安装")


@dataclass
class StoredEntity:
    name: str
    entity_type: str
    frequency: int = 1
    summary: str = ""

    def to_dict(self) -> Dict:
        return {"name": self.name, "type": self.entity_type, "frequency": self.frequency}


@dataclass
class StoredRelation:
    source: str
    target: str
    relation_type: str
    weight: float = 1.0
    evidence: str = ""


class Neo4jStore:
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
        self._driver = None
        self._connected = False

        if not NEO4J_AVAILABLE:
            return

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")

        if password:
            try:
                self._driver = GraphDatabase.driver(uri, auth=(user, password))
                self._driver.verify_connectivity()
                self._connected = True
                logger.info(f"Neo4j 连接成功")
            except Exception as e:
                logger.error(f"Neo4j 连接失败: {e}")

    def _get_session(self):
        return self._driver.session() if self._driver else None

    def init_constraints(self):
        if not self._connected:
            return
        try:
            with self._get_session() as session:
                session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        except Exception as e:
            logger.debug(f"约束已存在: {e}")

    def save_entities_batch(self, entities: List[StoredEntity]) -> int:
        if not self._connected or not entities:
            return 0
        query = """
        UNWIND $entities AS e
        MERGE (entity:Entity {name: e.name})
        SET entity.type = e.type,
            entity.frequency = COALESCE(entity.frequency, 0) + e.frequency,
            entity.summary = e.summary
        RETURN count(entity)
        """
        try:
            with self._get_session() as session:
                result = session.run(query, {"entities": [e.__dict__ for e in entities]})
                return result.single()[0] if result else 0
        except Exception as e:
            logger.error(f"保存实体失败: {e}")
            return 0

    def save_relations_batch(self, relations: List[StoredRelation]) -> int:
        if not self._connected or not relations:
            return 0
        query = """
        UNWIND $relations AS r
        MATCH (s:Entity {name: r.source})
        MATCH (t:Entity {name: r.target})
        MERGE (s)-[rel:RELATION {type: r.relation_type}]->(t)
        SET rel.weight = COALESCE(rel.weight, 0) + r.weight,
            rel.evidence = r.evidence
        RETURN count(rel)
        """
        try:
            with self._get_session() as session:
                result = session.run(query, {"relations": [r.__dict__ for r in relations]})
                return result.single()[0] if result else 0
        except Exception as e:
            logger.error(f"保存关系失败: {e}")
            return 0

    def get_neighbors(self, entity_name: str, max_depth: int = 2) -> List[Tuple[str, int, str]]:
        if not self._connected:
            return []
        query = f"""
        MATCH path = (start:Entity {{name: $name}})-[*1..{max_depth}]-(neighbor:Entity)
        WHERE start <> neighbor
        RETURN DISTINCT neighbor.name AS name, length(path) AS depth, 'RELATED_TO' AS rel
        LIMIT 100
        """
        try:
            with self._get_session() as session:
                results = session.run(query, {"name": entity_name})
                return [(r["name"], r["depth"], r["rel"]) for r in results]
        except Exception as e:
            return []

    def get_path_between(self, source: str, target: str, max_depth: int = 3) -> List[Dict]:
        if not self._connected:
            return []
        query = f"""
        MATCH path = shortestPath((s:Entity {{name: $source}})-[*1..{max_depth}]-(t:Entity {{name: $target}}))
        RETURN [node in nodes(path) | node.name] AS nodes,
               [rel in relationships(path) | rel.type] AS relations,
               length(path) AS length
        """
        try:
            with self._get_session() as session:
                result = session.run(query, {"source": source, "target": target}).single()
                if result:
                    return [{"nodes": result["nodes"], "relations": result["relations"], "length": result["length"]}]
            return []
        except Exception:
            return []

    def get_all_communities(self) -> List[Dict]:
        if not self._connected:
            return []
        try:
            with self._get_session() as session:
                results = session.run("MATCH (c:Community) RETURN c.community_id AS id, c.entities AS entities, c.summary AS summary, c.keywords AS keywords")
                return [{"community_id": r["id"], "entities": r["entities"] or [], "summary": r["summary"] or "", "keywords": r["keywords"] or []} for r in results]
        except Exception:
            return []

    def save_community(self, community_id: int, entities: List[str], summary: str = "", keywords: List[str] = None):
        if not self._connected:
            return
        query = """
        MERGE (c:Community {community_id: $id})
        SET c.entities = $entities, c.summary = $summary, c.keywords = $keywords, c.updated_at = datetime()
        """
        try:
            with self._get_session() as session:
                session.run(query, {"id": community_id, "entities": entities, "summary": summary, "keywords": keywords or []})
        except Exception as e:
            logger.error(f"保存社区失败: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        if not self._connected:
            return {"entity_count": 0, "relation_count": 0, "community_count": 0, "connected": False}
        try:
            with self._get_session() as session:
                e = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()
                r = session.run("MATCH ()-[rel:RELATION]->() RETURN count(rel) AS c").single()
                c = session.run("MATCH (c:Community) RETURN count(c) AS c").single()
                return {"entity_count": e["c"] if e else 0, "relation_count": r["c"] if r else 0, "community_count": c["c"] if c else 0, "connected": True}
        except Exception as e:
            return {"error": str(e), "connected": False}

    def delete_graph(self):
        if not self._connected:
            return
        try:
            with self._get_session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.info("图数据已清空")
        except Exception as e:
            logger.error(f"清空失败: {e}")

    def close(self):
        if self._driver:
            self._driver.close()
            self._connected = False


_neo4j_store = None


def get_neo4j_store() -> Neo4jStore:
    global _neo4j_store
    if _neo4j_store is None:
        _neo4j_store = Neo4jStore()
    return _neo4j_store


__all__ = ['Neo4jStore', 'get_neo4j_store', 'StoredEntity', 'StoredRelation']