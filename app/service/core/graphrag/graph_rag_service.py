# app/service/core/graphrag/graph_rag_service.py
"""
GraphRAG 服务 - 整合所有组件，使用 Neo4j 作为图存储
"""

import os
import logging
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .entity_extractor import EntityExtractor, Entity, Relation
from .community_detection import CommunityDetector
from .hierarchical_summarizer import HierarchicalSummarizer
from .graph_retriever import GraphRetriever
from .neo4j_store import (
    Neo4jStore, get_neo4j_store,
    StoredEntity, StoredRelation
)
from .graph_cache import get_graph_cache  # 新增导入

logger = logging.getLogger(__name__)


def get_graph_config() -> Dict[str, Any]:
    """从环境变量获取 GraphRAG 配置"""
    return {
        "enabled": os.getenv("GRAPH_RAG_ENABLED", "true").lower() == "true",
        "default_top_k": int(os.getenv("GRAPH_RAG_TOP_K", "8")),
        "max_context_length": int(os.getenv("GRAPH_RAG_MAX_CONTEXT_LENGTH", "4000")),
        "max_documents": int(os.getenv("GRAPH_RAG_MAX_DOCS", "50")),
        "use_llm_entity_extract": os.getenv("ENTITY_EXTRACT_USE_LLM", "true").lower() == "true",
        "min_entity_frequency": int(os.getenv("ENTITY_MIN_FREQUENCY", "1")),
        "min_community_size": int(os.getenv("COMMUNITY_MIN_SIZE", "2")),
        "max_hierarchy_levels": int(os.getenv("HIERARCHICAL_MAX_LEVELS", "3")),
        "use_louvain": os.getenv("COMMUNITY_USE_LOUVAIN", "true").lower() == "true",
        "enable_cache": os.getenv("ENABLE_GRAPH_CACHE", "true").lower() == "true",  # 新增
    }


class GraphRAGService:
    """
    GraphRAG 服务 - 使用 Neo4j 作为图存储
    提供完整的 GraphRAG 能力：知识图谱构建、社区发现、层次摘要、图检索
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

        # 加载配置
        self.config = get_graph_config()

        if not self.config["enabled"]:
            logger.info("GraphRAG 已禁用")
            return

        # 初始化组件
        from app.service.core.llm import get_llm_service
        from app.service.core.vector_store import get_vector_search_service

        self.llm_service = get_llm_service()
        self.vector_search = get_vector_search_service()

        # 初始化 Neo4j 存储
        self.neo4j = get_neo4j_store()

        # 初始化缓存
        self.cache = get_graph_cache() if self.config["enable_cache"] else None

        # 初始化约束和索引（首次运行时）
        try:
            self.neo4j.init_constraints_and_indexes()
        except Exception as e:
            logger.warning(f"初始化 Neo4j 约束失败: {e}")

        # 初始化各组件
        self.entity_extractor = EntityExtractor(
            use_llm=self.config["use_llm_entity_extract"],
            llm_service=self.llm_service,
            min_frequency=self.config["min_entity_frequency"]
        )

        self.community_detector = CommunityDetector(
            use_louvain=self.config["use_louvain"],
            min_community_size=self.config["min_community_size"]
        )

        self.summarizer = HierarchicalSummarizer(
            llm_service=self.llm_service,
            max_context_length=self.config["max_context_length"]
        )

        self.retriever = GraphRetriever(
            vector_search_service=self.vector_search,
            llm_service=self.llm_service,
            neo4j_store=self.neo4j
        )

        self._is_initialized = True

        # 打印统计信息
        stats = self.neo4j.get_statistics()
        logger.info(f"GraphRAGService 初始化完成, Neo4j 统计: {stats}")

    def _get_documents_version(self, user_level: str = None) -> str:
        """
        获取当前文档集合的版本号（用于缓存验证）
        """
        from app.service.core.vector_store import get_vector_store
        from pymilvus import Collection

        store = get_vector_store()
        index_name = "rag_documents"

        if not store.index_exists(index_name):
            return "empty"

        try:
            collection = Collection(index_name)
            collection.load()

            # 获取文档名称和用户等级
            results = collection.query(
                expr="",
                output_fields=["docnm", "user_level", "create_timestamp_flt"],
                limit=5000
            )

            # 根据用户等级过滤
            doc_info = {}
            for r in results:
                docnm = r.get("docnm", "")
                doc_level = r.get("user_level", "normal")
                ts = r.get("create_timestamp_flt", 0)

                # 只包含用户等级可访问的文档
                if user_level:
                    level_priority = {"normal": 1, "admin": 2, "owner": 3}
                    doc_priority = level_priority.get(doc_level, 1)
                    user_priority = level_priority.get(user_level, 1)
                    if user_priority < doc_priority:
                        continue

                if docnm:
                    doc_info[docnm] = max(doc_info.get(docnm, 0), ts)

            # 生成版本哈希
            version_str = json.dumps(sorted(doc_info.items()), sort_keys=True)
            return hashlib.md5(version_str.encode()).hexdigest()[:16]

        except Exception as e:
            logger.error(f"获取文档版本失败: {e}")
            return "unknown"

    def build_knowledge_graph(
            self,
            documents: List[str],
            doc_names: List[str] = None,
            graph_id: str = None,
            use_cache: bool = True,
            user_level: str = None,
            force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        从文档构建知识图谱并存入 Neo4j

        Args:
            documents: 文档文本列表
            doc_names: 文档名称列表
            graph_id: 图谱标识
            use_cache: 是否使用缓存
            user_level: 用户等级
            force_rebuild: 是否强制重建（忽略缓存）

        Returns:
            知识图谱数据
        """
        if not self.config["enabled"]:
            return {"success": False, "error": "GraphRAG 未启用"}

        # ========== 缓存检查 ==========
        if use_cache and self.cache and not force_rebuild:
            # 获取当前文档版本
            current_version = self._get_documents_version(user_level)

            cached_data = self.cache.get(user_level)
            if cached_data:
                cached_version = cached_data.get("_documents_version")
                if cached_version == current_version:
                    logger.info(f"图谱缓存命中: user_level={user_level}, version={current_version}")
                    return cached_data
                else:
                    logger.info(f"文档版本已变更: {cached_version} -> {current_version}")

        # ========== 原有构建逻辑 ==========
        if not documents:
            return {"success": False, "error": "没有提供文档"}

        # 限制文档数量
        if len(documents) > self.config["max_documents"]:
            logger.warning(f"文档数量 {len(documents)} 超过限制 {self.config['max_documents']}，将截断")
            documents = documents[:self.config["max_documents"]]

        # 生成图谱 ID
        combined_text = "\n\n".join(documents)
        if graph_id is None:
            graph_id = hashlib.md5(combined_text.encode()).hexdigest()[:16]

        logger.info(f"开始构建知识图谱，文档数: {len(documents)}, graph_id={graph_id}, user_level={user_level}")

        # 1. 提取实体和关系
        all_entities: Dict[str, Entity] = {}
        all_relations: Dict[str, Relation] = {}

        for i, text in enumerate(documents):
            if not text.strip():
                continue

            entities, relations = self.entity_extractor.extract_from_text(
                text, doc_names[i] if doc_names and i < len(doc_names) else f"doc_{i}"
            )

            for e in entities:
                if e.name in all_entities:
                    all_entities[e.name].frequency += e.frequency
                    all_entities[e.name].mentions.extend(e.mentions)
                else:
                    all_entities[e.name] = e

            for r in relations:
                key = f"{r.source}|{r.target}|{r.relation_type}"
                if key in all_relations:
                    all_relations[key].weight += r.weight
                else:
                    all_relations[key] = r

        # 2. 存入 Neo4j
        stored_entities = [
            StoredEntity(
                name=e.name,
                entity_type=e.type,
                frequency=e.frequency,
                summary=f"{e.type}类型实体，出现{e.frequency}次"
            )
            for e in all_entities.values()
            if e.frequency >= self.config["min_entity_frequency"]
        ]

        self.neo4j.save_entities_batch(stored_entities)

        stored_relations = [
            StoredRelation(
                source=r.source,
                target=r.target,
                relation_type=r.relation_type,
                weight=r.weight,
                evidence=getattr(r, 'evidence', '')
            )
            for r in all_relations.values()
        ]
        self.neo4j.save_relations_batch(stored_relations)

        # 3. 社区发现
        entities_dict = [e.to_dict() for e in all_entities.values()]
        relations_dict = [r.to_dict() for r in all_relations.values()]

        communities = self.community_detector.detect_communities(
            entities_dict, relations_dict
        )

        # 4. 生成社区摘要并存入 Neo4j
        entity_details = {e.name: e.to_dict() for e in all_entities.values()}
        summaries = self.summarizer.generate_summaries(
            communities, entity_details, combined_text[:self.config["max_context_length"]]
        )

        for comm_id, summary in summaries.items():
            if comm_id in communities:
                comm = communities[comm_id]
                self.neo4j.save_community(
                    community_id=int(comm_id) if isinstance(comm_id, int) else hash(comm_id),
                    entities=comm.entities[:20],
                    summary=summary.summary,
                    keywords=summary.key_insights[:10],
                    level=getattr(comm, 'level', 0),
                    density=getattr(comm, 'density', 0.0)
                )

        # 5. 获取统计信息
        stats = self.neo4j.get_statistics()

        # 6. 构建返回结果
        result = {
            "success": True,
            "graph_id": graph_id,
            "total_documents": len(documents),
            "statistics": stats,
            "entities": [e.to_dict() for e in stored_entities[:50]],
            "relations": [r.to_dict() for r in stored_relations[:50]],
            "communities": [
                {
                    "id": c.id,
                    "size": c.size,
                    "density": c.density,
                    "keywords": c.keywords,
                    "summary": c.summary
                }
                for c in communities.values()
            ],
            "summaries": {
                str(cid): {
                    "title": s.title,
                    "summary": s.summary,
                    "key_insights": s.key_insights
                }
                for cid, s in summaries.items()
            },
            "_documents_version": self._get_documents_version(user_level),  # 添加版本号
            "_cached_at": datetime.now().isoformat()
        }

        # ========== 缓存结果 ==========
        if use_cache and self.cache and result.get("success"):
            self.cache.set(result, user_level)
            logger.info(f"知识图谱已缓存: user_level={user_level}")

        logger.info(f"知识图谱构建完成: 实体={stats.get('entity_count', 0)}, "
                    f"关系={stats.get('relation_count', 0)}, "
                    f"社区={stats.get('community_count', 0)}")
        return result

    def get_cached_graph(self, user_level: str = None) -> Optional[Dict]:
        """获取缓存的图谱数据"""
        if not self.config["enable_cache"] or not self.cache:
            return None

        cached = self.cache.get(user_level)
        if cached:
            logger.info(f"返回缓存图谱: user_level={user_level}")
        return cached

    def invalidate_cache(self, user_level: str = None):
        """使图谱缓存失效"""
        if not self.config["enable_cache"] or not self.cache:
            return

        self.cache.invalidate(user_level)
        logger.info(f"图谱缓存已失效: user_level={user_level or 'all'}")

    def graph_search(
            self,
            question: str,
            graph_id: str = None,
            index_name: str = None,
            top_k: int = None
    ) -> Tuple[List[Dict], Dict]:
        """使用知识图谱进行检索"""
        if not self.config["enabled"]:
            return self._fallback_search(question, index_name, top_k)

        top_k = top_k or self.config["default_top_k"]
        index_name = index_name or os.getenv("VECTOR_INDEX_NAME", "rag_documents")

        # 1. 从问题中提取实体
        entities = self.retriever.extract_entities_from_question(question)

        # 2. 获取社区数据
        communities = {}
        for comm in self.neo4j.get_all_communities():
            communities[comm["community_id"]] = {
                "entities": comm.get("entities", []),
                "size": len(comm.get("entities", [])),
                "density": comm.get("density", 0)
            }

        # 3. 获取摘要
        summaries = {}
        for comm in self.neo4j.get_all_communities():
            if comm.get("summary"):
                summaries[comm["community_id"]] = comm["summary"]

        # 4. 获取实体关系图
        entity_graph = self.neo4j.get_entity_graph()

        # 5. 执行混合图检索
        results, metadata = self.retriever.hybrid_graph_search(
            question=question,
            entities=entities,
            communities=communities,
            summaries=summaries,
            entity_graph=entity_graph,
            index_name=index_name,
            top_k=top_k
        )

        metadata["graph_enabled"] = True
        metadata["graph_statistics"] = self.neo4j.get_statistics()

        return results, metadata

    def _fallback_search(
            self,
            question: str,
            index_name: str,
            top_k: int
    ) -> Tuple[List[Dict], Dict]:
        """降级到普通检索"""
        from app.service.core.rag.search import enhanced_search_with_hybrid_and_rerank

        result = enhanced_search_with_hybrid_and_rerank(
            question=question,
            index_name=index_name or os.getenv("VECTOR_INDEX_NAME", "rag_documents"),
            top_k=top_k or self.config["default_top_k"],
            recall_k=(top_k or self.config["default_top_k"]) * 2,
            enable_rerank=True,
            enable_query_rewrite=True,
            verbose=False
        )

        results = result.get("results", [])
        metadata = {
            "graph_enabled": False,
            "fallback": True,
            "reason": "GraphRAG disabled or error"
        }

        return results, metadata

    def get_related_entities(
            self,
            entity_names: List[str],
            top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """获取与给定实体相关的实体"""
        return self.neo4j.find_related_entities(entity_names, top_k)

    def get_entity_info(self, entity_name: str) -> Optional[StoredEntity]:
        """获取实体信息"""
        return self.neo4j.get_entity(entity_name)

    def get_entity_neighbors(self, entity_name: str, max_depth: int = 2) -> List[Tuple[str, int, str]]:
        """获取实体的邻居"""
        return self.neo4j.get_neighbors(entity_name, max_depth)

    def get_path(self, source: str, target: str, max_depth: int = 3) -> List[Dict]:
        """获取两个实体之间的路径"""
        return self.neo4j.get_path_between(source, target, max_depth)

    def delete_graph(self):
        """删除所有图谱数据并清理缓存"""
        self.neo4j.delete_graph()
        self.invalidate_cache()  # 清理缓存
        logger.info("图谱数据已清空，缓存已失效")

    def get_statistics(self) -> Dict[str, Any]:
        """获取图统计信息"""
        return self.neo4j.get_statistics()

    def clear_cache(self):
        """清除缓存"""
        self.invalidate_cache()

    def close(self):
        """关闭 Neo4j 连接"""
        self.neo4j.close()


def get_graph_rag_service() -> GraphRAGService:
    """获取 GraphRAG 服务单例"""
    return GraphRAGService()


__all__ = ['GraphRAGService', 'get_graph_rag_service', 'get_graph_config']