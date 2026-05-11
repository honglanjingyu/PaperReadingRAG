"""
GraphRAG 检索器 - 支持实体为中心的检索和社区检索，使用 Neo4j
"""

import logging
import os
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class GraphRetriever:
    """
    GraphRAG 检索器 - 使用 Neo4j 进行图查询
    """

    def __init__(self, vector_search_service=None, llm_service=None, neo4j_store=None):
        """
        初始化 GraphRAG 检索器

        Args:
            vector_search_service: 向量搜索服务
            llm_service: LLM 服务
            neo4j_store: Neo4j 存储实例
        """
        from app.service.core.vector_store import get_vector_search_service
        from app.service.core.llm import get_llm_service
        from .neo4j_store import get_neo4j_store

        self.vector_search = vector_search_service or get_vector_search_service()
        self.llm_service = llm_service or get_llm_service()
        self.neo4j = neo4j_store or get_neo4j_store()

        logger.info("GraphRetriever 初始化完成 (使用 Neo4j)")

    def extract_entities_from_question(self, question: str) -> List[str]:
        """
        从用户问题中提取实体

        Args:
            question: 用户问题

        Returns:
            实体列表
        """
        entities = []

        # 尝试从 Neo4j 中匹配实体
        all_entities = self.neo4j.get_all_entities(limit=100)
        entity_names = [e.name for e in all_entities]

        # 在问题中查找已知实体
        for entity in entity_names:
            if entity and entity in question:
                entities.append(entity)

        # 使用正则提取额外实体
        patterns = [
            r"关于([\u4e00-\u9fff]{2,})",
            r"([\u4e00-\u9fff]{2,})(?:公司|产品|技术|方案|策略)",
            r"什么是([\u4e00-\u9fff]{2,})",
            r"([\u4e00-\u9fff]{2,})的(?:业绩|表现|情况)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, question)
            for match in matches:
                if match not in entities and len(match) >= 2:
                    entities.append(match)

        logger.info(f"从问题中提取实体: {entities[:10]}")
        return entities[:10]

    def entity_search(
            self,
            entities: List[str],
            index_name: str,
            top_k: int = 10
    ) -> List[Dict]:
        """
        基于实体的检索

        Args:
            entities: 实体列表
            index_name: 索引名称
            top_k: 返回数量

        Returns:
            相关文档列表
        """
        if not entities:
            return []

        # 构建实体查询
        query = " ".join(entities)

        # 使用向量检索
        from app.service.core.embedding import get_embedding_service
        embedding_service = get_embedding_service()
        query_vector = embedding_service.generate_embedding(query)

        if not query_vector:
            return []

        results = self.vector_search.similarity_search(
            query_vector=query_vector,
            index_name=index_name,
            top_k=top_k,
            similarity_threshold=0.3
        )

        logger.info(f"实体检索完成: {len(entities)} 个实体 -> {len(results)} 个结果")
        return results

    def expand_with_entity_relations(
            self,
            entities: List[str],
            max_depth: int = 1
    ) -> List[str]:
        """
        基于实体关系扩展查询（使用 Neo4j）

        Args:
            entities: 原始实体列表
            max_depth: 最大扩展深度

        Returns:
            扩展后的实体列表
        """
        expanded = set(entities)

        for entity in entities:
            neighbors = self.neo4j.get_neighbors(entity, max_depth)
            for neighbor_name, depth, _ in neighbors:
                if depth <= max_depth:
                    expanded.add(neighbor_name)

        result = list(expanded)
        if len(expanded) > len(entities):
            logger.info(f"实体扩展: {len(entities)} -> {len(result)}")

        return result

    def community_search(
            self,
            entities: List[str],
            index_name: str,
            top_k: int = 5
    ) -> List[Dict]:
        """
        基于社区的检索（使用 Neo4j 社区数据）

        Args:
            entities: 实体列表
            index_name: 索引名称
            top_k: 返回数量

        Returns:
            相关文档列表
        """
        if not entities:
            return []

        # 找到与这些实体相关的社区
        all_communities = self.neo4j.get_all_communities()

        # 计算每个社区的相关性
        community_scores = []
        for comm in all_communities:
            comm_entities = set(comm.get("entities", []))
            overlap = len(set(entities) & comm_entities)
            if overlap > 0:
                score = overlap / len(comm_entities) if comm_entities else 0
                community_scores.append((comm, score))

        community_scores.sort(key=lambda x: x[1], reverse=True)

        if not community_scores:
            return []

        # 使用最相关社区的实体进行检索
        top_community = community_scores[0][0]
        community_entities = top_community.get("entities", [])[:10]

        if community_entities:
            query = " ".join(community_entities)
            from app.service.core.embedding import get_embedding_service
            embedding_service = get_embedding_service()
            query_vector = embedding_service.generate_embedding(query)

            if query_vector:
                results = self.vector_search.similarity_search(
                    query_vector=query_vector,
                    index_name=index_name,
                    top_k=top_k,
                    similarity_threshold=0.3
                )
                logger.info(f"社区检索完成: community_id={top_community.get('community_id')} -> {len(results)} 个结果")
                return results

        return []

    def global_search(
            self,
            question: str,
            index_name: str,
            top_k: int = 8
    ) -> List[Dict]:
        """
        全局检索（基于社区摘要，使用 Neo4j）

        Args:
            question: 用户问题
            index_name: 索引名称
            top_k: 返回数量

        Returns:
            相关文档列表
        """
        # 获取所有社区摘要
        communities = self.neo4j.get_all_communities()

        if not communities:
            return []

        from app.service.core.embedding import get_embedding_service
        embedding_service = get_embedding_service()

        question_vector = embedding_service.generate_embedding(question)
        if not question_vector:
            return []

        # 计算每个社区摘要与问题的相关性
        relevance_scores = []
        for comm in communities:
            summary = comm.get("summary", "")
            if not summary:
                continue

            summary_vector = embedding_service.generate_embedding(summary[:500])
            if summary_vector:
                import numpy as np
                similarity = np.dot(question_vector, summary_vector) / (
                        np.linalg.norm(question_vector) * np.linalg.norm(summary_vector) + 1e-8
                )
                relevance_scores.append((comm, similarity))

        relevance_scores.sort(key=lambda x: x[1], reverse=True)

        if not relevance_scores:
            return []

        # 从最相关的社区中检索文档
        all_results = []
        seen_ids = set()

        for comm, score in relevance_scores[:5]:
            community_entities = comm.get("entities", [])[:10]
            if community_entities:
                query = " ".join(community_entities)
                query_vector = embedding_service.generate_embedding(query)

                if query_vector:
                    results = self.vector_search.similarity_search(
                        query_vector=query_vector,
                        index_name=index_name,
                        top_k=top_k // 2,
                        similarity_threshold=0.25
                    )

                    for r in results:
                        doc_id = r.get("_id", r.get("id", ""))
                        if doc_id and doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            r["_community_score"] = score
                            all_results.append(r)

        all_results.sort(key=lambda x: x.get("_score", 0), reverse=True)
        logger.info(f"全局检索完成: 返回 {len(all_results[:top_k])} 个结果")
        return all_results[:top_k]

    def hybrid_graph_search(
            self,
            question: str,
            entities: List[str],
            communities: Dict,
            summaries: Dict[int, str],
            entity_graph: Dict[str, Set[str]],
            index_name: str,
            top_k: int = 8
    ) -> Tuple[List[Dict], Dict]:
        """
        混合图检索（结合实体、社区和全局检索）

        Returns:
            (results, metadata): 检索结果和元数据
        """
        all_results = []
        search_sources = []

        # 1. 实体检索
        if entities:
            entity_results = self.entity_search(entities, index_name, top_k * 2)
            for r in entity_results:
                r["_source"] = "entity"
            all_results.extend(entity_results)
            search_sources.append(f"entity({len(entities)})")

        # 2. 扩展实体检索（使用 Neo4j 关系）
        if entities:
            expanded_entities = self.expand_with_entity_relations(entities)
            if len(expanded_entities) > len(entities):
                expanded_results = self.entity_search(expanded_entities, index_name, top_k)
                for r in expanded_results:
                    r["_source"] = "expanded_entity"
                all_results.extend(expanded_results)
                search_sources.append(f"expanded_entity({len(expanded_entities)})")

        # 3. 社区检索
        if entities:
            community_results = self.community_search(entities, index_name, top_k)
            for r in community_results:
                r["_source"] = "community"
            all_results.extend(community_results)
            search_sources.append("community")

        # 4. 全局检索（基于摘要）
        global_results = self.global_search(question, index_name, top_k)
        for r in global_results:
            r["_source"] = "global"
        all_results.extend(global_results)
        search_sources.append("global")

        # 去重
        seen_ids = set()
        unique_results = []
        for r in all_results:
            doc_id = r.get("_id", r.get("id", ""))
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(r)

        # 按分数排序
        unique_results.sort(key=lambda x: (
                x.get("_score", 0) * (1.2 if x.get("_source") == "global" else 1.0)
        ), reverse=True)

        metadata = {
            "search_sources": search_sources,
            "total_recalled": len(unique_results),
            "entities_extracted": entities,
            "graph_enabled": True
        }

        return unique_results[:top_k], metadata