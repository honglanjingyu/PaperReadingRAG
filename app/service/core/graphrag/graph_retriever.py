"""
GraphRAG 检索器 - 支持实体为中心的检索和社区检索，使用 Neo4j
"""

import logging
import os
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

# 添加 RRF 融合的导入
from app.service.core.retrieval.base import ScoreMerger

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

        # RRF 参数
        self._rrf_k = int(os.getenv("RRF_K", "60"))

        # 检索源权重配置
        self._source_weights = {
            "entity": float(os.getenv("GRAPH_RRF_ENTITY_WEIGHT", "1.0")),
            "expanded_entity": float(os.getenv("GRAPH_RRF_EXPANDED_WEIGHT", "0.8")),
            "community": float(os.getenv("GRAPH_RRF_COMMUNITY_WEIGHT", "0.9")),
            "global": float(os.getenv("GRAPH_RRF_GLOBAL_WEIGHT", "1.0")),
        }

        logger.info(f"GraphRetriever RRF 配置: k={self._rrf_k}, weights={self._source_weights}")

        # 加载配置
        self._load_config()

        logger.info("GraphRetriever 初始化完成 (使用 Neo4j)")

    def _load_config(self):
        """从配置文件加载别名映射"""
        try:
            from app.service.graphrag_configs import get_graphrag_config
            self.config = get_graphrag_config()
            self.entity_aliases = self.config.get_all_aliases()
            logger.info(f"GraphRetriever 加载别名映射: {len(self.entity_aliases)} 个")
        except Exception as e:
            logger.warning(f"加载 GraphRAG 配置失败: {e}，将使用空别名映射")
            self.entity_aliases = {}

    def extract_entities_from_question(self, question: str) -> List[str]:
        """
        从用户问题中提取实体（支持别名匹配）

        Args:
            question: 用户问题

        Returns:
            实体名称列表
        """
        import re
        entities = set()

        # 1. 获取所有已知实体名称
        all_entities = self.neo4j.get_all_entities(limit=200)
        entity_names = sorted([e.name for e in all_entities], key=len, reverse=True)

        # 2. 精确匹配
        for entity in entity_names:
            if entity and entity in question:
                entities.add(entity)

        # 3. 别名匹配（从配置加载）
        for alias, full_name in self.entity_aliases.items():
            if alias in question:
                if full_name in entity_names:
                    entities.add(full_name)
                else:
                    entities.add(alias)

        # 4. ========== 从 entity_extraction.yaml 加载实体提取模式 ==========
        try:
            from app.service.graphrag_configs import get_graphrag_config
            config = get_graphrag_config()
            patterns = config.get_entity_extraction_patterns()
            min_length = config.get_min_entity_length()
            max_entities = config.get_max_extracted_entities()
        except Exception as e:
            logger.debug(f"从配置加载实体提取模式失败: {e}")
            min_length = 2
            max_entities = 20

        # 应用配置的模式提取实体
        for pattern in patterns:
            try:
                matches = re.findall(pattern, question)
                for match in matches:
                    if match and len(match) >= min_length:
                        entities.add(match)
            except re.error as e:
                logger.warning(f"正则表达式错误: {pattern}, 错误: {e}")
                continue

        # 5. 限制实体数量
        if len(entities) > max_entities:
            # 优先保留长度较短的实体（更可能是精确实体）
            entities = set(sorted(entities, key=len)[:max_entities])

        # 6. 过滤太短的实体
        entities = {e for e in entities if len(e) >= min_length}

        logger.info(f"从问题中提取实体: {list(entities)}")
        return list(entities)

    def entity_search(
            self,
            entities: List[str],
            index_name: str,
            top_k: int = 10,
            user_level: str = None
    ) -> List[Dict]:
        """
        基于实体的检索

        Args:
            entities: 实体列表
            index_name: 索引名称
            top_k: 返回数量
            user_level: 用户等级

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
            similarity_threshold=0.3,
            user_level=user_level
        )

        # 标记检索源和原始排名（用于 RRF）
        for rank, r in enumerate(results, 1):
            r["_source"] = "entity"
            r["_source_name"] = "entity"
            r["_original_rank"] = rank
            if "score" not in r:
                r["score"] = r.get("_score", 0)

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
            top_k: int = 5,
            user_level: str = None
    ) -> List[Dict]:
        """
        基于社区的检索（使用 Neo4j 社区数据）

        Args:
            entities: 实体列表
            index_name: 索引名称
            top_k: 返回数量
            user_level: 用户等级

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
                    similarity_threshold=0.3,
                    user_level=user_level
                )

                # 标记检索源和原始排名（用于 RRF）
                for rank, r in enumerate(results, 1):
                    r["_source"] = "community"
                    r["_source_name"] = "community"
                    r["_original_rank"] = rank
                    r["_community_score"] = community_scores[0][1]

                logger.info(f"社区检索完成: community_id={top_community.get('community_id')} -> {len(results)} 个结果")
                return results

        return []

    def global_search(
            self,
            question: str,
            index_name: str,
            top_k: int = 8,
            user_level: str = None
    ) -> List[Dict]:
        """
        全局检索（基于社区摘要，使用 Neo4j）

        Args:
            question: 用户问题
            index_name: 索引名称
            top_k: 返回数量
            user_level: 用户等级

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
                        similarity_threshold=0.25,
                        user_level=user_level
                    )

                    for r in results:
                        doc_id = r.get("_id", r.get("id", ""))
                        if doc_id and doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            r["_source"] = "global"
                            r["_source_name"] = "global"
                            r["_community_score"] = score

        # 标记原始排名（用于 RRF）
        for rank, r in enumerate(all_results, 1):
            r["_original_rank"] = rank

        logger.info(f"全局检索完成: 返回 {len(all_results[:top_k])} 个结果")
        return all_results[:top_k]

    # ========== RRF 融合的核心方法 ==========

    def hybrid_graph_search(
            self,
            question: str,
            entities: List[str],
            communities: Dict,
            summaries: Dict[int, str],
            entity_graph: Dict[str, Set[str]],
            index_name: str,
            top_k: int = 8,
            user_level: str = None
    ) -> Tuple[List[Dict], Dict]:
        """
        混合图检索 - 使用 RRF 融合（结合实体、扩展实体、社区和全局检索）

        Returns:
            (results, metadata): 检索结果和元数据
        """
        results_lists = []
        source_names = []
        source_weights = []

        # 1. 实体检索
        if entities:
            entity_results = self.entity_search(entities, index_name, top_k * 3, user_level)
            if entity_results:
                results_lists.append(entity_results)
                source_names.append("entity")
                source_weights.append(self._source_weights["entity"])

        # 2. 扩展实体检索（使用 Neo4j 关系）
        if entities:
            expanded_entities = self.expand_with_entity_relations(entities)
            if len(expanded_entities) > len(entities):
                expanded_results = self.entity_search(expanded_entities, index_name, top_k * 2, user_level)
                if expanded_results:
                    results_lists.append(expanded_results)
                    source_names.append("expanded_entity")
                    source_weights.append(self._source_weights["expanded_entity"])

        # 3. 社区检索
        if entities:
            community_results = self.community_search(entities, index_name, top_k * 2, user_level)
            if community_results:
                results_lists.append(community_results)
                source_names.append("community")
                source_weights.append(self._source_weights["community"])

        # 4. 全局检索（基于摘要）
        global_results = self.global_search(question, index_name, top_k * 2, user_level)
        if global_results:
            results_lists.append(global_results)
            source_names.append("global")
            source_weights.append(self._source_weights["global"])

        # 如果没有结果，返回空
        if not results_lists:
            logger.warning("GraphRAG RRF 检索: 无任何检索结果")
            return [], {"search_sources": [], "total_recalled": 0}

        # 使用 RRF 融合所有结果
        rrf_scores = ScoreMerger.reciprocal_rank_fusion_with_weights(
            results_lists, source_weights, self._rrf_k
        )

        # 收集文档详情
        doc_map = {}
        for results, source_name in zip(results_lists, source_names):
            for doc in results:
                doc_id = doc.get("_id", doc.get("id", ""))
                if not doc_id:
                    continue

                if doc_id not in doc_map:
                    doc_map[doc_id] = {
                        **doc,
                        "rrf_score": 0.0,
                        "_search_types": [source_name],
                        "_source_names": [source_name]
                    }
                else:
                    if source_name not in doc_map[doc_id]["_search_types"]:
                        doc_map[doc_id]["_search_types"].append(source_name)
                        doc_map[doc_id]["_source_names"].append(source_name)

        # 应用 RRF 分数
        for doc_id, rrf_score in rrf_scores.items():
            if doc_id in doc_map:
                doc_map[doc_id]["rrf_score"] = rrf_score
                doc_map[doc_id]["score"] = rrf_score  # 统一 score 字段

        # 转换为列表并按 RRF 分数排序
        results = list(doc_map.values())
        results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)

        # 过滤低分结果
        min_rrf_score = float(os.getenv("RRF_MIN_SCORE", "0.005"))
        filtered = [r for r in results if r.get("rrf_score", 0) >= min_rrf_score]

        logger.info(f"RRF 混合检索完成: {len(filtered)} 个结果 (来自 {len(results_lists)} 个源)")

        # 构建相关社区摘要列表
        relevant_summaries = []
        if entities and summaries:
            for comm_id, summary in summaries.items():
                if comm_id in communities:
                    comm_entities = communities[comm_id].get("entities", [])
                    if any(e in entities for e in comm_entities):
                        relevant_summaries.append({
                            "title": f"Community {comm_id}",
                            "summary": summary,
                            "entities": comm_entities[:10]
                        })

        metadata = {
            "search_sources": source_names,
            "source_weights": source_weights,
            "total_recalled": len(filtered),
            "entities_extracted": entities,
            "graph_enabled": True,
            "rrf_k": self._rrf_k,
            "relevant_summaries": relevant_summaries[:5],
            "rrf_scores_top": [
                {"id": r.get("_id", "")[:16], "score": r.get("rrf_score", 0)}
                for r in filtered[:5]
            ] if filtered else []
        }

        return filtered[:top_k], metadata

    def hybrid_graph_search_with_paths(
            self,
            question: str,
            entities: List[str],
            communities: Dict,
            summaries: Dict[int, str],
            entity_graph: Dict[str, Set[str]],
            index_name: str,
            top_k: int = 8,
            user_level: str = None
    ) -> Tuple[List[Dict], Dict]:
        """
        混合图检索（带推理路径）- 使用 RRF 融合

        Args:
            question: 用户问题
            entities: 实体列表
            communities: 社区数据
            summaries: 社区摘要
            entity_graph: 实体关系图
            index_name: 索引名称
            top_k: 返回数量
            user_level: 用户等级

        Returns:
            (results, metadata): 检索结果和元数据
        """
        # 1. 执行 RRF 检索
        results, metadata = self.hybrid_graph_search(
            question=question,
            entities=entities,
            communities=communities,
            summaries=summaries,
            entity_graph=entity_graph,
            index_name=index_name,
            top_k=top_k,
            user_level=user_level
        )

        # 2. 提取推理路径
        reasoning_paths = []
        if len(entities) >= 2:
            for i in range(len(entities) - 1):
                path_result = self.neo4j.get_path_between(entities[i], entities[i + 1], max_depth=3)
                if path_result:
                    for path in path_result:
                        reasoning_paths.append({
                            "nodes": path.get("nodes", []),
                            "relations": path.get("relations", []),
                            "length": path.get("length", 0),
                            "relation": " → ".join(path.get("relations", [])) if path.get("relations") else "相关"
                        })

        # 去重
        unique_paths = []
        seen = set()
        for p in reasoning_paths:
            key = "→".join(p["nodes"])
            if key not in seen:
                seen.add(key)
                unique_paths.append(p)

        metadata["reasoning_paths"] = unique_paths[:5]

        return results, metadata


__all__ = ['GraphRetriever']