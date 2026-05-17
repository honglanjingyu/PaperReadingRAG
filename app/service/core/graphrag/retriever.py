# app/service/core/graphrag/retriever.py
"""GraphRAG 检索器和缓存 - 使用 YAML 配置"""

import os
import json
import hashlib
import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

# 导入配置管理器
from app.service.graphrag_configs import get_graphrag_config

logger = logging.getLogger(__name__)


class GraphCache:
    """知识图谱缓存 - Redis"""

    def __init__(self):
        self._redis = None
        self.cache_ttl = int(os.getenv("GRAPH_CACHE_TTL", "3600"))
        self.enabled = os.getenv("ENABLE_GRAPH_CACHE", "true").lower() == "true"

        if not self.enabled:
            return

        try:
            import redis
            self._redis = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                password=os.getenv("REDIS_PASSWORD") or None,
                decode_responses=True
            )
            self._redis.ping()
            logger.info("GraphCache Redis 连接成功")
        except Exception as e:
            logger.warning(f"Redis 不可用: {e}")
            self._redis = None

    def _key(self, user_level: str = None) -> str:
        return f"rag:cache:graph:{user_level or 'default'}"

    def get(self, user_level: str = None) -> Optional[Dict]:
        if not self._redis:
            return None
        try:
            cached = self._redis.get(self._key(user_level))
            return json.loads(cached) if cached else None
        except Exception:
            return None

    def set(self, data: Dict, user_level: str = None):
        if not self._redis:
            return
        try:
            self._redis.setex(self._key(user_level), self.cache_ttl, json.dumps(data))
        except Exception as e:
            logger.warning(f"缓存失败: {e}")

    def invalidate(self, user_level: str = None):
        if not self._redis:
            return
        try:
            if user_level:
                self._redis.delete(self._key(user_level))
            else:
                for key in self._redis.keys("rag:cache:graph:*"):
                    self._redis.delete(key)
            logger.info(f"缓存已失效: {user_level or 'all'}")
        except Exception as e:
            logger.warning(f"失效失败: {e}")


class GraphRetriever:
    """图检索器 - 使用 YAML 配置的实体提取和查询替换"""

    def __init__(self, vector_search_service=None, llm_service=None, neo4j_store=None):
        from app.service.core.embedding import get_embedding_service
        from app.service.core.vector_store import get_vector_search_service
        from .neo4j_store import get_neo4j_store

        self.embedding = get_embedding_service()
        self.vector_search = vector_search_service or get_vector_search_service()
        self.neo4j = neo4j_store or get_neo4j_store()
        self._rrf_k = int(os.getenv("RRF_K", "60"))

        # 加载配置
        self.config = get_graphrag_config()

        # 获取实体提取模式
        self.entity_extraction_patterns = self._compile_extraction_patterns()

        # 获取别名映射
        self.aliases = self.config.get_entity_aliases()
        self.case_insensitive = self.config.is_case_insensitive()

        # 获取查询替换规则
        self.query_replacements = self.config.get_query_replacements()
        self.query_replacement_enabled = self.config.is_query_replacement_enabled()

        # 实体长度限制
        self.min_entity_length = self.config.get_min_entity_length()
        self.max_entity_length = self.config.get_max_entity_length()
        self.max_extracted_entities = self.config.get_max_extracted_entities()

        logger.info(f"GraphRetriever 初始化: 实体模式数={len(self.entity_extraction_patterns)}, "
                    f"查询替换规则数={len(self.query_replacements)}")

    def _compile_extraction_patterns(self) -> List[re.Pattern]:
        """编译从问题中提取实体的正则表达式模式"""
        patterns = self.config.get_entity_extraction_patterns()
        compiled = []

        for pattern_str in patterns:
            try:
                # 处理 YAML 中的转义
                if pattern_str.startswith("r'") and pattern_str.endswith("'"):
                    pattern_str = pattern_str[2:-1]
                compiled.append(re.compile(pattern_str))
            except re.error as e:
                logger.warning(f"编译提取模式失败: {pattern_str} -> {e}")
        return compiled

    def apply_query_replacements(self, query: str) -> str:
        """应用查询替换规则"""
        if not self.query_replacement_enabled:
            return query

        result = query
        for alias, replacement in self.query_replacements:
            result = result.replace(alias, replacement)

        if result != query:
            logger.debug(f"查询替换: {query[:50]}... -> {result[:50]}...")

        return result

    def extract_entities_from_question(self, question: str) -> List[str]:
        """
        从问题中提取实体（使用 YAML 配置的模式）
        同时应用别名匹配
        """
        entities = set()

        # 预处理：应用查询替换
        processed_question = self.apply_query_replacements(question)

        # 使用配置的模式提取实体
        for pattern in self.entity_extraction_patterns:
            for match in pattern.finditer(processed_question):
                # 获取匹配的实体名称
                if match.groups():
                    entity = match.group(1).strip()
                else:
                    entity = match.group(0).strip()

                # 长度过滤
                if self.min_entity_length <= len(entity) <= self.max_entity_length:
                    # 应用别名标准化
                    entity = self.config.normalize_entity_name(entity)
                    entities.add(entity)

        # 也尝试使用实体模式中的名称提取（从配置的实体类型中匹配）
        compiled_entity_patterns = self.config.get_compiled_entity_patterns()
        for entity_type, patterns in compiled_entity_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(processed_question):
                    if match.groups():
                        entity = match.group(1).strip()
                    else:
                        entity = match.group(0).strip()

                    if self.min_entity_length <= len(entity) <= self.max_entity_length:
                        entity = self.config.normalize_entity_name(entity)
                        entities.add(entity)

        # 限制数量
        entities_list = list(entities)
        if len(entities_list) > self.max_extracted_entities:
            entities_list = entities_list[:self.max_extracted_entities]

        logger.info(f"从问题中提取实体: {len(entities_list)} 个 -> {entities_list[:5]}")
        return entities_list

    def expand_with_relations(self, entities: List[str], max_depth: int = 1) -> List[str]:
        """通过关系扩展实体"""
        expanded = set(entities)
        for e in entities:
            for neighbor, depth, _ in self.neo4j.get_neighbors(e, max_depth):
                if depth <= max_depth:
                    expanded.add(neighbor)
        return list(expanded)

    def entity_search(self, entities: List[str], index_name: str, top_k: int = 10,
                      user_level: str = None) -> List[Dict]:
        """实体检索"""
        if not entities:
            return []
        query = " ".join(entities)
        vec = self.embedding.generate_embedding(query)
        if not vec:
            return []
        results = self.vector_search.similarity_search(vec, index_name, top_k, 0.3, user_level=user_level)
        for r in results:
            r["_source"] = "entity"
        return results

    def hybrid_graph_search(
            self,
            question: str,
            entities: List[str],
            index_name: str,
            top_k: int = 8,
            user_level: str = None
    ) -> Tuple[List[Dict], Dict]:
        """混合图检索（添加推理路径）"""
        results = []

        # 实体检索
        if entities:
            results.extend(self.entity_search(entities, index_name, top_k * 2, user_level))

        # 扩展实体检索
        if entities:
            expanded = self.expand_with_relations(entities)
            if len(expanded) > len(entities):
                results.extend(self.entity_search(expanded, index_name, top_k, user_level))

        # 去重
        seen = set()
        unique = []
        for r in results:
            doc_id = r.get("_id", r.get("id", ""))
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                unique.append(r)

        unique.sort(key=lambda x: x.get("_score", 0), reverse=True)

        for r in unique:
            if "_score" in r and "score" not in r:
                r["score"] = r["_score"]
            elif "rrf_score" in r and "score" not in r:
                r["score"] = r["rrf_score"]
            elif "rerank_score" in r and "score" not in r:
                r["score"] = r["rerank_score"]

        if not unique:
            from app.service.core.retrieval import get_parent_child_retriever
            fallback = get_parent_child_retriever().search_with_query_rewrite(
                question, index_name, top_k, top_k * 2, user_level=user_level
            )
            unique = fallback.get("results", [])

        # ========== 添加推理路径 ==========
        reasoning_paths = []
        if len(entities) >= 2:
            logger.info(f"查找推理路径，实体列表: {entities}")
            for i in range(len(entities) - 1):
                path_result = self.neo4j.get_path_between(entities[i], entities[i + 1], max_depth=3)
                if path_result:
                    logger.info(f"找到路径 {entities[i]} -> {entities[i + 1]}: {path_result}")
                    for path in path_result:
                        reasoning_paths.append({
                            "nodes": path.get("nodes", []),
                            "relations": path.get("relations", []),
                            "length": path.get("length", 0),
                            "relation": " → ".join(path.get("relations", [])) if path.get("relations") else "相关"
                        })
                else:
                    logger.info(f"未找到路径 {entities[i]} -> {entities[i + 1]}")

        # 去重
        unique_paths = []
        seen_keys = set()
        for p in reasoning_paths:
            key = "→".join(p["nodes"])
            if key not in seen_keys:
                seen_keys.add(key)
                unique_paths.append(p)

        logger.info(f"推理路径生成完成: {len(unique_paths)} 条")

        metadata = {
            "search_sources": ["entity", "expanded"],
            "entities_extracted": entities,
            "graph_enabled": True,
            "reasoning_paths": unique_paths[:5],
            "query_replacements_applied": self.query_replacement_enabled
        }

        return unique[:top_k], metadata


def get_graph_cache() -> GraphCache:
    return GraphCache()


__all__ = ['GraphRetriever', 'GraphCache', 'get_graph_cache']