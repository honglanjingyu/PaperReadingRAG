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
from .graph_cache import get_graph_cache

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
        "enable_cache": os.getenv("ENABLE_GRAPH_CACHE", "true").lower() == "true",
        "max_entities": int(os.getenv("GRAPH_MAX_ENTITIES", "100")),  # 新增
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
            min_frequency=self.config["min_entity_frequency"],
            max_entities=self.config["max_entities"]  # 添加 max_entities 参数
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

        # 加载 GraphRAG 配置（用于查询替换）
        self._load_graphrag_config()

        self._is_initialized = True

        # 打印统计信息
        stats = self.neo4j.get_statistics()
        logger.info(f"GraphRAGService 初始化完成, Neo4j 统计: {stats}")

    def _load_graphrag_config(self):
        """加载 GraphRAG 配置文件"""
        try:
            from app.service.graphrag_configs import get_graphrag_config
            self.graphrag_config = get_graphrag_config()
            self.query_replacements = self.graphrag_config.get_query_replacements()
            self.query_replacement_enabled = self.graphrag_config.is_query_replacement_enabled()
            logger.info(f"加载 GraphRAG 配置: query_replacement_enabled={self.query_replacement_enabled}, "
                        f"replacements={len(self.query_replacements)}")
        except Exception as e:
            logger.warning(f"加载 GraphRAG 配置失败: {e}")
            self.graphrag_config = None
            self.query_replacements = []
            self.query_replacement_enabled = False

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
            "_documents_version": self._get_documents_version(user_level),
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
        """使用知识图谱进行检索 - 使用父子查询"""
        from app.service.core.retrieval.parent_child_retriever import get_parent_child_retriever

        if not self.config["enabled"]:
            return self._fallback_search(question, index_name, top_k)

        top_k = top_k or self.config["default_top_k"]
        index_name = index_name or os.getenv("VECTOR_INDEX_NAME", "rag_documents")

        # 1. 从问题中提取实体
        entities = self.retriever.extract_entities_from_question(question)

        # 2. 使用父子查询进行检索
        parent_child_retriever = get_parent_child_retriever()
        retrieval_result = parent_child_retriever.search_with_query_rewrite(
            question=question,
            index_name=index_name,
            top_k=top_k,
            recall_k=top_k * 3,
            enable_rerank=True,
            enable_query_rewrite=True
        )

        vector_docs = retrieval_result.get("results", [])

        # 3. 获取社区数据
        communities = {}
        for comm in self.neo4j.get_all_communities():
            communities[comm["community_id"]] = {
                "entities": comm.get("entities", []),
                "size": len(comm.get("entities", [])),
                "density": comm.get("density", 0)
            }

        # 4. 获取摘要
        summaries = {}
        for comm in self.neo4j.get_all_communities():
            if comm.get("summary"):
                summaries[comm["community_id"]] = comm["summary"]

        # 5. 获取实体关系图
        entity_graph = self.neo4j.get_entity_graph()

        # 6. 执行图检索
        results, metadata = self.retriever.hybrid_graph_search_with_paths(
            question=question,
            entities=entities,
            communities=communities,
            summaries=summaries,
            entity_graph=entity_graph,
            index_name=index_name,
            top_k=top_k
        )

        # 7. 合并向量检索结果（去重）
        existing_ids = {r.get("parent_id", r.get("_id", "")) for r in results}
        for doc in vector_docs:
            doc_id = doc.get("parent_id", doc.get("chunk_id", ""))
            if doc_id not in existing_ids:
                results.append({
                    "content": doc.get("content", ""),
                    "score": doc.get("score", 0),
                    "docnm": doc.get("document_name", ""),
                    "_source": "vector_parent"
                })
                existing_ids.add(doc_id)

        # 按分数排序
        results.sort(key=lambda x: x.get("score", 0), reverse=True)

        metadata["graph_enabled"] = True
        metadata["graph_statistics"] = self.neo4j.get_statistics()
        metadata["entities_extracted"] = entities
        metadata["parent_child_enabled"] = True

        return results[:top_k], metadata

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

    def _apply_query_replacements(self, question: str) -> str:
        """
        应用查询替换规则

        Args:
            question: 原始问题

        Returns:
            替换后的问题
        """
        if not self.query_replacement_enabled or not self.query_replacements:
            return question

        expanded_question = question
        for alias, replacement in self.query_replacements:
            if alias in expanded_question and replacement not in expanded_question:
                expanded_question = expanded_question.replace(alias, replacement)

        if expanded_question != question:
            logger.info(f"问题已扩展: {question} -> {expanded_question}")

        return expanded_question

    def graph_rag_ask(
            self,
            question: str,
            index_name: str = None,
            top_k: int = 8,
            user_level: str = None
    ) -> Dict[str, Any]:
        """
        GraphRAG 问答：结合向量检索和图检索
        """
        if not self.config["enabled"]:
            return {
                "success": False,
                "error": "GraphRAG 未启用",
                "fallback_to_advanced": True
            }

        index_name = index_name or os.getenv("VECTOR_INDEX_NAME", "rag_documents")

        # 应用查询替换（别名扩展）
        question = self._apply_query_replacements(question)

        # 1. 执行混合图检索（向量+图谱）
        graph_results, graph_metadata = self.graph_search(
            question=question,
            index_name=index_name,
            top_k=top_k
        )

        # 2. 如果没有检索结果或者检索结果太少，尝试扩展查询
        if not graph_results or len(graph_results) < 3:
            logger.info(f"GraphRAG 初始检索结果不足 ({len(graph_results)} 条)，尝试扩展查询...")

            # 从问题中提取更干净的实体
            import re
            # 移除多余的中文说明
            clean_question = re.sub(r'并说明.*$', '', question)
            clean_question = re.sub(r'以及.*$', '', clean_question)

            # 重新检索
            graph_results, graph_metadata = self.graph_search(
                question=clean_question,
                index_name=index_name,
                top_k=top_k
            )

        # 3. 构建带关系路径的上下文
        context = self._build_graph_context_with_paths(graph_results, graph_metadata)

        # 4. 生成答案
        answer, reasoning_path = self._generate_answer_with_paths(question, context, graph_metadata)

        # 5. 确保答案不为空
        if not answer or len(answer) < 50:
            # 降级到普通 RAG
            logger.warning("GraphRAG 生成答案质量不佳，降级到 Advanced RAG")
            from app.service.core.rag.search import enhanced_search_with_hybrid_and_rerank
            from app.service.core.rag.generation import generate_answer

            retrieval_result = enhanced_search_with_hybrid_and_rerank(
                question=question,
                index_name=index_name,
                top_k=top_k,
                recall_k=top_k * 2,
                user_level=user_level
            )

            if retrieval_result.get("success"):
                gen_result = generate_answer(
                    question=retrieval_result.get("rewritten_query", question),
                    results=retrieval_result.get("results", []),
                    template_name="detailed"
                )
                if gen_result.get("answer"):
                    answer = gen_result.get("answer")

        return {
            "success": True,
            "answer": answer or "抱歉，未能找到相关信息。",
            "reasoning_path": reasoning_path,
            "results": graph_results[:top_k],
            "graph_info": {
                "search_sources": graph_metadata.get("search_sources", []),
                "entities_extracted": graph_metadata.get("entities_extracted", []),
                "graph_statistics": graph_metadata.get("graph_statistics", {}),
                "reasoning_path": reasoning_path
            }
        }

    def _build_graph_context_with_paths(
            self,
            results: List[Dict],
            metadata: Dict
    ) -> str:
        """构建带关系路径的上下文"""
        context_parts = []

        # 添加实体关系路径
        if metadata.get("reasoning_paths"):
            context_parts.append("## 🔗 实体关系路径")
            for path in metadata["reasoning_paths"][:3]:
                path_str = " → ".join([f"「{node}」" for node in path["nodes"]])
                context_parts.append(f"- {path_str} ({path.get('relation', '关联')})")

        # 添加检索到的文档
        if results:
            context_parts.append("\n## 📄 相关文档内容")
            for i, r in enumerate(results[:5], 1):
                content = r.get("content", r.get("content_with_weight", ""))[:800]
                doc_name = r.get("docnm", "未知文档")
                source = r.get("_source", "检索")
                context_parts.append(f"\n### [{i}] 来自「{doc_name}」({source})\n{content}")

        # 添加社区摘要（使用 metadata 中的 relevant_summaries）
        if metadata.get("relevant_summaries"):
            context_parts.append("\n## 📊 知识社区摘要")
            for s in metadata["relevant_summaries"][:2]:
                title = s.get('title', '社区主题')
                summary = s.get('summary', '')
                if summary:
                    context_parts.append(f"\n### {title}\n{summary}")

        return "\n".join(context_parts)

    def _generate_answer_with_paths(
            self,
            question: str,
            context: str,
            metadata: Dict
    ) -> Tuple[str, str]:
        """生成带推理路径的答案"""

        # 提取推理路径描述
        reasoning_path_str = ""
        if metadata.get("reasoning_paths"):
            paths_desc = []
            for path in metadata["reasoning_paths"][:3]:
                path_desc = " → ".join([f"「{node}」" for node in path["nodes"]])
                if path.get("relation"):
                    paths_desc.append(f"{path_desc} ({path['relation']})")
                else:
                    paths_desc.append(path_desc)

            if paths_desc:
                reasoning_path_str = "## 🧠 推理路径\n" + "\n".join([f"- {p}" for p in paths_desc]) + "\n"

        # 构建系统提示
        system_prompt = """你是一个专业的知识图谱问答助手。回答问题时，请遵循以下要求：

1. **标注出处**：引用文档内容时，请使用【文档名】标注来源
2. **展示推理路径**：如果答案涉及多个实体之间的关系，请明确指出推理路径
3. **路径格式**：使用「实体A」 → (关系) → 「实体B」的格式

示例：
根据文档内容，推理路径为：
「RAG技术」 → (核心技术) → 「向量检索」 → (应用) → 「智能问答」

答案：...（具体内容）"""

        # 构建用户提示
        user_prompt = f"""
{reasoning_path_str}

## 参考文档和知识图谱
{context}

## 用户问题
{question}

## 回答要求
1. 如果有推理路径，请先展示路径再给出答案
2. 每处引用都要标注文档出处
3. 答案要准确、完整

## 回答
"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            answer = self.llm_service.generate(messages)

            # 如果没有推理路径但 answer 中也没有，添加一个说明
            if not reasoning_path_str and answer and "推理路径" not in answer:
                # 从实体中构建简单路径
                entities = metadata.get("entities_extracted", [])
                if len(entities) >= 2:
                    simple_path = " → ".join([f"「{e}」" for e in entities[:3]])
                    reasoning_path_str = f"## 🧠 推理路径\n- {simple_path} (相关关系)\n\n"
                    answer = reasoning_path_str + answer

            return answer or "抱歉，无法生成答案。", reasoning_path_str

        except Exception as e:
            logger.error(f"GraphRAG 生成答案失败: {e}")
            return f"生成答案时出错: {e}", ""

    def extract_reasoning_paths(
            self,
            question: str,
            entities: List[str],
            max_depth: int = 2
    ) -> List[Dict]:
        """
        提取实体之间的推理路径
        """
        paths = []

        if len(entities) < 2:
            return paths

        # 获取实体之间的连接路径
        for i in range(len(entities) - 1):
            source = entities[i]
            target = entities[i + 1]

            # 查找最短路径
            path_result = self.neo4j.get_path_between(source, target, max_depth)

            if path_result:
                for path in path_result:
                    paths.append({
                        "nodes": path.get("nodes", []),
                        "relations": path.get("relations", []),
                        "length": path.get("length", 0)
                    })

        # 去重
        unique_paths = []
        seen = set()
        for p in paths:
            key = "→".join(p["nodes"])
            if key not in seen:
                seen.add(key)
                unique_paths.append(p)

        return unique_paths[:5]

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
        self.invalidate_cache()
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


def get_graph_rag_service() -> "GraphRAGService":
    """获取 GraphRAG 服务单例"""
    return GraphRAGService()


__all__ = ['GraphRAGService', 'get_graph_rag_service', 'get_graph_config']