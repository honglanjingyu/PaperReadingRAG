# app/service/core/graphrag/service.py
"""GraphRAG 主服务 - 使用 YAML 配置"""

import os
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .core import EntityExtractor
from .community import CommunityDetector
from .retriever import GraphRetriever, GraphCache
from .neo4j_store import get_neo4j_store, StoredEntity, StoredRelation

# 导入配置管理器
from app.service.graphrag_configs import get_graphrag_config

logger = logging.getLogger(__name__)


class HierarchicalSummarizer:
    """层次摘要生成器"""

    def __init__(self, llm_service=None, max_context_length: int = 4000):
        self.llm_service = llm_service
        self.max_context_length = max_context_length

    def generate_summaries(self, communities: Dict, entity_details: Dict, original_text: str = "") -> Dict:
        summaries = {}
        for cid, comm in communities.items():
            summary_text = self._generate_summary(comm, original_text) if self.llm_service else comm.summary
            summaries[cid] = type('obj', (object,), {
                'title': f"社区_{cid}",
                'summary': summary_text or comm.summary,
                'key_insights': comm.keywords
            })()
        return summaries

    def _generate_summary(self, community, original_text: str) -> Optional[str]:
        if not self.llm_service or not community.entities:
            return None
        entities_str = "、".join(community.entities[:10])
        prompt = f"请为以下实体集合生成简短摘要：{entities_str}\n摘要："
        try:
            return self.llm_service.generate([{"role": "user", "content": prompt}])
        except Exception:
            return None


class GraphRAGService:
    """GraphRAG 主服务 - 使用 YAML 配置"""

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

        self.enabled = os.getenv("GRAPH_RAG_ENABLED", "true").lower() == "true"
        if not self.enabled:
            logger.info("GraphRAG 已禁用")
            return

        # 加载配置
        self.config = get_graphrag_config()

        from app.service.core.llm import get_llm_service
        self.llm_service = get_llm_service()
        self.neo4j = get_neo4j_store()
        self.cache = GraphCache()
        self.neo4j.init_constraints()

        # 从配置获取参数
        use_llm = os.getenv("ENTITY_EXTRACT_USE_LLM", "true").lower() == "true"
        min_frequency = self.config.get_min_entity_length()  # 复用为最小频率
        max_entities = self.config.get_max_entities_per_doc()

        self.entity_extractor = EntityExtractor(
            use_llm=use_llm,
            llm_service=self.llm_service,
            min_frequency=int(os.getenv("ENTITY_MIN_FREQUENCY", "1")),
            max_entities=max_entities
        )

        self.community_detector = CommunityDetector(
            min_community_size=int(os.getenv("COMMUNITY_MIN_SIZE", "2")),
            resolution=float(os.getenv("COMMUNITY_RESOLUTION", "1.0"))
        )
        self.summarizer = HierarchicalSummarizer(self.llm_service)
        self.retriever = GraphRetriever(neo4j_store=self.neo4j)

        # 获取关系类型定义（用于可视化）
        self.relation_types = self.config.get_relation_types()
        self.entity_types = self.config.get_entity_types()

        logger.info(f"GraphRAGService 初始化完成: 实体类型={len(self.entity_types)}, "
                    f"关系类型={len(self.relation_types)}")

    def _get_documents_version(self, user_level: str = None) -> str:
        from app.service.core.vector_store import get_vector_store
        from pymilvus import Collection

        store = get_vector_store()
        index_name = "rag_documents"
        if not store.index_exists(index_name):
            return "empty"

        try:
            collection = Collection(index_name)
            collection.load()
            results = collection.query(expr="", output_fields=["docnm", "user_level"], limit=5000)
            docs = sorted(set(r.get("docnm", "") for r in results))
            return hashlib.md5("".join(docs).encode()).hexdigest()[:16]
        except Exception:
            return "unknown"

    def build_knowledge_graph(
            self,
            documents: List[str],
            doc_names: List[str] = None,
            user_level: str = None,
            force_rebuild: bool = False,
            use_cache: bool = True
    ) -> Dict[str, Any]:
        """构建知识图谱（支持缓存）"""
        if not self.enabled:
            return {"success": False, "error": "GraphRAG 未启用"}

        # 检查缓存
        if use_cache and self.cache and not force_rebuild:
            cached = self.cache.get(user_level)
            if cached:
                logger.info(f"图谱缓存命中")
                return cached

        if not documents:
            return {"success": False, "error": "没有文档"}

        max_docs = int(os.getenv("GRAPH_RAG_MAX_DOCS", "50"))
        documents = documents[:max_docs]

        # 提取实体和关系
        all_entities = {}
        all_relations = []

        for i, text in enumerate(documents):
            if not text.strip():
                continue
            entities, relations = self.entity_extractor.extract_from_text(
                text, doc_names[i] if doc_names and i < len(doc_names) else f"doc_{i}"
            )
            for e in entities:
                if e.name in all_entities:
                    all_entities[e.name].frequency += e.frequency
                else:
                    all_entities[e.name] = e
            all_relations.extend(relations)

        # 存入 Neo4j
        stored_entities = [StoredEntity(name=e.name, entity_type=e.type, frequency=e.frequency, summary="")
                           for e in all_entities.values()]
        self.neo4j.save_entities_batch(stored_entities)

        stored_relations = [
            StoredRelation(source=r.source, target=r.target, relation_type=r.relation_type, weight=r.weight)
            for r in all_relations]
        self.neo4j.save_relations_batch(stored_relations)

        # 社区检测
        communities = self.community_detector.detect_communities(
            [e.to_dict() for e in all_entities.values()],
            [r.to_dict() for r in all_relations]
        )

        # 生成摘要并存入 Neo4j
        for cid, comm in communities.items():
            self.neo4j.save_community(cid, comm.entities[:20], comm.summary, comm.keywords)

        result = {
            "success": True,
            "statistics": self.neo4j.get_statistics(),
            "entities": [e.to_dict() for e in list(all_entities.values())[:50]],
            "relations": [r.to_dict() for r in all_relations[:50]],
            "communities": [{"id": c.id, "size": c.size, "keywords": c.keywords, "summary": c.summary}
                            for c in communities.values()],
            "entity_types": self.entity_types,  # 添加实体类型定义
            "relation_types": self.relation_types,  # 添加关系类型定义
            "_cached_at": datetime.now().isoformat()
        }

        if use_cache and self.cache:
            self.cache.set(result, user_level)

        return result

    def get_cached_graph(self, user_level: str = None) -> Optional[Dict]:
        return self.cache.get(user_level) if self.cache else None

    def invalidate_cache(self, user_level: str = None):
        if self.cache:
            self.cache.invalidate(user_level)

    def graph_search(
            self,
            question: str,
            graph_id: str = None,
            index_name: str = None,
            top_k: int = 8,
            user_level: str = None
    ) -> Tuple[List[Dict], Dict]:
        """图检索"""
        if not self.enabled:
            return [], {"graph_enabled": False}

        index_name = index_name or os.getenv("VECTOR_INDEX_NAME", "rag_documents")
        entities = self.retriever.extract_entities_from_question(question)
        results, metadata = self.retriever.hybrid_graph_search(
            question, entities, index_name, top_k, user_level
        )
        for r in results:
            if "_score" in r and "score" not in r:
                r["score"] = r["_score"]
        return results, metadata

    def _build_graph_context(self, results: List[Dict], metadata: Dict) -> str:
        """构建 GraphRAG 上下文"""
        context_parts = []

        # 添加检索到的文档
        if results:
            for i, r in enumerate(results[:5], 1):
                content = r.get("content", r.get("content_with_weight", ""))[:800]
                doc_name = r.get("docnm", "未知文档")
                context_parts.append(f"[文档{i}] 【{doc_name}】\n{content}\n")

        return "\n".join(context_parts)

    def _generate_answer_with_paths(
            self,
            question: str,
            context: str,
            metadata: Dict
    ) -> Tuple[str, str]:
        """生成带推理路径的答案 - 使用指定提示词格式"""

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
                reasoning_path_str = "\n".join([f"- {p}" for p in paths_desc])

        # 系统提示词
        system_prompt = """你是一个专业的知识图谱问答助手。回答问题时，请遵循以下要求：

1. **标注出处**：引用文档内容时，请使用【文档名】标注来源
2. **展示推理路径**：如果答案涉及多个实体之间的关系，请明确指出推理路径
3. **路径格式**：使用「实体A」 → (关系) → 「实体B」的格式

示例：
根据文档内容，推理路径为：
「RAG技术」 → (核心技术) → 「向量检索」 → (应用) → 「智能问答」

答案：...（具体内容）"""

        # 用户提示词
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

            # 如果没有推理路径但 answer 中也没有，从实体中构建简单路径
            if not reasoning_path_str and answer and "推理路径" not in answer:
                entities = metadata.get("entities_extracted", [])
                if len(entities) >= 2:
                    simple_path = " → ".join([f"「{e}」" for e in entities[:3]])
                    reasoning_path_str = f"推理路径：{simple_path} (相关关系)\n"
                    answer = reasoning_path_str + "\n" + answer

            return answer or "抱歉，无法生成答案。", reasoning_path_str

        except Exception as e:
            logger.error(f"GraphRAG 生成答案失败: {e}")
            return f"生成答案时出错: {e}", ""

    def graph_rag_ask(
            self,
            question: str,
            index_name: str = None,
            top_k: int = 8,
            user_level: str = None
    ) -> Dict[str, Any]:
        """GraphRAG 问答"""
        if not self.enabled:
            return {"success": False, "error": "GraphRAG 未启用"}

        index_name = index_name or os.getenv("VECTOR_INDEX_NAME", "rag_documents")

        # 执行图检索
        results, metadata = self.graph_search(
            question=question,
            graph_id=None,
            index_name=index_name,
            top_k=top_k,
            user_level=user_level
        )

        if not results:
            return {
                "success": False,
                "answer": "未找到相关信息",
                "results": [],
                "graph_info": metadata
            }

        # 构建上下文
        context = self._build_graph_context(results, metadata)

        # 生成答案
        answer, reasoning_path = self._generate_answer_with_paths(question, context, metadata)

        return {
            "success": True,
            "answer": answer or "抱歉，未能生成答案。",
            "reasoning_path": reasoning_path,
            "results": results[:top_k],
            "graph_info": metadata
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息，包含配置信息"""
        stats = self.neo4j.get_statistics() if self.enabled else {"enabled": False}
        if self.enabled:
            stats["config"] = {
                "entity_types_count": len(self.entity_types),
                "relation_types_count": len(self.relation_types),
                "entity_extraction_patterns": len(self.config.get_entity_extraction_patterns()),
                "query_replacements_count": len(self.config.get_query_replacements()),
                "alias_count": len(self.config.get_entity_aliases())
            }
        return stats

    def close(self):
        self.neo4j.close()


def get_graph_rag_service() -> GraphRAGService:
    return GraphRAGService()


__all__ = ['GraphRAGService', 'get_graph_rag_service']