# app/api/routes/graph_rag.py
"""
GraphRAG API 路由
"""

from fastapi import APIRouter, HTTPException, Header, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import json
import logging
import asyncio

from app.service.core.vector_store import get_vector_storage_service
from app.service.core.rag import enhanced_search_with_hybrid_and_rerank
from app.service.core.llm import get_llm_service
from app.auth.jwt_utils import get_user_id_from_token
from app.db.database import get_db_manager

logger = logging.getLogger(__name__)
router = APIRouter()


class GraphRAGRequest(BaseModel):
    """GraphRAG 请求模型"""
    question: str = Field(..., description="用户问题")
    session_id: Optional[str] = Field(None, description="会话ID")
    top_k: int = Field(8, description="返回结果数量")
    rebuild_graph: bool = Field(False, description="是否重建知识图谱")
    include_summaries: bool = Field(True, description="是否包含社区摘要")


class GraphBuildRequest(BaseModel):
    """图谱构建请求"""
    document_ids: Optional[List[str]] = Field(None, description="文档ID列表")
    all_documents: bool = Field(True, description="是否使用所有文档")
    force_rebuild: bool = Field(False, description="是否强制重建（忽略缓存）")


@router.post("/chat/graph/ask/stream")
async def graph_rag_ask_stream(
        request: GraphRAGRequest,
        authorization: Optional[str] = Header(None)
):
    """GraphRAG 流式问答"""
    from app.service.core.graphrag import get_graph_rag_service

    # 获取用户等级
    user_level = "normal"
    if authorization:
        token = authorization[7:] if authorization.startswith("Bearer ") else authorization
        user_id = get_user_id_from_token(token)
        if user_id:
            db = get_db_manager()
            user = db.get_user_by_id(user_id)
            if user:
                user_level = user.role.value

    graph_service = get_graph_rag_service()
    index_name = "rag_documents"

    async def generate():
        try:
            # 发送开始标记
            yield json.dumps({"type": "start", "content": ""}) + "\n"
            yield json.dumps({
                "type": "info",
                "content": "正在构建知识图谱检索...",
                "mode": "graph"
            }) + "\n"

            # 获取或构建知识图谱
            storage_service = get_vector_storage_service()

            # 获取所有文档
            doc_list_result = await asyncio.to_thread(
                _get_document_contents,
                index_name,
                user_level
            )

            documents = doc_list_result.get("documents", [])

            if not documents:
                yield json.dumps({
                    "type": "error",
                    "content": "知识库为空，请先上传文档"
                }) + "\n"
                return

            # 构建或获取图谱（使用缓存）
            graph_data = None

            # 先尝试从缓存获取
            if not request.rebuild_graph:
                graph_data = graph_service.get_cached_graph(user_level)
                if graph_data:
                    yield json.dumps({
                        "type": "info",
                        "content": f"✅ 从缓存加载知识图谱: {graph_data['statistics']['entity_count']} 个实体, "
                                   f"{graph_data['statistics']['relation_count']} 个关系"
                    }) + "\n"

            # 如果需要重建或缓存未命中
            if not graph_data or request.rebuild_graph:
                yield json.dumps({
                    "type": "info",
                    "content": f"正在分析 {len(documents)} 个文档，构建知识图谱..."
                }) + "\n"

                graph_data = await asyncio.to_thread(
                    graph_service.build_knowledge_graph,
                    [d["content"] for d in documents],
                    [d["name"] for d in documents],
                    use_cache=True,
                    user_level=user_level,
                    force_rebuild=request.rebuild_graph
                )

                if graph_data.get("success"):
                    yield json.dumps({
                        "type": "info",
                        "content": f"✅ 知识图谱构建完成: {graph_data['statistics']['entity_count']} 个实体, "
                                   f"{graph_data['statistics']['relation_count']} 个关系"
                    }) + "\n"
                else:
                    yield json.dumps({
                        "type": "info",
                        "content": "⚠️ 图谱构建失败，使用标准检索"
                    }) + "\n"

            # 执行图检索
            yield json.dumps({
                "type": "info",
                "content": "正在查询知识图谱..."
            }) + "\n"

            results, metadata = await asyncio.to_thread(
                graph_service.graph_search,
                request.question,
                None,
                index_name,
                request.top_k
            )

            # 发送检索结果
            if results:
                formatted_results = []
                for r in results[:request.top_k]:
                    formatted_results.append({
                        "content": r.get("content", r.get("content_with_weight", ""))[:300],
                        "score": r.get("score", 0),
                        "document_name": r.get("docnm", ""),
                        "source": r.get("_source", "unknown")
                    })

                yield json.dumps({
                    "type": "retrieval_results",
                    "results": formatted_results,
                    "retrieval_info": metadata
                }) + "\n"

            # 发送社区摘要（如果有）
            if request.include_summaries and metadata.get("relevant_summaries"):
                yield json.dumps({
                    "type": "summaries",
                    "summaries": metadata["relevant_summaries"]
                }) + "\n"

            # 生成答案
            yield json.dumps({
                "type": "info",
                "content": "正在生成答案..."
            }) + "\n"

            # 构建上下文
            context = _build_graph_context(results, metadata)

            # 流式生成答案
            llm_service = get_llm_service()
            messages = [
                {"role": "system", "content": _get_graph_system_prompt()},
                {"role": "user", "content": _build_graph_prompt(request.question, context, metadata)}
            ]

            full_answer = ""
            for chunk in llm_service.generate_stream(messages):
                if chunk:
                    full_answer += chunk
                    yield json.dumps({"type": "answer", "content": chunk}) + "\n"

            # 结束标记
            yield json.dumps({
                "type": "end",
                "session_id": request.session_id,
                "graph_info": metadata
            }) + "\n"

        except Exception as e:
            logger.error(f"GraphRAG 流式问答失败: {e}", exc_info=True)
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.post("/chat/graph/build")
async def build_knowledge_graph(
        request: GraphBuildRequest,
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    构建知识图谱（支持缓存）

    参数：
    - force_rebuild: 强制重建，忽略缓存
    """
    from app.service.core.graphrag import get_graph_rag_service

    user_level = "normal"
    if authorization:
        token = authorization[7:] if authorization.startswith("Bearer ") else authorization
        user_id = get_user_id_from_token(token)
        if user_id:
            db = get_db_manager()
            user = db.get_user_by_id(user_id)
            if user:
                user_level = user.role.value

    index_name = "rag_documents"

    # 获取文档
    doc_result = _get_document_contents(index_name, user_level)
    documents = doc_result.get("documents", [])

    if not documents:
        graph_service = get_graph_rag_service()
        # 尝试返回缓存数据
        cached = graph_service.get_cached_graph(user_level)
        if cached:
            logger.info(f"文档为空，返回缓存图谱: user_level={user_level}")
            return cached
        return {"success": False, "error": "知识库为空，请先上传文档"}

    graph_service = get_graph_rag_service()

    # 构建图谱（支持缓存）
    graph_data = graph_service.build_knowledge_graph(
        [d["content"] for d in documents],
        [d["name"] for d in documents],
        use_cache=True,
        user_level=user_level,
        force_rebuild=request.force_rebuild
    )

    return graph_data


@router.post("/chat/graph/invalidate-cache")
async def invalidate_graph_cache(
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    使知识图谱缓存失效（文档更新后调用）
    """
    from app.service.core.graphrag import get_graph_rag_service

    user_level = None
    if authorization:
        token = authorization[7:] if authorization.startswith("Bearer ") else authorization
        user_id = get_user_id_from_token(token)
        if user_id:
            db = get_db_manager()
            user = db.get_user_by_id(user_id)
            if user:
                user_level = user.role.value

    graph_service = get_graph_rag_service()
    graph_service.invalidate_cache(user_level)

    return {
        "success": True,
        "message": f"图谱缓存已失效 (user_level={user_level or 'all'})"
    }

# app/api/routes/graph_rag.py - 优化 /chat/graph/status 接口

@router.get("/chat/graph/status")
async def get_graph_status(
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """获取知识图谱状态（包括缓存状态）"""
    from app.service.core.graphrag import get_graph_rag_service

    user_level = "normal"
    if authorization:
        token = authorization[7:] if authorization.startswith("Bearer ") else authorization
        user_id = get_user_id_from_token(token)
        if user_id:
            db = get_db_manager()
            user = db.get_user_by_id(user_id)
            if user:
                user_level = user.role.value

    graph_service = get_graph_rag_service()

    # 一次性获取统计信息（避免重复调用）
    stats = graph_service.neo4j.get_statistics() if hasattr(graph_service, 'neo4j') else {}

    # 只查一次缓存
    cached = graph_service.get_cached_graph(user_level)

    cache_info = {
        "is_cached": cached is not None,
        "cached_at": cached.get("_cached_at") if cached else None,
        "cached_entities": cached.get("statistics", {}).get("entity_count", 0) if cached else 0
    } if cached else {"is_cached": False}

    return {
        "success": True,
        "has_cache": cache_info["is_cached"],
        "cached_graphs": 1 if cache_info["is_cached"] else 0,
        "service_initialized": graph_service._is_initialized if hasattr(graph_service, '_is_initialized') else True,
        "statistics": stats if stats else {"entity_count": 0, "relation_count": 0, "community_count": 0},
        "cache_info": cache_info
    }


@router.get("/chat/graph/cache-stats")
async def get_graph_cache_stats(
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """获取图谱缓存统计信息"""
    from app.service.core.graphrag import get_graph_rag_service

    graph_service = get_graph_rag_service()

    # 分别检查不同等级的缓存状态（每个只查一次）
    normal_cached = graph_service.get_cached_graph("normal")
    admin_cached = graph_service.get_cached_graph("admin")
    owner_cached = graph_service.get_cached_graph("owner")

    return {
        "success": True,
        "cache_enabled": graph_service.config.get("enable_cache", True),
        "cache_ttl": int(os.getenv("GRAPH_CACHE_TTL", "3600")),
        "user_level_stats": {
            "normal": normal_cached is not None,
            "admin": admin_cached is not None,
            "owner": owner_cached is not None
        }
    }

def _get_document_contents(index_name: str, user_level: str) -> Dict:
    """获取所有文档内容"""
    from app.service.core.vector_store import get_vector_store

    store = get_vector_store()
    if not store.index_exists(index_name):
        return {"documents": []}

    # 简化实现：从向量库获取文档列表
    from pymilvus import Collection

    try:
        collection = Collection(index_name)
        collection.load()

        # 获取所有唯一的文档名，支持等级过滤
        expr_parts = []
        if user_level:
            level_priority = {"normal": 1, "admin": 2, "owner": 3}
            current_priority = level_priority.get(user_level, 1)
            allowed_levels = [level for level, priority in level_priority.items() if priority <= current_priority]
            levels_str = ", ".join([f"'{level}'" for level in allowed_levels])
            expr_parts.append(f"user_level in [{levels_str}]")

        expr = " and ".join(expr_parts) if expr_parts else ""

        results = collection.query(
            expr=expr,
            output_fields=["docnm", "content_with_weight", "user_level"],
            limit=500
        )

        # 去重
        doc_map = {}
        for r in results:
            doc_name = r.get("docnm", "")
            content = r.get("content_with_weight", "")
            if doc_name and doc_name not in doc_map:
                doc_map[doc_name] = {
                    "name": doc_name,
                    "content": content,
                    "user_level": r.get("user_level", "normal")
                }

        documents = list(doc_map.values())
        logger.info(f"获取到 {len(documents)} 个文档 (user_level={user_level})")
        return {"documents": documents}

    except Exception as e:
        logger.error(f"获取文档失败: {e}")
        return {"documents": []}


def _build_graph_context(results: List[Dict], metadata: Dict) -> str:
    """构建 GraphRAG 上下文"""
    context_parts = []

    # 添加检索结果
    if results:
        context_parts.append("## 相关文档内容")
        for i, r in enumerate(results[:5], 1):
            content = r.get("content", r.get("content_with_weight", ""))[:800]
            doc_name = r.get("docnm", "未知文档")
            source = r.get("_source", "检索")
            context_parts.append(f"\n### [{i}] 来自 {doc_name} ({source})\n{content}")

    # 添加社区摘要
    if metadata.get("relevant_summaries"):
        context_parts.append("\n## 知识图谱社区摘要")
        for s in metadata["relevant_summaries"][:2]:
            context_parts.append(f"\n### {s.get('title', '社区主题')}")
            context_parts.append(s.get('summary', ''))

    return "\n".join(context_parts)


def _build_graph_prompt(question: str, context: str, metadata: Dict) -> str:
    """构建 GraphRAG Prompt"""
    return f"""请基于以下知识图谱检索结果回答用户问题。

## 知识图谱信息
- 图谱统计: {metadata.get('graph_statistics', {})}
- 搜索方式: {', '.join(metadata.get('search_sources', []))}
- 是否启用图谱: {metadata.get('graph_enabled', False)}

## 检索到的文档和社区摘要
{context}

## 用户问题
{question}

## 回答要求
1. 结合文档内容和知识图谱信息回答
2. 如果问题涉及实体关系，请指出实体间的关联
3. 如果有社区摘要，可以引用其中的洞察
4. 回答要准确、完整、有条理
5. 避免编造信息

## 回答
"""


def _get_graph_system_prompt() -> str:
    """获取 GraphRAG 系统 Prompt"""
    return """你是一个专业的知识图谱问答助手。你不仅能检索文档，还能利用知识图谱中的实体关系和社区摘要来回答问题。

能力特点：
- 理解实体之间的关联关系
- 能够进行跨文档的知识推理
- 可以提供更全面的知识洞察

回答时请注意：
1. 如果答案涉及多个相关实体，请说明它们之间的关系
2. 可以引用知识图谱中的社区摘要作为补充信息
3. 保持回答的准确性和可读性
4. 对于不确定的信息，请明确说明"""


# 导出
__all__ = ['router']