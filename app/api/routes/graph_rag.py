# app/api/routes/graph_rag.py (精简版)
"""
GraphRAG API 路由 - 仅保留图谱构建和管理功能
添加异步处理避免阻塞
"""

from fastapi import APIRouter, HTTPException, Header, BackgroundTasks, Query
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import json
import logging
import asyncio

from app.service.core.vector_store import get_vector_storage_service
from app.auth.jwt_utils import get_user_id_from_token
from app.db.database import get_db_manager
from app.service.core.graphrag import get_graph_rag_service

logger = logging.getLogger(__name__)
router = APIRouter()


class GraphBuildRequest(BaseModel):
    """图谱构建请求"""
    document_ids: Optional[List[str]] = Field(None, description="文档ID列表")
    all_documents: bool = Field(True, description="是否使用所有文档")
    force_rebuild: bool = Field(False, description="是否强制重建（忽略缓存）")


# ========== 辅助函数 ==========

def _get_user_level_from_token(authorization: Optional[str]) -> str:
    """从 token 获取用户等级"""
    user_level = "normal"
    if authorization:
        token = authorization[7:] if authorization.startswith("Bearer ") else authorization
        user_id = get_user_id_from_token(token)
        if user_id:
            db = get_db_manager()
            user = db.get_user_by_id(user_id)
            if user:
                user_level = user.role.value
    return user_level


def _get_document_contents_sync(index_name: str, user_level: str) -> Dict:
    """同步获取文档内容"""
    from app.service.core.vector_store import get_vector_store
    from pymilvus import Collection

    store = get_vector_store()
    if not store.index_exists(index_name):
        return {"documents": []}

    try:
        collection = Collection(index_name)
        collection.load()

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


# ========== GraphRAG 管理接口 ==========

@router.post("/chat/graph/build")
async def build_knowledge_graph(
        request: GraphBuildRequest,
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    构建知识图谱（支持缓存）- 使用异步处理避免阻塞
    """
    user_level = _get_user_level_from_token(authorization)
    index_name = "rag_documents"

    # 使用 asyncio.to_thread 避免阻塞
    doc_result = await asyncio.to_thread(_get_document_contents_sync, index_name, user_level)
    documents = doc_result.get("documents", [])

    if not documents:
        graph_service = get_graph_rag_service()
        cached = await asyncio.to_thread(graph_service.get_cached_graph, user_level)
        if cached:
            logger.info(f"文档为空，返回缓存图谱: user_level={user_level}")
            return cached
        return {"success": False, "error": "知识库为空，请先上传文档"}

    graph_service = get_graph_rag_service()

    # 异步构建图谱
    graph_data = await asyncio.to_thread(
        graph_service.build_knowledge_graph,
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
    """使知识图谱缓存失效（文档更新后调用）"""
    user_level = _get_user_level_from_token(authorization)

    graph_service = get_graph_rag_service()
    await asyncio.to_thread(graph_service.invalidate_cache, user_level)

    return {
        "success": True,
        "message": f"图谱缓存已失效 (user_level={user_level or 'all'})"
    }


@router.get("/chat/graph/status")
async def get_graph_status(
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """获取知识图谱状态（包括缓存状态）"""
    user_level = _get_user_level_from_token(authorization)

    graph_service = get_graph_rag_service()

    stats = await asyncio.to_thread(graph_service.neo4j.get_statistics) if hasattr(graph_service, 'neo4j') else {}
    cached = await asyncio.to_thread(graph_service.get_cached_graph, user_level)

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
    import os
    graph_service = get_graph_rag_service()

    normal_cached = await asyncio.to_thread(graph_service.get_cached_graph, "normal")
    admin_cached = await asyncio.to_thread(graph_service.get_cached_graph, "admin")
    owner_cached = await asyncio.to_thread(graph_service.get_cached_graph, "owner")

    return {
        "success": True,
        "cache_enabled": graph_service.config.get("enable_cache", True) if hasattr(graph_service, 'config') else True,
        "cache_ttl": int(os.getenv("GRAPH_CACHE_TTL", "3600")),
        "user_level_stats": {
            "normal": normal_cached is not None,
            "admin": admin_cached is not None,
            "owner": owner_cached is not None
        }
    }


__all__ = ['router']