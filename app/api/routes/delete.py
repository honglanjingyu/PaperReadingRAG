# app/api/routes/delete.py

import os
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Header
from pathlib import Path

from app.api.config import UPLOAD_DIR
from app.auth.jwt_utils import get_user_id_from_token
from app.db.database import get_db_manager
from app.service.core.cache import get_document_cache
from app.service.core.vector_store import get_vector_store
from app.service.core.retrieval.es_bm25_retriever import get_es_bm25_retriever
from app.service.core.memory import get_memory_manager
from app.service.core.rag.cached_search import CachedSearchService

router = APIRouter()
logger = logging.getLogger(__name__)


# ========== 辅助函数 ==========

def _get_document_level_sync(filename: str, index_name: str) -> str:
    """同步获取文档等级"""
    from app.service.core.cache import get_document_cache
    from app.service.core.vector_store import get_vector_store
    from pymilvus import Collection

    doc_cache = get_document_cache()

    # 先查缓存
    cached_level = doc_cache.get_document_level(filename)
    if cached_level:
        return cached_level

    # 缓存未命中，查询 Milvus
    try:
        store = get_vector_store()
        if store and store.index_exists(index_name):
            collection = Collection(index_name)
            collection.load()

            expr = f'docnm == "{filename}"'
            results = collection.query(
                expr=expr,
                output_fields=["docnm", "user_level"],
                limit=1
            )
            if results:
                level = results[0].get("user_level", "normal")
                doc_cache.set_document_level(filename, level)
                return level
    except Exception as e:
        logger.warning(f"获取文档等级失败 {filename}: {e}")

    return "normal"


def _check_delete_permission(user_level: str, doc_level: str) -> tuple[bool, str]:
    """
    检查用户是否有权删除文档
    权限规则:
        - owner: 可删除所有文档
        - admin: 可删除 normal 和 admin 文档
        - normal: 只能删除 normal 文档
    """
    level_priority = {"normal": 1, "admin": 2, "owner": 3}
    current_priority = level_priority.get(user_level, 1)
    doc_priority = level_priority.get(doc_level, 1)

    if current_priority >= doc_priority:
        return True, f"权限允许 (用户等级={user_level}, 文档等级={doc_level})"
    else:
        return False, f"权限不足: 用户等级={user_level} 无法删除等级={doc_level} 的文档"

def _delete_from_local_storage(filename: str) -> bool:
    """删除本地存储的文件"""
    try:
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()
            logger.info(f"本地文件已删除: {filename}")
            return True
        return False
    except Exception as e:
        logger.error(f"本地文件删除失败: {e}")
        return False


def _delete_from_milvus(filename: str, index_name: str) -> int:
    """从 Milvus 删除文档的所有分块"""
    try:
        store = get_vector_store()
        if store and store.index_exists(index_name):
            deleted = store.delete(index_name, {"docnm": filename})
            logger.info(f"Milvus 删除完成: {filename}, 删除 {deleted} 条记录")
            return deleted if isinstance(deleted, int) else 0
    except Exception as e:
        logger.error(f"Milvus 删除失败: {e}")
    return 0


def _delete_from_elasticsearch(filename: str, index_name: str) -> int:
    """从 Elasticsearch 删除文档的所有相关记录"""
    try:
        es_retriever = get_es_bm25_retriever()
        if es_retriever and es_retriever.is_available():
            es_index = f"rag_bm25_{index_name}"
            client = es_retriever._client

            if client and client.indices.exists(index=es_index):
                query = {
                    "query": {"term": {"document_name": filename}},
                    "_source": False
                }

                response = client.search(
                    index=es_index,
                    body=query,
                    scroll="2m",
                    size=1000
                )

                scroll_id = response.get("_scroll_id")
                hits = response.get("hits", {}).get("hits", [])
                doc_ids = [hit["_id"] for hit in hits]

                while hits:
                    response = client.scroll(scroll_id=scroll_id, scroll="2m")
                    scroll_id = response.get("_scroll_id")
                    hits = response.get("hits", {}).get("hits", [])
                    doc_ids.extend([hit["_id"] for hit in hits])

                if scroll_id:
                    client.clear_scroll(scroll_id=scroll_id)

                if doc_ids:
                    from elasticsearch.helpers import bulk
                    actions = [
                        {"_op_type": "delete", "_index": es_index, "_id": doc_id}
                        for doc_id in doc_ids
                    ]
                    success, failed = bulk(client, actions, stats_only=True, raise_on_error=False)
                    logger.info(f"Elasticsearch 删除完成: {filename}, 删除 {success} 条记录")
                    return success
        return 0
    except Exception as e:
        logger.error(f"Elasticsearch 删除失败: {e}")
    return 0


def _delete_from_redis_and_memory(filename: str) -> int:
    """从 Redis 和对话记忆中删除与文档相关的历史"""
    try:
        memory_manager = get_memory_manager()
        if not memory_manager or not memory_manager.redis_client:
            logger.warning("Redis 不可用，跳过对话历史清理")
            return 0

        deleted_count = 0
        redis_client = memory_manager.redis_client

        pattern = f"{memory_manager._key_prefix}*"
        meta_pattern = f"{memory_manager._meta_prefix}*"

        all_keys = redis_client.keys(pattern)
        meta_keys = redis_client.keys(meta_pattern)
        all_session_keys = list(set(all_keys + meta_keys))

        logger.info(f"找到 {len(all_session_keys)} 个会话相关 key")

        for session_key in all_session_keys:
            try:
                key_type = redis_client.type(session_key)

                if key_type == 'list':
                    messages = redis_client.lrange(session_key, 0, -1)
                    if not messages:
                        continue

                    import json
                    parsed_messages = []
                    for msg_json in messages:
                        try:
                            parsed_messages.append(json.loads(msg_json))
                        except:
                            parsed_messages.append({"role": "unknown", "content": msg_json})

                    indices_to_remove = set()

                    for idx, msg in enumerate(parsed_messages):
                        content = msg.get("content", "")
                        role = msg.get("role", "")

                        if filename in content or f"文档「{filename}」" in content:
                            indices_to_remove.add(idx)
                            if role == "assistant" and idx > 0:
                                prev_msg = parsed_messages[idx - 1]
                                if prev_msg.get("role") == "user":
                                    indices_to_remove.add(idx - 1)

                    if indices_to_remove:
                        valid_messages = []
                        for idx, msg_json in enumerate(messages):
                            if idx not in indices_to_remove:
                                valid_messages.append(msg_json)

                        redis_client.delete(session_key)
                        if valid_messages:
                            redis_client.rpush(session_key, *valid_messages)

                        deleted_count += len(indices_to_remove)

                elif key_type == 'hash':
                    meta_data = redis_client.hgetall(session_key)
                    if meta_data:
                        has_doc_ref = any(filename in str(value) for value in meta_data.values())
                        if has_doc_ref:
                            redis_client.delete(session_key)
                            deleted_count += 1

                elif key_type == 'string':
                    value = redis_client.get(session_key)
                    if value and filename in str(value):
                        redis_client.delete(session_key)
                        deleted_count += 1

            except Exception as e:
                logger.warning(f"处理 key {session_key} 时出错: {e}")
                continue

        logger.info(f"Redis 清理完成: 删除了 {deleted_count} 条相关记录")
        return deleted_count

    except Exception as e:
        logger.error(f"Redis 清理失败: {e}", exc_info=True)
        return 0


def _invalidate_document_cache(filename: str) -> bool:
    """使文档等级缓存失效"""
    try:
        doc_cache = get_document_cache()
        doc_cache.delete_document_level(filename)
        logger.debug(f"文档等级缓存已清除: {filename}")
        return True
    except Exception as e:
        logger.error(f"文档等级缓存清除失败 {filename}: {e}")
        return False


def _invalidate_search_cache(filename: str) -> bool:
    """使搜索缓存失效"""
    try:
        search_cache = CachedSearchService()
        search_cache.invalidate_cache(pattern=f"*{filename}*")
        logger.debug(f"搜索缓存已清除: {filename}")
        return True
    except Exception as e:
        logger.warning(f"搜索缓存清除失败 {filename}: {e}")
        return False


# ========== 单个删除 API ==========

@router.delete("/upload/{filename}")
async def delete_document(
        filename: str,
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    删除单个文档（同时删除 Milvus、Elasticsearch、Redis、本地存储的所有相关数据）
    """
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")

    index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")

    # ========== 获取当前用户等级 ==========
    user_level = "normal"
    user_id = None
    username = None

    if authorization:
        token = authorization[7:] if authorization.startswith("Bearer ") else authorization
        user_id = get_user_id_from_token(token)
        if user_id:
            db = get_db_manager()
            user = db.get_user_by_id(user_id)
            if user:
                user_level = user.role.value
                username = user.username
                logger.info(f"用户 {username} (等级={user_level}) 请求删除文档: {filename}")

    # 未登录不允许删除
    if user_id is None:
        raise HTTPException(status_code=401, detail="请先登录")

    # ========== 获取文档等级 ==========
    doc_level = _get_document_level_sync(filename, index_name)
    logger.info(f"文档 {filename}: 等级={doc_level}")

    # ========== 验证权限 ==========
    can_delete, reason = _check_delete_permission(user_level, doc_level)

    if not can_delete:
        logger.warning(f"用户 {username} (等级={user_level}) 无权删除 {filename} (等级={doc_level}): {reason}")
        raise HTTPException(status_code=403, detail=reason)

    logger.info(f"权限验证通过: {reason}")

    result = {
        "success": True,
        "filename": filename,
        "deleted": {
            "local": False,
            "milvus": 0,
            "elasticsearch": 0,
            "redis_messages": 0,
            "cache": False
        },
        "message": ""
    }

    details = [f"✅ 权限验证通过: {reason}"]

    try:
        # 1. 删除本地文件
        if _delete_from_local_storage(filename):
            result["deleted"]["local"] = True
            details.append("✅ 本地文件已删除")
        else:
            details.append("⚠️ 本地文件删除失败")

        # 2. 从 Milvus 删除向量数据
        milvus_deleted = _delete_from_milvus(filename, index_name)
        result["deleted"]["milvus"] = milvus_deleted
        if milvus_deleted > 0:
            details.append(f"✅ Milvus 已删除 {milvus_deleted} 条向量记录")
        else:
            details.append("⚠️ Milvus 无相关记录或删除失败")

        # 3. 从 Elasticsearch 删除 BM25 索引数据
        es_deleted = _delete_from_elasticsearch(filename, index_name)
        result["deleted"]["elasticsearch"] = es_deleted
        if es_deleted > 0:
            details.append(f"✅ Elasticsearch 已删除 {es_deleted} 条记录")
        else:
            details.append("⚠️ Elasticsearch 无相关记录或删除失败")

        # 4. 从 Redis 清理对话历史
        redis_deleted = _delete_from_redis_and_memory(filename)
        result["deleted"]["redis_messages"] = redis_deleted
        if redis_deleted > 0:
            details.append(f"✅ Redis 已清理 {redis_deleted} 条相关对话")
        else:
            details.append("✅ Redis 无相关对话记录")

        # 5. 使文档等级缓存失效
        _invalidate_document_cache(filename)
        result["deleted"]["cache"] = True
        details.append("✅ 文档等级缓存已清除")

        # 6. 使搜索缓存失效
        _invalidate_search_cache(filename)
        details.append("✅ 搜索缓存已清除")

        result["message"] = "\n".join(details)
        logger.info(f"文档删除完成: {filename}, 删除者: {username} (等级={user_level})")

        # 7. 使知识图谱缓存失效
        from app.service.core.graphrag import get_graph_rag_service
        graph_service = get_graph_rag_service()
        graph_service.invalidate_cache(user_level)

        return result

    except Exception as e:
        logger.error(f"删除文档失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


__all__ = ['router']