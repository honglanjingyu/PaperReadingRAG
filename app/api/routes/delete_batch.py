# app/api/routes/delete_batch.py
"""
批量文档删除路由 - 只包含批量删除功能
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException, Header
from pathlib import Path

from app.api.config import UPLOAD_DIR
from app.auth.jwt_utils import get_user_id_from_token
from app.db.database import get_db_manager
from app.service.core.cache import get_document_cache
from app.service.core.vector_store import get_vector_store
from app.service.core.retrieval.es_bm25_retriever import get_es_bm25_retriever
from app.service.core.memory import get_memory_manager
from app.service.core.rag.cached_search import CachedSearchService

logger = logging.getLogger(__name__)
router = APIRouter()

# 创建删除专用的线程池
DELETE_MAX_WORKERS = int(os.getenv("ASYNC_PROCESSOR_MAX_WORKERS", "3"))
from concurrent.futures import ThreadPoolExecutor
_delete_executor = ThreadPoolExecutor(max_workers=DELETE_MAX_WORKERS)

logger.info(f"批量删除线程池初始化完成，max_workers={DELETE_MAX_WORKERS}")


# ========== 批量删除辅助函数 ==========

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

def _batch_delete_from_local_storage(filenames: List[str]) -> Dict[str, bool]:
    """批量删除本地文件"""
    results = {}
    for filename in filenames:
        try:
            file_path = UPLOAD_DIR / filename
            if file_path.exists():
                file_path.unlink()
                results[filename] = True
                logger.info(f"本地文件已删除: {filename}")
            else:
                results[filename] = False
                logger.warning(f"本地文件不存在: {filename}")
        except Exception as e:
            logger.error(f"本地文件删除失败 {filename}: {e}")
            results[filename] = False
    return results


def _batch_delete_from_milvus(filenames: List[str], index_name: str) -> Dict[str, int]:
    """批量从 Milvus 删除向量数据"""
    results = {filename: 0 for filename in filenames}

    try:
        store = get_vector_store()
        if store and store.index_exists(index_name):
            from pymilvus import Collection
            collection = Collection(index_name)
            collection.load()

            expr_parts = [f"docnm == '{filename}'" for filename in filenames]
            expr = " or ".join(expr_parts)

            for filename in filenames:
                count_expr = f"docnm == '{filename}'"
                result = collection.query(
                    expr=count_expr,
                    output_fields=["docnm"],
                    limit=10000
                )
                results[filename] = len(result)

            collection.delete(expr)
            collection.flush()

            logger.info(f"Milvus 批量删除完成: {results}")
    except Exception as e:
        logger.error(f"Milvus 批量删除失败: {e}")

    return results


def _batch_delete_from_elasticsearch(filenames: List[str], index_name: str) -> Dict[str, int]:
    """批量从 Elasticsearch 删除文档"""
    results = {filename: 0 for filename in filenames}

    try:
        es_retriever = get_es_bm25_retriever()
        if es_retriever and es_retriever.is_available():
            es_index = f"rag_bm25_{index_name}"
            client = es_retriever._client

            if client and client.indices.exists(index=es_index):
                from elasticsearch.helpers import bulk

                all_doc_ids = []

                for filename in filenames:
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

                    results[filename] = len(doc_ids)
                    all_doc_ids.extend(doc_ids)

                if all_doc_ids:
                    actions = [
                        {"_op_type": "delete", "_index": es_index, "_id": doc_id}
                        for doc_id in all_doc_ids
                    ]
                    success, failed = bulk(client, actions, stats_only=True, raise_on_error=False)
                    logger.info(f"Elasticsearch 批量删除完成: 成功 {success}, 失败 {failed}")
    except Exception as e:
        logger.error(f"Elasticsearch 批量删除失败: {e}")

    return results


def _batch_delete_from_redis_and_memory(filenames: List[str]) -> Dict[str, int]:
    """批量从 Redis 清理对话历史"""
    results = {filename: 0 for filename in filenames}

    try:
        memory_manager = get_memory_manager()
        if not memory_manager or not memory_manager.redis_client:
            logger.warning("Redis 不可用，跳过对话历史清理")
            return results

        redis_client = memory_manager.redis_client

        pattern = f"{memory_manager._key_prefix}*"
        meta_pattern = f"{memory_manager._meta_prefix}*"

        all_keys = redis_client.keys(pattern)
        meta_keys = redis_client.keys(meta_pattern)
        all_session_keys = list(set(all_keys + meta_keys))

        logger.info(f"找到 {len(all_session_keys)} 个会话相关 key")

        for filename in filenames:
            deleted_in_session = 0

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
                            if filename in content or f"文档「{filename}」" in content:
                                indices_to_remove.add(idx)
                                if msg.get("role") == "assistant" and idx > 0:
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

                            deleted_in_session += len(indices_to_remove)

                    elif key_type == 'hash':
                        meta_data = redis_client.hgetall(session_key)
                        if meta_data:
                            has_doc_ref = any(filename in str(value) for value in meta_data.values())
                            if has_doc_ref:
                                redis_client.delete(session_key)
                                deleted_in_session += 1

                except Exception as e:
                    logger.warning(f"处理 key {session_key} 时出错: {e}")
                    continue

            results[filename] = deleted_in_session
            logger.info(f"Redis 清理完成 {filename}: 删除了 {deleted_in_session} 条相关记录")

    except Exception as e:
        logger.error(f"Redis 清理失败: {e}", exc_info=True)

    return results


def _batch_invalidate_document_cache(filenames: List[str]) -> Dict[str, bool]:
    """批量使文档等级缓存失效"""
    results = {}
    doc_cache = get_document_cache()

    for filename in filenames:
        try:
            doc_cache.delete_document_level(filename)
            results[filename] = True
            logger.debug(f"文档等级缓存已清除: {filename}")
        except Exception as e:
            logger.error(f"文档等级缓存清除失败 {filename}: {e}")
            results[filename] = False

    return results


def _batch_invalidate_search_cache(filenames: List[str]) -> Dict[str, bool]:
    """批量使搜索缓存失效"""
    results = {}

    for filename in filenames:
        try:
            search_cache = CachedSearchService()
            search_cache.invalidate_cache(pattern=f"*{filename}*")
            results[filename] = True
            logger.debug(f"搜索缓存已清除: {filename}")
        except Exception as e:
            logger.warning(f"搜索缓存清除失败 {filename}: {e}")
            results[filename] = False

    return results


def _get_document_level_sync(filename: str, index_name: str) -> str:
    """同步获取文档等级（在线程池中执行）"""
    from app.service.core.cache import get_document_cache
    from app.service.core.vector_store import get_vector_store
    from pymilvus import Collection

    doc_cache = get_document_cache()

    cached_level = doc_cache.get_document_level(filename)
    if cached_level:
        return cached_level

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


# ========== 批量删除 API ==========

@router.post("/upload/delete-batch")
async def delete_documents_batch(
        request: Request,
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    批量删除文档（优化版 - 使用批量 API）
    一次性删除多个文档，大幅减少 API 调用次数
    """
    import json

    # 解析请求体
    try:
        body = await request.json()
        filenames = body.get('filenames', [])
    except:
        raise HTTPException(status_code=400, detail="请求体必须包含 filenames 数组")

    if not filenames:
        raise HTTPException(status_code=400, detail="请提供要删除的文件名列表")

    # 获取当前用户信息
    user_level = "normal"
    user_id = None
    user = None

    if authorization:
        token = authorization[7:] if authorization.startswith("Bearer ") else authorization
        user_id = get_user_id_from_token(token)
        if user_id:
            db = get_db_manager()
            user = db.get_user_by_id(user_id)
            if user:
                user_level = user.role.value
                logger.info(f"用户 {user.username} (等级={user_level}) 请求批量删除 {len(filenames)} 个文档")

    # 等级优先级
    level_priority = {"normal": 1, "admin": 2, "owner": 3}
    current_priority = level_priority.get(user_level, 1)

    # 获取索引名称
    index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")

    # ========== 第一步：并行验证用户权限 ==========
    async def get_doc_level_async(filename: str) -> tuple:
        loop = asyncio.get_event_loop()
        level = await loop.run_in_executor(
            _delete_executor,
            _get_document_level_sync,
            filename,
            index_name
        )
        return filename, level

    level_tasks = [get_doc_level_async(filename) for filename in filenames]
    level_results = await asyncio.gather(*level_tasks)

    # 过滤权限
    verified_filenames = []
    permission_denied = []
    not_found = []

    for filename, doc_level in level_results:
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            not_found.append(filename)
            continue

        can_delete, reason = _check_delete_permission(user_level, doc_level)
        if can_delete:
            verified_filenames.append(filename)
            logger.debug(f"✅ {filename}: {reason}")
        else:
            permission_denied.append({"filename": filename, "reason": reason})
            logger.warning(f"❌ {filename}: {reason}")

    if not verified_filenames:
        return {
            "success": False,
            "total": len(filenames),
            "success_count": 0,
            "fail_count": 0,
            "permission_denied": permission_denied,
            "not_found": not_found,
            "message": f"没有权限删除任何文档"
        }

    logger.info(f"权限验证完成: 可删除 {len(verified_filenames)} 个")

    # ========== 第二步：使用批量 API 并发执行 ==========
    loop = asyncio.get_event_loop()

    # 创建批量任务（每个任务处理一个存储系统）
    batch_tasks = [
        loop.run_in_executor(_delete_executor, _batch_delete_from_local_storage, verified_filenames),
        loop.run_in_executor(_delete_executor, _batch_delete_from_milvus, verified_filenames, index_name),
        loop.run_in_executor(_delete_executor, _batch_delete_from_elasticsearch, verified_filenames, index_name),
        loop.run_in_executor(_delete_executor, _batch_delete_from_redis_and_memory, verified_filenames),
        loop.run_in_executor(_delete_executor, _batch_invalidate_document_cache, verified_filenames),
        loop.run_in_executor(_delete_executor, _batch_invalidate_search_cache, verified_filenames),
    ]

    logger.info(f"开始并发批量删除 {len(verified_filenames)} 个文档 (max_workers={DELETE_MAX_WORKERS})")
    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

    # 解析结果
    local_results = batch_results[0] if isinstance(batch_results[0], dict) else {}
    milvus_results = batch_results[1] if isinstance(batch_results[1], dict) else {}
    es_results = batch_results[2] if isinstance(batch_results[2], dict) else {}
    redis_results = batch_results[3] if isinstance(batch_results[3], dict) else {}
    cache_results = batch_results[4] if isinstance(batch_results[4], dict) else {}
    search_cache_results = batch_results[5] if isinstance(batch_results[5], dict) else {}

    # ========== 第三步：汇总结果 ==========
    results = {}
    success_count = 0
    fail_count = 0

    for filename in verified_filenames:
        success = (
                local_results.get(filename, False) or
                milvus_results.get(filename, 0) > 0 or
                es_results.get(filename, 0) > 0 or
                redis_results.get(filename, 0) > 0 or
                cache_results.get(filename, False) or
                search_cache_results.get(filename, False)
        )

        results[filename] = {
            "success": success,
            "deleted": {
                "local": local_results.get(filename, False),
                "milvus": milvus_results.get(filename, 0),
                "elasticsearch": es_results.get(filename, 0),
                "redis_messages": redis_results.get(filename, 0),
                "cache": cache_results.get(filename, False),
                "search_cache": search_cache_results.get(filename, False)
            }
        }

        if success:
            success_count += 1
            logger.info(f"批量删除成功: {filename}")
        else:
            fail_count += 1
            logger.warning(f"批量删除失败: {filename}")

    from app.service.core.graphrag import get_graph_rag_service
    graph_service = get_graph_rag_service()
    graph_service.invalidate_cache(user_level)

    return {
        "success": success_count > 0,
        "total": len(filenames),
        "success_count": success_count,
        "fail_count": fail_count,
        "permission_denied": permission_denied,
        "not_found": not_found,
        "results": results,
        "message": f"成功删除 {success_count} 个文档，失败 {fail_count} 个，无权限 {len(permission_denied)} 个，不存在 {len(not_found)} 个"
    }


__all__ = ['router']