# app/api/routes/delete.py
"""文档删除路由 - 包含单个删除和批量删除功能"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException, Header, BackgroundTasks
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from app.api.config import UPLOAD_DIR
from app.auth.jwt_utils import get_user_id_from_token
from app.db.database import get_db_manager
from app.service.core.cache import get_document_cache
from app.service.core.vector_store import get_vector_store
from app.service.core.retrieval import get_es_bm25_retriever
from app.service.core.memory import get_memory_manager
from app.service.core.rag.cached_search import CachedSearchService
from app.service.core.graphrag import get_graph_rag_service
from app.service.core.streaming import get_stream_processor, is_kafka_enabled

router = APIRouter()
logger = logging.getLogger(__name__)

# 创建删除专用的线程池
DELETE_MAX_WORKERS = int(os.getenv("ASYNC_PROCESSOR_MAX_WORKERS", "3"))
_delete_executor = ThreadPoolExecutor(max_workers=DELETE_MAX_WORKERS)
logger.info(f"批量删除线程池初始化完成，max_workers={DELETE_MAX_WORKERS}")


# ========== 辅助函数 ==========

def _get_user_info_from_token(authorization: Optional[str]) -> tuple:
    """从token获取用户信息"""
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

    return user_level, user_id, username


def _check_delete_permission(user_level: str, doc_level: str) -> tuple:
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


def _delete_from_local_storage(filename: str) -> bool:
    """删除本地存储的文件（包括提取的临时文本文件）"""
    try:
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()
            logger.info(f"本地文件已删除: {filename}")

        # 删除对应的提取文字临时文件
        temp_text_path = UPLOAD_DIR / f"{filename}.extracted.txt"
        if temp_text_path.exists():
            temp_text_path.unlink()
            logger.info(f"临时文本文件已删除: {filename}.extracted.txt")

        return True
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


def _invalidate_cache(filename: str) -> bool:
    """使文档相关缓存失效（包括多模态缓存）"""
    try:
        doc_cache = get_document_cache()
        doc_cache.delete_document_level(filename)

        # 使用正确的方法名清理搜索缓存
        from app.service.core.rag.cached_search import CachedSearchService
        search_cache = CachedSearchService()

        # 检查方法是否存在
        if hasattr(search_cache, 'invalidate'):
            search_cache.invalidate(pattern=f"*{filename}*")
        elif hasattr(search_cache, 'invalidate_cache'):
            search_cache.invalidate_cache(pattern=f"*{filename}*")
        else:
            # 直接删除缓存
            from app.service.core.cache import get_cache_manager
            cache = get_cache_manager()
            cache.delete_pattern(f"search:*{filename}*")

        # 清理多模态缓存
        from app.service.core.cache import get_cache_manager
        cache = get_cache_manager()
        cache.delete("multimodal", filename)

        return True
    except Exception as e:
        logger.error(f"缓存失效失败 {filename}: {e}")
        return False


# ========== 后台任务函数 ==========

async def send_kafka_delete_event_background(
        filename: str,
        user_level: str,
        chunk_ids: List[str] = None
):
    """后台发送 Kafka delete 事件"""
    try:
        logger.info(f"🚀 [Kafka后台] 开始发送 delete 事件: filename={filename}, "
                    f"user_level={user_level}, chunk_count={len(chunk_ids) if chunk_ids else 0}")

        if not is_kafka_enabled():
            logger.warning(f"⚠️ [Kafka后台] Kafka 未启用，跳过发送: {filename}")
            return

        processor = get_stream_processor()
        result = await processor.emit_change_event(
            filename=filename,
            content="",
            user_level=user_level,
            file_path=None,
            event_type="delete"
        )

        if result:
            logger.info(f"✅ [Kafka后台] delete 事件发送成功: {filename}")
        else:
            logger.warning(f"⚠️ [Kafka后台] delete 事件发送失败: {filename}")

    except Exception as e:
        logger.error(f"❌ [Kafka后台] delete 事件发送异常: {filename}, error={e}", exc_info=True)


# ========== 批量删除辅助函数 ==========

def _batch_delete_from_local_storage(filenames: List[str]) -> Dict[str, bool]:
    """批量删除本地文件（包括临时文件）"""
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

            temp_text_path = UPLOAD_DIR / f"{filename}.extracted.txt"
            if temp_text_path.exists():
                temp_text_path.unlink()

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

            for filename in filenames:
                count_expr = f"docnm == '{filename}'"
                result = collection.query(
                    expr=count_expr,
                    output_fields=["docnm"],
                    limit=10000
                )
                results[filename] = len(result)

            expr_parts = [f"docnm == '{filename}'" for filename in filenames]
            expr = " or ".join(expr_parts)
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


def _batch_invalidate_cache(filenames: List[str]) -> Dict[str, bool]:
    """批量使缓存失效"""
    results = {}
    doc_cache = get_document_cache()

    for filename in filenames:
        try:
            doc_cache.delete_document_level(filename)

            # 使用与 _invalidate_cache 相同的方式
            from app.service.core.rag.cached_search import CachedSearchService
            search_cache = CachedSearchService()

            if hasattr(search_cache, 'invalidate'):
                search_cache.invalidate(pattern=f"*{filename}*")
            elif hasattr(search_cache, 'invalidate_cache'):
                search_cache.invalidate_cache(pattern=f"*{filename}*")
            else:
                from app.service.core.cache import get_cache_manager
                cache = get_cache_manager()
                cache.delete_pattern(f"search:*{filename}*")

            # 清理多模态缓存
            from app.service.core.cache import get_cache_manager
            cache = get_cache_manager()
            cache.delete("multimodal", filename)

            results[filename] = True
        except Exception as e:
            logger.error(f"缓存失效失败 {filename}: {e}")
            results[filename] = False

    return results


# ========== 单个文档删除 ==========

@router.delete("/upload/{filename}")
async def delete_document(
        filename: str,
        background_tasks: BackgroundTasks,
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    删除单个文档 - 通过 Kafka 异步删除

    流程：
    1. 验证用户权限
    2. 发送删除事件到 Kafka
    3. Kafka 消费者执行实际删除操作

    注意：Kafka 必须启用，否则返回错误
    """
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")

    index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")

    user_level, user_id, username = _get_user_info_from_token(authorization)

    # 未登录不允许删除
    if user_id is None:
        raise HTTPException(status_code=401, detail="请先登录")

    logger.info(f"用户 {username} (等级={user_level}) 请求删除文档: {filename}")

    # ========== 检查 Kafka 是否启用 ==========
    if not is_kafka_enabled():
        logger.error(f"Kafka 未启用，无法删除文档: {filename}")
        raise HTTPException(
            status_code=503,
            detail="Kafka 服务未启用，无法执行删除操作。请检查 Kafka 配置。"
        )

    # 获取文档等级
    doc_level = _get_document_level_sync(filename, index_name)
    logger.info(f"文档 {filename}: 等级={doc_level}")

    # 验证权限
    can_delete, reason = _check_delete_permission(user_level, doc_level)
    if not can_delete:
        logger.warning(f"用户 {username} (等级={user_level}) 无权删除 {filename} (等级={doc_level}): {reason}")
        raise HTTPException(status_code=403, detail=reason)

    logger.info(f"权限验证通过: {reason}")

    # 获取 chunk_ids（可选，用于更精确的删除）
    chunk_ids = []
    try:
        store = get_vector_store()
        if store and store.index_exists(index_name):
            from pymilvus import Collection
            collection = Collection(index_name)
            collection.load()
            expr = f'docnm == "{filename}"'
            query_results = collection.query(
                expr=expr,
                output_fields=["id"],
                limit=10000
            )
            chunk_ids = [r.get("id", "") for r in query_results if r.get("id")]
            logger.info(f"获取到 {len(chunk_ids)} 个 chunk_id 用于 Kafka 事件")
    except Exception as e:
        logger.warning(f"获取 chunk_ids 失败: {e}")

    # 发送删除事件到 Kafka（消费者会执行实际删除）
    background_tasks.add_task(
        send_kafka_delete_event_background,
        filename=filename,
        user_level=user_level,
        chunk_ids=chunk_ids
    )

    logger.info(f"📋 [Kafka] delete 事件已加入后台队列: {filename}")

    # 立即失效文档等级缓存（避免前端显示问题）
    doc_cache = get_document_cache()
    doc_cache.delete_document_level(filename)
    logger.info(f"文档等级缓存已删除: {filename}")

    # 使 GraphRAG 缓存失效
    graph_service = get_graph_rag_service()
    graph_service.invalidate_cache(user_level)

    return {
        "success": True,
        "filename": filename,
        "user_level": user_level,
        "doc_level": doc_level,
        "message": f"删除请求已发送到 Kafka 队列，文档 {filename} 将被异步删除",
        "kafka_sent": True,
        "chunk_count": len(chunk_ids)
    }


# ========== 批量删除 ==========

@router.post("/upload/delete-batch")
async def delete_documents_batch(
        request: Request,
        background_tasks: BackgroundTasks,
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    批量删除文档（优化版 - 使用批量 API）
    一次性删除多个文档，大幅减少 API 调用次数
    """
    # 解析请求体
    try:
        body = await request.json()
        filenames = body.get('filenames', [])
    except:
        raise HTTPException(status_code=400, detail="请求体必须包含 filenames 数组")

    if not filenames:
        raise HTTPException(status_code=400, detail="请提供要删除的文件名列表")

    user_level, user_id, username = _get_user_info_from_token(authorization)

    logger.info(f"用户 {username or 'unknown'} (等级={user_level}) 请求批量删除 {len(filenames)} 个文档")

    index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")

    # 第一步：并行验证用户权限
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

    # 第二步：使用批量 API 并发执行
    loop = asyncio.get_event_loop()

    batch_tasks = [
        loop.run_in_executor(_delete_executor, _batch_delete_from_local_storage, verified_filenames),
        loop.run_in_executor(_delete_executor, _batch_delete_from_milvus, verified_filenames, index_name),
        loop.run_in_executor(_delete_executor, _batch_delete_from_elasticsearch, verified_filenames, index_name),
        loop.run_in_executor(_delete_executor, _batch_delete_from_redis_and_memory, verified_filenames),
        loop.run_in_executor(_delete_executor, _batch_invalidate_cache, verified_filenames),
    ]

    logger.info(f"开始并发批量删除 {len(verified_filenames)} 个文档 (max_workers={DELETE_MAX_WORKERS})")
    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

    local_results = batch_results[0] if isinstance(batch_results[0], dict) else {}
    milvus_results = batch_results[1] if isinstance(batch_results[1], dict) else {}
    es_results = batch_results[2] if isinstance(batch_results[2], dict) else {}
    redis_results = batch_results[3] if isinstance(batch_results[3], dict) else {}
    cache_results = batch_results[4] if isinstance(batch_results[4], dict) else {}

    # 第三步：汇总结果
    results = {}
    success_count = 0
    fail_count = 0

    for filename in verified_filenames:
        success = (
                local_results.get(filename, False) or
                milvus_results.get(filename, 0) > 0 or
                es_results.get(filename, 0) > 0 or
                redis_results.get(filename, 0) > 0 or
                cache_results.get(filename, False)
        )

        results[filename] = {
            "success": success,
            "deleted": {
                "local": local_results.get(filename, False),
                "milvus": milvus_results.get(filename, 0),
                "elasticsearch": es_results.get(filename, 0),
                "redis_messages": redis_results.get(filename, 0),
                "cache": cache_results.get(filename, False)
            }
        }

        if success:
            success_count += 1
            logger.info(f"批量删除成功: {filename}")
        else:
            fail_count += 1
            logger.warning(f"批量删除失败: {filename}")

    graph_service = get_graph_rag_service()
    graph_service.invalidate_cache(user_level)

    # ========== 批量发送 Kafka 删除事件 ==========
    if is_kafka_enabled() and verified_filenames:
        logger.info(f"📋 [Kafka] 将 {len(verified_filenames)} 个 delete 事件加入后台队列")
        for filename in verified_filenames:
            background_tasks.add_task(
                send_kafka_delete_event_background,
                filename=filename,
                user_level=user_level,
                chunk_ids=[]
            )
        logger.info(f"✅ [Kafka] {len(verified_filenames)} 个 delete 事件已加入后台队列")

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