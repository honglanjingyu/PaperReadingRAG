# app/api/routes/upload.py
"""
文档上传路由
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Header, Request
from pathlib import Path
from typing import Dict, Any,Optional, List
import hashlib
import time
import shutil
import os

from app.api.config import UPLOAD_DIR, SUPPORTED_EXTENSIONS, settings, processing_status
from app.api.dependencies import get_document_service
from app.auth.jwt_utils import get_user_id_from_token
from app.db.database import get_db_manager
import re
import logging
from datetime import datetime

from app.service.core.cache import get_document_cache
from app.service.core.vector_store import get_vector_storage_service, get_vector_store
from app.service.core.retrieval.es_bm25_retriever import get_es_bm25_retriever
from app.service.core.memory import get_memory_manager



router = APIRouter()
logger = logging.getLogger(__name__)

# app/api/routes/upload.py - 修改上传接口

@router.post("/upload")
async def upload_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        chunk_size: Optional[int] = None,
        enable_vectorization: bool = True,
        enable_storage: bool = True,
        from_page: int = 0,
        to_page: Optional[int] = None,
        authorization: Optional[str] = Header(None)  # 添加认证头
) -> Dict[str, Any]:
    """
    上传并处理文档（记录上传者等级）
    """
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

    # 验证文件类型
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}。支持的类型: {list(SUPPORTED_EXTENSIONS.keys())}"
        )

    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    process_id = hashlib.md5(f"{file.filename}_{time.time()}".encode()).hexdigest()[:16]

    if chunk_size is None:
        chunk_size = settings.chunk_size

    if to_page is None:
        to_page = settings.max_pages

    processing_status[process_id] = {
        "status": "processing",
        "filename": file.filename,
        "progress": 0,
        "message": "开始处理文档...",
        "user_level": user_level  # 记录上传者等级
    }

    document_service = get_document_service()
    background_tasks.add_task(
        document_service.process_document_task,
        process_id,
        str(file_path),
        chunk_size,
        enable_vectorization,
        enable_storage,
        from_page,
        to_page,
        user_level  # 传递用户等级
    )

    return {
        "success": True,
        "process_id": process_id,
        "filename": file.filename,
        "message": "文档已上传，正在后台处理",
        "status_url": f"/api/upload/status/{process_id}"
    }


@router.get("/upload/status/{process_id}")
async def get_processing_status(process_id: str) -> Dict[str, Any]:
    """获取文档处理状态"""
    status = processing_status.get(process_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"未找到处理任务: {process_id}")

    return status


@router.get("/upload/stats/{filename}")
async def get_document_stats(filename: str) -> Dict[str, Any]:
    """获取文档统计信息"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")

    try:
        document_service = get_document_service()
        stats = document_service.get_processing_stats(str(file_path))
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


# app/api/routes/upload.py - 替换 list_documents 函数

@router.get("/upload/list")
async def list_documents(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """列出已上传的文档（根据用户等级过滤）- 使用缓存优化"""
    import logging
    import os

    from app.auth.jwt_utils import get_user_id_from_token
    from app.db.database import get_db_manager
    from app.service.core.cache import get_document_cache

    logger = logging.getLogger(__name__)

    # 获取用户等级
    user_level = "normal"
    user_id = None
    if authorization:
        token = authorization[7:] if authorization.startswith("Bearer ") else authorization
        user_id = get_user_id_from_token(token)
        if user_id:
            db = get_db_manager()
            user = db.get_user_by_id(user_id)
            if user:
                user_level = user.role.value
                logger.info(f"用户 {user.username} (等级={user_level}) 请求文档列表")

    level_priority = {"normal": 1, "admin": 2, "owner": 3}
    current_priority = level_priority.get(user_level, 1)

    # 允许访问的等级列表
    allowed_levels = [level for level, priority in level_priority.items() if priority <= current_priority]
    logger.info(f"用户等级 {user_level} (优先级={current_priority}) 允许访问的等级: {allowed_levels}")

    documents = []
    index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")
    from app.api.config import UPLOAD_DIR
    upload_dir = UPLOAD_DIR

    try:
        # 1. 快速读取 uploads 目录中的文件列表
        if not upload_dir.exists():
            logger.warning(f"上传目录不存在: {upload_dir}")
            return {"success": True, "total": 0, "documents": []}

        # 收集文件信息
        file_infos = []
        for file_path in upload_dir.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    file_infos.append({
                        "filename": file_path.name,
                        "size": file_path.stat().st_size,
                        "created": file_path.stat().st_ctime,
                    })

        if not file_infos:
            logger.info("没有找到支持的文件")
            return {"success": True, "total": 0, "documents": []}

        # 2. 使用缓存获取文档等级
        doc_cache = get_document_cache()
        filenames = [f["filename"] for f in file_infos]

        # 从缓存批量获取
        cached_levels, missing_filenames = doc_cache.batch_get_levels(filenames)

        logger.info(f"缓存状态: 命中 {len(cached_levels)} 个, 缺失 {len(missing_filenames)} 个")

        # 3. 只查询缺失的文档等级（从 Milvus）
        doc_level_map = cached_levels.copy()

        if missing_filenames:
            try:
                from app.service.core.vector_store import get_vector_store
                from pymilvus import Collection

                store = get_vector_store()

                if store and store.index_exists(index_name):
                    collection = Collection(index_name)
                    collection.load()

                    # 分批查询，避免 SQL 过长
                    batch_size = 50
                    for i in range(0, len(missing_filenames), batch_size):
                        batch_names = missing_filenames[i:i + batch_size]
                        # 构建查询表达式
                        names_str = ', '.join([f'"{name}"' for name in batch_names])
                        expr = f"docnm in [{names_str}]"

                        results = collection.query(
                            expr=expr,
                            output_fields=["docnm", "user_level"],
                            limit=len(batch_names) * 10
                        )

                        # 去重，取第一个找到的等级
                        found_in_batch = {}
                        for result in results:
                            docnm = result.get("docnm", "")
                            level = result.get("user_level", "normal")
                            if docnm and docnm not in found_in_batch:
                                found_in_batch[docnm] = level

                        # 更新映射并写入缓存
                        for docnm, level in found_in_batch.items():
                            doc_level_map[docnm] = level
                            doc_cache.set_document_level(docnm, level)

                        # 记录未找到的文档
                        for filename in batch_names:
                            if filename not in found_in_batch:
                                doc_level_map[filename] = "normal"
                                doc_cache.set_document_level(filename, "normal")
                                logger.info(f"文档 {filename} 不在 Milvus 中，默认等级 normal")

                    logger.info(
                        f"从 Milvus 查询完成: 新增 {len([k for k in doc_level_map.keys() if k not in cached_levels])} 个文档等级")
                else:
                    # 索引不存在，所有缺失文档默认为 normal
                    for filename in missing_filenames:
                        doc_level_map[filename] = "normal"
                        doc_cache.set_document_level(filename, "normal")
                    logger.warning(f"索引 {index_name} 不存在，所有文档等级设为 normal")

            except Exception as e:
                logger.error(f"从 Milvus 获取文档等级失败: {e}")
                import traceback
                traceback.print_exc()
                # 降级：所有缺失文档默认为 normal
                for filename in missing_filenames:
                    doc_level_map[filename] = "normal"

        # 4. 根据用户权限过滤文档
        for file_info in file_infos:
            filename = file_info["filename"]
            doc_level = doc_level_map.get(filename, "normal")
            doc_priority = level_priority.get(doc_level, 1)

            if current_priority >= doc_priority:
                documents.append({
                    "filename": filename,
                    "size": file_info["size"],
                    "created": file_info["created"],
                    "user_level": doc_level
                })
                logger.debug(f"  包含文档: {filename} (等级={doc_level})")
            else:
                logger.debug(f"  跳过文档: {filename} (等级={doc_level}, 权限不足)")

        # 按创建时间倒序排序
        documents.sort(key=lambda x: x.get("created", 0), reverse=True)

        logger.info(f"返回 {len(documents)} 个文档给用户等级 {user_level}")

        return {
            "success": True,
            "total": len(documents),
            "documents": documents,
            "cache_info": {
                "hits": len(cached_levels),
                "misses": len(missing_filenames),
                "ttl": doc_cache.cache_ttl
            }
        }

    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        import traceback
        traceback.print_exc()

        # 降级：只返回 uploads 目录中的文件（不过滤等级）
        documents = []
        if upload_dir.exists():
            for file_path in upload_dir.iterdir():
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        documents.append({
                            "filename": file_path.name,
                            "size": file_path.stat().st_size,
                            "created": file_path.stat().st_ctime,
                            "user_level": "unknown"
                        })
        documents.sort(key=lambda x: x.get("created", 0), reverse=True)

        return {
            "success": True,
            "total": len(documents),
            "documents": documents,
            "warning": "等级过滤暂时不可用"
        }


def _delete_from_milvus(filename: str, index_name: str) -> int:
    """从 Milvus 删除文档的所有分块"""
    try:
        store = get_vector_store()
        if store and store.index_exists(index_name):
            # 删除 docnm 匹配的所有文档
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
            # 需要先查询所有匹配的文档ID
            es_index = f"rag_bm25_{index_name}"

            # 使用 scroll 查询获取所有匹配的文档ID
            from elasticsearch import Elasticsearch
            client = es_retriever._client

            if client and client.indices.exists(index=es_index):
                # 查询所有匹配的文档
                query = {
                    "query": {
                        "term": {"document_name": filename}
                    },
                    "_source": False  # 只返回ID
                }

                # 使用 scroll 分页获取所有结果
                response = client.search(
                    index=es_index,
                    body=query,
                    scroll="2m",
                    size=1000
                )

                scroll_id = response.get("_scroll_id")
                hits = response.get("hits", {}).get("hits", [])
                doc_ids = [hit["_id"] for hit in hits]

                # 继续 scroll 获取剩余结果
                while hits:
                    response = client.scroll(scroll_id=scroll_id, scroll="2m")
                    scroll_id = response.get("_scroll_id")
                    hits = response.get("hits", {}).get("hits", [])
                    doc_ids.extend([hit["_id"] for hit in hits])

                # 清除 scroll
                if scroll_id:
                    client.clear_scroll(scroll_id=scroll_id)

                # 批量删除文档
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
    """
    从 Redis 和对话记忆中删除与文档相关的历史
    修复：正确处理不同数据类型的 key，同时删除用户问题和对应的回答
    """
    try:
        memory_manager = get_memory_manager()
        if not memory_manager or not memory_manager.redis_client:
            logger.warning("Redis 不可用，跳过对话历史清理")
            return 0

        deleted_count = 0
        redis_client = memory_manager.redis_client

        # 获取所有会话相关的 key
        pattern = f"{memory_manager._key_prefix}*"
        meta_pattern = f"{memory_manager._meta_prefix}*"

        all_keys = redis_client.keys(pattern)
        meta_keys = redis_client.keys(meta_pattern)

        # 合并所有需要检查的 key
        all_session_keys = list(set(all_keys + meta_keys))

        logger.info(f"找到 {len(all_session_keys)} 个会话相关 key")

        for session_key in all_session_keys:
            try:
                # 检查 key 类型
                key_type = redis_client.type(session_key)
                logger.debug(f"检查 key: {session_key}, type: {key_type}")

                if key_type == 'list':
                    # 处理 List 类型（消息列表）
                    messages = redis_client.lrange(session_key, 0, -1)
                    if not messages:
                        continue

                    import json

                    # 先解析所有消息
                    parsed_messages = []
                    for msg_json in messages:
                        try:
                            parsed_messages.append(json.loads(msg_json))
                        except:
                            parsed_messages.append({"role": "unknown", "content": msg_json})

                    # 找出需要删除的索引
                    indices_to_remove = set()

                    for idx, msg in enumerate(parsed_messages):
                        content = msg.get("content", "")
                        role = msg.get("role", "")

                        # 检查是否引用了该文档
                        if filename in content or f"文档「{filename}」" in content:
                            indices_to_remove.add(idx)
                            # 如果是助手回答，同时删除它前面的用户问题
                            if role == "assistant" and idx > 0:
                                prev_msg = parsed_messages[idx - 1]
                                if prev_msg.get("role") == "user":
                                    indices_to_remove.add(idx - 1)
                                    logger.debug(
                                        f"找到匹配的对话对: session={session_key}, user_idx={idx - 1}, assistant_idx={idx}")
                            logger.debug(f"找到匹配消息: session={session_key}, idx={idx}")

                    # 删除匹配的消息
                    if indices_to_remove:
                        valid_messages = []
                        for idx, msg_json in enumerate(messages):
                            if idx not in indices_to_remove:
                                valid_messages.append(msg_json)

                        redis_client.delete(session_key)
                        if valid_messages:
                            redis_client.rpush(session_key, *valid_messages)
                            logger.debug(
                                f"会话 {session_key} 已更新: 删除 {len(indices_to_remove)} 条，保留 {len(valid_messages)} 条")
                        else:
                            logger.debug(f"会话 {session_key} 已清空")

                        deleted_count += len(indices_to_remove)

                elif key_type == 'hash':
                    # 处理 Hash 类型（元数据）
                    meta_data = redis_client.hgetall(session_key)
                    if meta_data:
                        has_doc_ref = False
                        for value in meta_data.values():
                            if filename in str(value):
                                has_doc_ref = True
                                break

                        if has_doc_ref:
                            redis_client.delete(session_key)
                            deleted_count += 1
                            logger.debug(f"删除元数据: {session_key}")

                elif key_type == 'string':
                    value = redis_client.get(session_key)
                    if value and filename in str(value):
                        redis_client.delete(session_key)
                        deleted_count += 1
                        logger.debug(f"删除字符串 key: {session_key}")

            except Exception as e:
                logger.warning(f"处理 key {session_key} 时出错: {e}")
                continue

        logger.info(f"Redis 清理完成: 删除了 {deleted_count} 条相关记录")
        return deleted_count

    except Exception as e:
        logger.error(f"Redis 清理失败: {e}", exc_info=True)
        return 0

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


@router.delete("/upload/{filename}")
async def delete_document(
        filename: str,
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    删除文档（同时删除 Milvus、Elasticsearch、Redis、本地存储的所有相关数据）
    """
    from app.api.config import UPLOAD_DIR
    from app.service.core.cache import get_document_cache
    from app.service.core.vector_store import get_vector_storage_service
    from app.db.database import get_db_manager

    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")

    # 获取索引名称
    index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")

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

    details = []

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
        doc_cache = get_document_cache()
        doc_cache.delete_document_level(filename)
        result["deleted"]["cache"] = True
        details.append("✅ 文档等级缓存已清除")

        # 6. 使搜索缓存失效
        try:
            from app.service.core.rag.cached_search import CachedSearchService
            search_cache = CachedSearchService()
            search_cache.invalidate_cache(pattern=f"*{filename}*")
            details.append("✅ 搜索缓存已清除")
        except Exception as e:
            logger.warning(f"搜索缓存清除失败: {e}")

        # 7. 从数据库中删除会话关联（可选）
        try:
            db = get_db_manager()
            # 查找包含该文档名的会话并解除关联
            # 注意：这取决于你如何存储文档与会话的关联
            # 如果 user_sessions 表中没有文档关联，可以跳过
        except Exception as e:
            logger.warning(f"数据库会话关联清理失败: {e}")

        result["message"] = "\n".join(details)
        logger.info(f"文档删除完成: {filename}, 详情: {result['deleted']}")

        return result

    except Exception as e:
        logger.error(f"删除文档失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")

@router.post("/upload/delete-batch")
async def delete_documents_batch(
        request: Request,
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    批量删除文档（所有用户都可以批量删除，但只能删除自己有权限的文档）
    """
    from app.auth.jwt_utils import get_user_id_from_token
    from app.db.database import get_db_manager
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

    # 允许用户删除的等级（只能删除等级 <= 自己等级的文档）
    allowed_levels = [level for level, priority in level_priority.items() if priority <= current_priority]

    # 获取索引名称
    index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")

    # 第一步：验证用户权限，过滤出有权限删除的文档
    verified_filenames = []
    permission_denied = []
    not_found = []

    for filename in filenames:
        # 检查文件是否存在
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            not_found.append(filename)
            continue

        # 获取文档等级
        doc_level = "normal"
        try:
            # 从缓存或 Milvus 获取文档等级
            doc_cache = get_document_cache()
            cached_level = doc_cache.get_document_level(filename)

            if cached_level:
                doc_level = cached_level
            else:
                # 从 Milvus 查询
                store = get_vector_store()
                if store and store.index_exists(index_name):
                    from pymilvus import Collection
                    collection = Collection(index_name)
                    collection.load()

                    # 查询文档等级
                    expr = f'docnm == "{filename}"'
                    results = collection.query(
                        expr=expr,
                        output_fields=["docnm", "user_level"],
                        limit=1
                    )
                    if results:
                        doc_level = results[0].get("user_level", "normal")
                        # 缓存结果
                        doc_cache.set_document_level(filename, doc_level)
        except Exception as e:
            logger.warning(f"获取文档等级失败 {filename}: {e}")

        # 检查权限：用户等级必须 >= 文档等级
        doc_priority = level_priority.get(doc_level, 1)
        if current_priority >= doc_priority:
            verified_filenames.append(filename)
        else:
            permission_denied.append(filename)

    # 第二步：执行删除操作
    results = {}
    success_count = 0
    fail_count = 0

    for filename in verified_filenames:
        try:
            # 删除各处的数据
            local_deleted = _delete_from_local_storage(filename)
            milvus_deleted = _delete_from_milvus(filename, index_name)
            es_deleted = _delete_from_elasticsearch(filename, index_name)
            redis_deleted = _delete_from_redis_and_memory(filename)

            # 清除缓存
            doc_cache = get_document_cache()
            doc_cache.delete_document_level(filename)

            # 清除搜索缓存
            try:
                from app.service.core.rag.cached_search import CachedSearchService
                search_cache = CachedSearchService()
                search_cache.invalidate_cache(pattern=f"*{filename}*")
            except Exception as e:
                logger.warning(f"搜索缓存清除失败: {e}")

            results[filename] = {
                "success": True,
                "deleted": {
                    "local": local_deleted,
                    "milvus": milvus_deleted,
                    "elasticsearch": es_deleted,
                    "redis_messages": redis_deleted
                }
            }
            success_count += 1
            logger.info(f"批量删除成功: {filename}")

        except Exception as e:
            results[filename] = {"success": False, "error": str(e)}
            fail_count += 1
            logger.error(f"批量删除失败 {filename}: {e}")

    # 返回结果
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