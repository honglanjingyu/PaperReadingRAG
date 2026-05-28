# app/api/routes/upload.py
# 完整修复版 - 包含 /upload/async 和 /upload/task/{task_id} 路由

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Header, Query
from pathlib import Path
from typing import List, Optional, Dict, Any
import shutil
import os
import logging
import asyncio
import time
import hashlib

from app.api.config import UPLOAD_DIR, SUPPORTED_EXTENSIONS, MEDIA_TYPE_MAP, settings
from app.api.dependencies import get_document_service
from app.auth.jwt_utils import get_user_id_from_token
from app.db.database import get_db_manager
from app.service.core.multimodal import get_multimodal_parser, ExtractedContent
from pydantic import BaseModel, Field
from app.service.core.graphrag import get_graph_rag_service
from app.service.core.streaming import get_stream_processor, is_kafka_enabled
from app.service.core.cache import get_cache_manager, get_document_cache

router = APIRouter()
logger = logging.getLogger(__name__)


# ========== 辅助函数 ==========

def _get_user_level_from_token(authorization: Optional[str]) -> tuple:
    """从token获取用户等级和用户ID"""
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


def _validate_file(file: UploadFile) -> tuple:
    """验证文件并返回扩展名"""
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}。支持的类型: {list(SUPPORTED_EXTENSIONS.keys())}"
        )
    return file_ext


def _save_upload_file(file: UploadFile, file_path: Path) -> None:
    """保存上传的文件"""
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")


def _invalidate_graph_cache(user_level: str) -> None:
    """使 GraphRAG 缓存失效"""
    try:
        graph_service = get_graph_rag_service()
        graph_service.invalidate_cache(user_level)
    except Exception as e:
        logger.warning(f"GraphRAG 缓存失效失败: {e}")


def _extract_text_from_file(file_path: str, file_name: str) -> ExtractedContent:
    """从文件中提取文字内容（支持多模态）"""
    parser = get_multimodal_parser()
    return parser.parse(file_path, file_name)


# ========== 后台发送 Kafka 事件（带任务状态更新） ==========

async def send_to_kafka(
        filename: str,
        content: str,
        user_level: str,
        file_path: Optional[str] = None,
        media_type: str = "document",
        task_id: str = None
):
    """发送文档到 Kafka 处理，并更新任务状态"""
    cache = get_cache_manager()

    try:
        # 更新任务状态：开始处理
        if task_id:
            existing = cache.get("task", task_id) or {}
            cache.set("task", task_id, {
                "status": "processing",
                "filename": filename,
                "media_type": media_type,
                "progress": 10,
                "message": "正在发送到 Kafka 队列...",
                "updated_at": time.time(),
                "created_at": existing.get("created_at", time.time())
            }, ttl=3600)

        logger.info(f"🚀 [Kafka] 发送文档: filename={filename}, "
                    f"user_level={user_level}, content_length={len(content) if content else 0}, "
                    f"media_type={media_type}, task_id={task_id}")

        if not is_kafka_enabled():
            logger.error(f"❌ [Kafka] Kafka 未启用，无法处理: {filename}")
            if task_id:
                cache.set("task", task_id, {
                    "status": "failed",
                    "filename": filename,
                    "media_type": media_type,
                    "progress": 0,
                    "message": "Kafka 服务未启用",
                    "error": "Kafka 服务未启用",
                    "updated_at": time.time()
                }, ttl=3600)
            return

        processor = get_stream_processor()
        result = await processor.emit_change_event(
            filename=filename,
            content=content,
            user_level=user_level,
            file_path=file_path,
            event_type="upsert"
        )

        if result:
            logger.info(f"✅ [Kafka] 文档发送成功: {filename}")
            if task_id:
                cache.set("task", task_id, {
                    "status": "processing",
                    "filename": filename,
                    "media_type": media_type,
                    "progress": 50,
                    "message": "文档已发送到 Kafka，等待处理...",
                    "updated_at": time.time()
                }, ttl=3600)
        else:
            logger.error(f"❌ [Kafka] 文档发送失败: {filename}")
            if task_id:
                cache.set("task", task_id, {
                    "status": "failed",
                    "filename": filename,
                    "media_type": media_type,
                    "progress": 0,
                    "message": "发送到 Kafka 失败",
                    "error": "Kafka 发送失败",
                    "updated_at": time.time()
                }, ttl=3600)

    except Exception as e:
        logger.error(f"❌ [Kafka] 发送异常: {filename}, error={e}", exc_info=True)
        if task_id:
            cache.set("task", task_id, {
                "status": "failed",
                "filename": filename,
                "media_type": media_type,
                "progress": 0,
                "message": f"发送异常: {str(e)}",
                "error": str(e),
                "updated_at": time.time()
            }, ttl=3600)


# ========== 异步上传接口 ==========

@router.post("/upload/async")
async def upload_document_async(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        chunk_size: Optional[int] = None,
        from_page: int = 0,
        to_page: Optional[int] = None,
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    异步上传并处理文档 - 通过 Kafka 异步处理
    """
    from app.service.core.cache import get_cache_manager

    user_level, user_id, username = _get_user_level_from_token(authorization)

    # 验证文件类型
    file_ext = _validate_file(file)
    media_type = MEDIA_TYPE_MAP.get(file_ext, "document")

    logger.info(f"用户 {username} (等级={user_level}) 上传文件: {file.filename}, 类型={media_type}")

    # 检查 Kafka 是否启用
    if not is_kafka_enabled():
        logger.error(f"Kafka 未启用，无法处理文档: {file.filename}")
        raise HTTPException(
            status_code=503,
            detail="Kafka 服务未启用，无法处理文档。请检查 Kafka 配置。"
        )

    file_path = UPLOAD_DIR / file.filename
    if file_path.exists():
        raise HTTPException(status_code=400, detail=f"文件已存在: {file.filename}")

    # 保存原始文件
    _save_upload_file(file, file_path)

    # 多模态预处理：提取文字内容
    extracted = _extract_text_from_file(str(file_path), file.filename)

    if not extracted.success:
        # 解析失败，删除文件
        file_path.unlink()
        raise HTTPException(status_code=400, detail=f"文件解析失败: {extracted.error}")

    # 记录多模态元数据到缓存
    cache = get_cache_manager()
    cache.set("multimodal", file.filename, {
        "media_type": media_type,
        "ocr_confidence": extracted.ocr_confidence if media_type == "image" else 0,
        "transcript_confidence": extracted.transcript_confidence if media_type in ["audio", "video"] else 0,
        "duration": extracted.duration_seconds if media_type in ["audio", "video"] else 0,
        "original_filename": file.filename,
        "extracted_text_length": len(extracted.text_content),
    }, ttl=86400)

    # 使 GraphRAG 缓存失效
    _invalidate_graph_cache(user_level)

    # 准备 Kafka 消息
    kafka_file_path = str(file_path) if media_type not in ["image", "audio", "video"] else None

    # 生成任务 ID
    task_id = hashlib.md5(f"{file.filename}_{user_level}".encode()).hexdigest()[:16]

    # 记录初始任务状态
    cache.set("task", task_id, {
        "status": "pending",
        "filename": file.filename,
        "media_type": media_type,
        "progress": 0,
        "message": "文件已接收，等待处理...",
        "created_at": time.time(),
        "updated_at": time.time()
    }, ttl=3600)

    # 发送到 Kafka，传递 task_id
    background_tasks.add_task(
        send_to_kafka,
        filename=file.filename,
        content=extracted.text_content,
        user_level=user_level,
        file_path=kafka_file_path,
        media_type=media_type,
        task_id=task_id
    )

    logger.info(f"📋 [Kafka] upsert 事件已加入后台队列: {file.filename}, task_id={task_id}")

    return {
        "success": True,
        "task_id": task_id,
        "filename": file.filename,
        "media_type": media_type,
        "extracted_text_length": len(extracted.text_content),
        "message": f"文件已发送到 Kafka 处理队列{'（已通过OCR/ASR提取文字）' if media_type in ['image', 'audio', 'video'] else ''}",
        "kafka_sent": True,
        "status_url": f"/api/upload/task/{task_id}"
    }


# ========== 任务状态查询接口 ==========

@router.get("/upload/task/{task_id}")
async def get_task_status(
        task_id: str,
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    查询异步任务状态

    Args:
        task_id: 任务ID

    Returns:
        任务状态信息
    """
    from app.service.core.cache import get_cache_manager

    cache = get_cache_manager()
    task_data = cache.get("task", task_id)

    if not task_data:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    # 可选：验证用户权限
    if authorization:
        user_level, user_id, username = _get_user_level_from_token(authorization)
        # 可以在这里验证用户是否有权查看此任务

    return {
        "success": True,
        "task_id": task_id,
        "status": task_data.get("status", "unknown"),
        "progress": task_data.get("progress", 0),
        "message": task_data.get("message", ""),
        "media_type": task_data.get("media_type", "document"),
        "filename": task_data.get("filename", ""),
        "error": task_data.get("error"),
        "result": task_data.get("result", {}),
        "created_at": task_data.get("created_at"),
        "updated_at": task_data.get("updated_at")
    }


# ========== 同步上传接口（保留兼容） ==========

@router.post("/upload")
async def upload_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        chunk_size: Optional[int] = None,
        from_page: int = 0,
        to_page: Optional[int] = None,
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    同步上传接口 - 直接返回结果（无轮询）
    """
    from app.service.core.rag import process_document_with_text

    user_level, user_id, username = _get_user_level_from_token(authorization)

    # 验证文件类型
    file_ext = _validate_file(file)
    media_type = MEDIA_TYPE_MAP.get(file_ext, "document")

    logger.info(f"用户 {username} (等级={user_level}) 上传文件: {file.filename}, 类型={media_type}")

    file_path = UPLOAD_DIR / file.filename
    if file_path.exists():
        raise HTTPException(status_code=400, detail=f"文件已存在: {file.filename}")

    # 保存原始文件
    _save_upload_file(file, file_path)

    # 多模态预处理：提取文字内容
    extracted = _extract_text_from_file(str(file_path), file.filename)

    if not extracted.success:
        file_path.unlink()
        raise HTTPException(status_code=400, detail=f"文件解析失败: {extracted.error}")

    # 直接处理文档（同步）
    chunk_count = process_document_with_text(
        text_content=extracted.text_content,
        file_name=file.filename,
        enable_vectorization=True,
        enable_storage=True,
        user_level=user_level
    )

    if not chunk_count:
        raise HTTPException(status_code=500, detail="文档处理失败")

    # 记录元数据
    cache = get_cache_manager()
    cache.set("multimodal", file.filename, {
        "media_type": media_type,
        "ocr_confidence": extracted.ocr_confidence if media_type == "image" else 0,
        "transcript_confidence": extracted.transcript_confidence if media_type in ["audio", "video"] else 0,
        "duration": extracted.duration_seconds if media_type in ["audio", "video"] else 0,
        "original_filename": file.filename,
        "extracted_text_length": len(extracted.text_content),
    }, ttl=86400)

    _invalidate_graph_cache(user_level)

    return {
        "success": True,
        "filename": file.filename,
        "media_type": media_type,
        "chunk_count": chunk_count,
        "message": f"文件处理完成，共 {chunk_count} 个分块"
    }


# ========== 文档列表接口 ==========

@router.get("/upload/list")
async def list_documents(
        authorization: Optional[str] = Header(None),
        page: int = Query(1, ge=1, description="页码"),
        page_size: int = Query(20, ge=1, le=100, description="每页数量")
) -> Dict[str, Any]:
    """列出已上传的文档（分页）"""
    from app.auth.jwt_utils import get_user_id_from_token
    from app.db.database import get_db_manager
    from app.service.core.cache import get_document_cache
    from app.service.core.vector_store import get_vector_store
    from pymilvus import Collection

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

    level_priority = {"normal": 1, "admin": 2, "owner": 3}
    current_priority = level_priority.get(user_level, 1)

    documents = []
    index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")
    upload_dir = UPLOAD_DIR

    cache = get_cache_manager()

    try:
        if not upload_dir.exists():
            return {"success": True, "total": 0, "documents": [], "page": page, "page_size": page_size}

        file_infos = []
        for file_path in upload_dir.iterdir():
            if file_path.is_file() and not file_path.name.endswith(".extracted.txt"):
                ext = file_path.suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    media_type = MEDIA_TYPE_MAP.get(ext, "document")
                    multimodal_info = cache.get("multimodal", file_path.name) or {}

                    file_infos.append({
                        "filename": file_path.name,
                        "size": file_path.stat().st_size,
                        "created": file_path.stat().st_ctime,
                        "media_type": media_type,
                        "extracted_length": multimodal_info.get("extracted_text_length", 0),
                        "ocr_confidence": multimodal_info.get("ocr_confidence", 0),
                        "duration": multimodal_info.get("duration", 0)
                    })

        if not file_infos:
            return {"success": True, "total": 0, "documents": [], "page": page, "page_size": page_size}

        file_infos.sort(key=lambda x: x.get("created", 0), reverse=True)

        doc_cache = get_document_cache()
        filenames = [f["filename"] for f in file_infos]
        cached_levels, missing_filenames = doc_cache.batch_get_levels(filenames)

        doc_level_map = cached_levels.copy()

        if missing_filenames:
            try:
                store = get_vector_store()
                if store and store.index_exists(index_name):
                    collection = Collection(index_name)
                    collection.load()

                    batch_size = 50
                    for i in range(0, len(missing_filenames), batch_size):
                        batch_names = missing_filenames[i:i + batch_size]
                        names_str = ', '.join([f'"{name}"' for name in batch_names])
                        expr = f"docnm in [{names_str}]"

                        results = collection.query(
                            expr=expr,
                            output_fields=["docnm", "user_level"],
                            limit=len(batch_names) * 10
                        )

                        found_in_batch = {}
                        for result in results:
                            docnm = result.get("docnm", "")
                            level = result.get("user_level", "normal")
                            if docnm and docnm not in found_in_batch:
                                found_in_batch[docnm] = level

                        for docnm, level in found_in_batch.items():
                            doc_level_map[docnm] = level
                            doc_cache.set_document_level(docnm, level)

                        for filename in batch_names:
                            if filename not in found_in_batch:
                                doc_level_map[filename] = "normal"
            except Exception as e:
                logger.error(f"从 Milvus 获取文档等级失败: {e}")
                for filename in missing_filenames:
                    doc_level_map[filename] = "normal"

        allowed_docs = []
        for file_info in file_infos:
            filename = file_info["filename"]
            doc_level = doc_level_map.get(filename, "normal")
            doc_priority = level_priority.get(doc_level, 1)

            if current_priority >= doc_priority:
                allowed_docs.append({
                    "filename": filename,
                    "size": file_info["size"],
                    "created": file_info["created"],
                    "user_level": doc_level,
                    "media_type": file_info.get("media_type", "document"),
                    "extracted_text_length": file_info.get("extracted_length", 0),
                    "ocr_confidence": file_info.get("ocr_confidence", 0),
                    "duration_seconds": file_info.get("duration", 0)
                })

        total = len(allowed_docs)
        start = (page - 1) * page_size
        end = start + page_size
        paged_docs = allowed_docs[start:end]

        return {
            "success": True,
            "total": total,
            "page": page,
            "page_size": page_size,
            "documents": paged_docs,
            "cache_info": {
                "hits": len(cached_levels),
                "misses": len(missing_filenames)
            }
        }

    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        return {"success": True, "total": 0, "documents": [], "page": page, "page_size": page_size,
                "warning": "加载失败"}


__all__ = ['router']