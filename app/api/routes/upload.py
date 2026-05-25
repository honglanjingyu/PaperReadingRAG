# app/api/routes/upload.py (修改版 - 保留原始上传文件)

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Header, Query
from pathlib import Path
from typing import List, Optional, Dict, Any
import shutil
import os
import logging

# ========== 导入多模态解析器 ==========
from app.api.config import UPLOAD_DIR, SUPPORTED_EXTENSIONS, MEDIA_TYPE_MAP, settings
from app.api.dependencies import get_document_service
from app.auth.jwt_utils import get_user_id_from_token
from app.db.database import get_db_manager
from app.service.core.rag.async_processor import get_async_processor
from app.service.core.multimodal import get_multimodal_parser, ExtractedContent
from pydantic import BaseModel, Field
from app.service.core.graphrag import get_graph_rag_service

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
    """
    从文件中提取文字内容（支持多模态）
    """
    parser = get_multimodal_parser()
    return parser.parse(file_path, file_name)


# ========== 单个文档上传（支持多模态） ==========

@router.post("/upload")
async def upload_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        chunk_size: Optional[int] = None,
        enable_vectorization: bool = True,
        enable_storage: bool = True,
        from_page: int = 0,
        to_page: Optional[int] = None,
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    上传并处理文档（支持：图片、音频、视频、PDF、DOCX、TXT等）
    """
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

    # ========== 多模态预处理：提取文字内容 ==========
    extracted = _extract_text_from_file(str(file_path), file.filename)

    if not extracted.success:
        # 解析失败，删除文件
        file_path.unlink()
        raise HTTPException(status_code=400, detail=f"文件解析失败: {extracted.error}")

    if not extracted.text_content or len(extracted.text_content.strip()) < 10:
        logger.warning(f"文件 {file.filename} 未提取到有效文字内容")
        # 不阻塞，允许上传空内容，但记录警告

    # 提交到异步处理队列
    processor = get_async_processor()

    # ========== 根据媒体类型选择不同的提交方式 ==========
    if media_type in ["image", "audio", "video"]:
        # 多模态文件：直接传递提取的文字，不创建临时文件
        task_id = await processor.submit_task_with_text(
            extracted_text=extracted.text_content,
            file_name=file.filename,
            original_filename=file.filename,
            media_type=media_type,
            extracted_length=len(extracted.text_content),
            user_level=user_level,
            chunk_size=chunk_size or settings.chunk_size,
            enable_vectorization=enable_vectorization,
            enable_storage=enable_storage
        )
        # ⚠️ 注意：不删除原始上传文件，保留原文件

    else:
        # 文档类型：使用传统文件路径方式
        task_id = await processor.submit_task(
            file_path=str(file_path),
            file_name=file.filename,
            original_filename=file.filename,
            media_type=media_type,
            extracted_length=len(extracted.text_content),
            user_level=user_level,
            chunk_size=chunk_size or settings.chunk_size,
            from_page=from_page,
            to_page=to_page or settings.max_pages,
            enable_vectorization=enable_vectorization,
            enable_storage=enable_storage
        )

    # 记录多模态元数据到缓存
    from app.service.core.cache import get_cache_manager
    cache = get_cache_manager()
    cache.set("multimodal", file.filename, {
        "media_type": media_type,
        "ocr_confidence": extracted.ocr_confidence if media_type == "image" else 0,
        "transcript_confidence": extracted.transcript_confidence if media_type in ["audio", "video"] else 0,
        "duration": extracted.duration_seconds if media_type in ["audio", "video"] else 0,
        "original_filename": file.filename,
        "extracted_text_length": len(extracted.text_content),
        "task_id": task_id
    }, ttl=86400)  # 24小时缓存

    _invalidate_graph_cache(user_level)

    return {
        "success": True,
        "task_id": task_id,
        "filename": file.filename,
        "media_type": media_type,
        "extracted_text_length": len(extracted.text_content),
        "message": f"文件已提交到处理队列{'（已通过OCR/ASR提取文字）' if media_type in ['image', 'audio', 'video'] else ''}",
        "status_url": f"/api/upload/task/{task_id}"
    }


@router.post("/upload/async")
async def upload_document_async(
        file: UploadFile = File(...),
        chunk_size: Optional[int] = None,
        enable_vectorization: bool = True,
        enable_storage: bool = True,
        from_page: int = 0,
        to_page: Optional[int] = None,
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """异步上传文档（使用队列处理器）- 支持多模态"""
    user_level, user_id, username = _get_user_level_from_token(authorization)

    file_ext = _validate_file(file)
    media_type = MEDIA_TYPE_MAP.get(file_ext, "document")

    file_path = UPLOAD_DIR / file.filename
    if file_path.exists():
        raise HTTPException(status_code=400, detail=f"文件已存在: {file.filename}")

    # 保存原始文件
    _save_upload_file(file, file_path)

    # 多模态预处理
    extracted = _extract_text_from_file(str(file_path), file.filename)

    if not extracted.success:
        file_path.unlink()
        raise HTTPException(status_code=400, detail=f"文件解析失败: {extracted.error}")

    processor = get_async_processor()

    # ========== 根据媒体类型选择不同的提交方式 ==========
    if media_type in ["image", "audio", "video"]:
        # 多模态文件：直接传递提取的文字
        task_id = await processor.submit_task_with_text(
            extracted_text=extracted.text_content,
            file_name=file.filename,
            original_filename=file.filename,
            media_type=media_type,
            extracted_length=len(extracted.text_content),
            user_level=user_level,
            chunk_size=chunk_size or settings.chunk_size,
            enable_vectorization=enable_vectorization,
            enable_storage=enable_storage
        )
        # ⚠️ 注意：不删除原始上传文件，保留原文件

    else:
        # 文档类型：使用传统方式
        task_id = await processor.submit_task(
            file_path=str(file_path),
            file_name=file.filename,
            original_filename=file.filename,
            media_type=media_type,
            extracted_length=len(extracted.text_content),
            user_level=user_level,
            chunk_size=chunk_size or settings.chunk_size,
            from_page=from_page,
            to_page=to_page or settings.max_pages,
            enable_vectorization=enable_vectorization,
            enable_storage=enable_storage
        )

    _invalidate_graph_cache(user_level)

    return {
        "success": True,
        "task_id": task_id,
        "filename": file.filename,
        "media_type": media_type,
        "extracted_text_length": len(extracted.text_content),
        "message": "文档已提交到处理队列",
        "status_url": f"/api/upload/task/{task_id}"
    }


# ========== 异步任务状态查询 ==========

def _get_status_message(status: str, media_type: str = None) -> str:
    """根据状态和媒体类型返回友好的状态消息"""
    if status == "pending":
        if media_type == "image":
            return "等待 OCR 识别..."
        elif media_type == "audio":
            return "等待语音转文字..."
        elif media_type == "video":
            return "等待视频处理..."
        return "等待处理..."
    elif status == "processing":
        if media_type == "image":
            return "正在通过 OCR 识别图片文字..."
        elif media_type == "audio":
            return "正在通过 ASR 转写音频..."
        elif media_type == "video":
            return "正在处理视频（提取音频+关键帧OCR）..."
        return "正在处理文档..."
    elif status == "completed":
        if media_type == "image":
            return "图片 OCR 识别完成"
        elif media_type == "audio":
            return "音频转文字完成"
        elif media_type == "video":
            return "视频处理完成"
        return "文档处理完成"
    elif status == "failed":
        return "处理失败"
    return "未知状态"


@router.get("/upload/task/{task_id}")
async def get_task_status(
        task_id: str,
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    获取异步任务状态（支持多模态）

    Args:
        task_id: 任务ID

    Returns:
        任务状态信息，包含：
        - status: pending/processing/completed/failed
        - progress: 进度百分比 (0-100)
        - message: 状态描述
        - media_type: 媒体类型（如果是多模态文件）
        - extracted_length: 提取的文字长度
    """
    from app.service.core.rag.async_processor import get_async_processor

    processor = get_async_processor()
    task_status = await processor.get_task_status(task_id)

    if task_status is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    # 增强多模态信息的返回
    media_type = task_status.get("media_type", "document")
    status = task_status.get("status", "pending")

    result = {
        "task_id": task_status.get("task_id"),
        "status": status,
        "file_name": task_status.get("file_name"),
        "error": task_status.get("error"),
        "progress": task_status.get("progress", 0),
        "message": task_status.get("message") or _get_status_message(status, media_type),
        "media_type": media_type,
        "extracted_length": task_status.get("extracted_length", 0)
    }

    return result


# ========== 批量上传 ==========

class BatchUploadRequest(BaseModel):
    """批量上传请求"""
    files: List[str] = Field(..., description="文件名列表")
    chunk_size: Optional[int] = Field(None, description="分块大小")
    from_page: int = Field(0, description="起始页")
    to_page: Optional[int] = Field(None, description="结束页")
    enable_vectorization: bool = Field(True, description="启用向量化")
    enable_storage: bool = Field(True, description="启用存储")


@router.post("/upload/batch")
async def batch_upload(
        request: BatchUploadRequest,
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    批量上传文档 - 使用上传者的等级（支持多模态）
    """
    user_level, user_id, username = _get_user_level_from_token(authorization)

    logger.info(f"用户 {username or 'unknown'} (等级={user_level}) 发起批量上传，共 {len(request.files)} 个文件")

    # 验证文件存在
    valid_files = []
    for file_name in request.files:
        file_path = UPLOAD_DIR / file_name
        if not file_path.exists():
            logger.warning(f"文件不存在: {file_name}")
            continue

        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.warning(f"不支持的文件类型: {file_name}")
            continue

        valid_files.append(file_name)

    if not valid_files:
        return {"success": False, "message": "没有有效的文件可上传"}

    # 提交任务
    processor = get_async_processor()
    parser = get_multimodal_parser()
    tasks = {}

    for file_name in valid_files:
        file_path = UPLOAD_DIR / file_name
        ext = file_path.suffix.lower()
        media_type = MEDIA_TYPE_MAP.get(ext, "document")

        try:
            # 多模态预处理
            extracted = parser.parse(str(file_path), file_name)

            if not extracted.success:
                logger.warning(f"文件解析失败 {file_name}: {extracted.error}")
                continue

            # ========== 根据媒体类型选择不同的提交方式 ==========
            if media_type in ["image", "audio", "video"]:
                # 多模态文件：直接传递提取的文字
                task_id = await processor.submit_task_with_text(
                    extracted_text=extracted.text_content,
                    file_name=file_name,
                    original_filename=file_name,
                    media_type=media_type,
                    extracted_length=len(extracted.text_content),
                    user_level=user_level,
                    chunk_size=request.chunk_size or settings.chunk_size,
                    enable_vectorization=request.enable_vectorization,
                    enable_storage=request.enable_storage
                )
                # ⚠️ 注意：不删除原始上传文件，保留原文件

            else:
                # 文档类型：使用传统方式
                task_id = await processor.submit_task(
                    file_path=str(file_path),
                    file_name=file_name,
                    original_filename=file_name,
                    media_type=media_type,
                    extracted_length=len(extracted.text_content),
                    user_level=user_level,
                    chunk_size=request.chunk_size or settings.chunk_size,
                    from_page=request.from_page,
                    to_page=request.to_page or settings.max_pages,
                    enable_vectorization=request.enable_vectorization,
                    enable_storage=request.enable_storage
                )

            tasks[task_id] = {
                "filename": file_name,
                "media_type": media_type,
                "extracted_length": len(extracted.text_content)
            }
            logger.info(f"任务已提交: {task_id} -> {file_name} (类型={media_type})")

        except Exception as e:
            logger.error(f"提交任务失败 {file_name}: {e}")

    _invalidate_graph_cache(user_level)

    return {
        "success": True,
        "total_files": len(valid_files),
        "tasks": tasks,
        "user_level": user_level,
        "message": f"已提交 {len(tasks)} 个文件，文档等级={user_level}"
    }


# ========== 文档列表 ==========

@router.get("/upload/list")
async def list_documents(
        authorization: Optional[str] = Header(None),
        page: int = Query(1, ge=1, description="页码"),
        page_size: int = Query(20, ge=1, le=100, description="每页数量")
) -> Dict[str, Any]:
    """列出已上传的文档（分页）- 显示媒体类型"""
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

    # 获取多模态缓存
    from app.service.core.cache import get_cache_manager
    cache = get_cache_manager()

    try:
        if not upload_dir.exists():
            return {"success": True, "total": 0, "documents": [], "page": page, "page_size": page_size}

        # 收集文件信息（排除临时文件）
        file_infos = []
        for file_path in upload_dir.iterdir():
            if file_path.is_file() and not file_path.name.endswith(".extracted.txt"):
                ext = file_path.suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    media_type = MEDIA_TYPE_MAP.get(ext, "document")

                    # 获取多模态元数据
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

        # 获取文档等级（原有逻辑）
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

        # 根据用户权限过滤文档
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

        # 分页
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


# ========== 文档统计信息 ==========

@router.get("/upload/stats/{filename}")
async def get_document_stats(filename: str) -> Dict[str, Any]:
    """获取文档统计信息（包含媒体类型）"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")

    ext = file_path.suffix.lower()
    media_type = MEDIA_TYPE_MAP.get(ext, "document")

    # 获取多模态元数据
    from app.service.core.cache import get_cache_manager
    cache = get_cache_manager()
    multimodal_info = cache.get("multimodal", filename) or {}

    try:
        document_service = get_document_service()
        stats = document_service.get_processing_stats(str(file_path))

        # 添加多模态信息
        stats["media_type"] = media_type
        stats["extracted_text_length"] = multimodal_info.get("extracted_text_length", 0)

        if media_type == "image":
            stats["ocr_confidence"] = multimodal_info.get("ocr_confidence", 0)
        elif media_type in ["audio", "video"]:
            stats["transcript_confidence"] = multimodal_info.get("transcript_confidence", 0)
            stats["duration_seconds"] = multimodal_info.get("duration", 0)

        return {"success": True, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


__all__ = ['router']