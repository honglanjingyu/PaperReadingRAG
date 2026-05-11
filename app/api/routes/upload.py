# app/api/routes/upload.py
"""
文档上传路由 - 只包含单个上传功能
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Header, Query
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib
import time
import shutil
import os
import logging

from app.api.config import UPLOAD_DIR, SUPPORTED_EXTENSIONS, settings, processing_status
from app.api.dependencies import get_document_service
from app.auth.jwt_utils import get_user_id_from_token
from app.db.database import get_db_manager
from app.service.core.rag.async_processor import get_async_processor
from app.service.core.cache import get_document_cache

router = APIRouter()
logger = logging.getLogger(__name__)


# ========== 文档上传接口 ==========

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
    """上传并处理文档"""
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

    # ========== 修复：检查文件是否已存在 ==========
    file_path = UPLOAD_DIR / file.filename
    if file_path.exists():
        raise HTTPException(status_code=400, detail=f"文件已存在: {file.filename}")

    # 保存文件
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
        "user_level": user_level
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
        user_level
    )

    from app.service.core.graphrag import get_graph_rag_service
    graph_service = get_graph_rag_service()
    graph_service.invalidate_cache(user_level)

    return {
        "success": True,
        "process_id": process_id,
        "filename": file.filename,
        "message": "文档已上传，正在后台处理",
        "status_url": f"/api/upload/status/{process_id}"
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
    """异步上传文档（使用队列处理器）"""
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
            detail=f"不支持的文件类型: {file_ext}"
        )

    # ========== 修复：检查文件是否已存在 ==========
    file_path = UPLOAD_DIR / file.filename
    if file_path.exists():
        raise HTTPException(status_code=400, detail=f"文件已存在: {file.filename}")

    # 保存文件
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    # 提交到异步队列
    processor = get_async_processor()
    task_id = await processor.submit_task(
        file_path=str(file_path),
        file_name=file.filename,
        user_level=user_level,
        chunk_size=chunk_size or settings.chunk_size,
        from_page=from_page,
        to_page=to_page or settings.max_pages,
        enable_vectorization=enable_vectorization,
        enable_storage=enable_storage
    )

    from app.service.core.graphrag import get_graph_rag_service
    graph_service = get_graph_rag_service()
    graph_service.invalidate_cache(user_level)

    return {
        "success": True,
        "task_id": task_id,
        "filename": file.filename,
        "message": "文档已提交到处理队列",
        "status_url": f"/api/upload/task/{task_id}"
    }


# ========== 状态查询接口 ==========

@router.get("/upload/status/{process_id}")
async def get_processing_status(process_id: str) -> Dict[str, Any]:
    """获取文档处理状态"""
    status = processing_status.get(process_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"未找到处理任务: {process_id}")

    return status


@router.get("/upload/task/{task_id}")
async def get_task_status(
        task_id: str,
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """获取异步任务状态"""
    processor = get_async_processor()
    status = await processor.get_task_status(task_id)

    if not status:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

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


# ========== 文档列表接口（带分页） ==========

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

    documents = []
    index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")
    upload_dir = UPLOAD_DIR

    try:
        if not upload_dir.exists():
            logger.warning(f"上传目录不存在: {upload_dir}")
            return {"success": True, "total": 0, "documents": [], "page": page, "page_size": page_size}

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
            return {"success": True, "total": 0, "documents": [], "page": page, "page_size": page_size}

        # 按创建时间排序
        file_infos.sort(key=lambda x: x.get("created", 0), reverse=True)

        # 使用缓存获取文档等级
        doc_cache = get_document_cache()
        filenames = [f["filename"] for f in file_infos]
        cached_levels, missing_filenames = doc_cache.batch_get_levels(filenames)

        # 查询缺失的文档等级
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
                                doc_cache.set_document_level(filename, "normal")
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
                    "user_level": doc_level
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
                "misses": len(missing_filenames),
                "ttl": doc_cache.cache_ttl
            }
        }

    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        return {
            "success": True,
            "total": 0,
            "documents": [],
            "page": page,
            "page_size": page_size,
            "warning": "加载失败"
        }


__all__ = ['router']