# app/api/routes/upload_batch.py

from fastapi import APIRouter, HTTPException, Header
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging
import os
from pathlib import Path

from app.api.config import UPLOAD_DIR, SUPPORTED_EXTENSIONS, settings
from app.auth.jwt_utils import get_user_id_from_token
from app.db.database import get_db_manager
from app.service.core.rag.async_processor import get_async_processor

router = APIRouter()
logger = logging.getLogger(__name__)


class BatchUploadRequest(BaseModel):
    """批量上传请求"""
    files: List[str] = Field(..., description="文件名列表")
    chunk_size: Optional[int] = Field(None, description="分块大小")
    from_page: int = Field(0, description="起始页")
    to_page: Optional[int] = Field(None, description="结束页")
    enable_vectorization: bool = Field(True, description="启用向量化")
    enable_storage: bool = Field(True, description="启用存储")


class BatchUploadResponse(BaseModel):
    """批量上传响应"""
    success: bool
    batch_id: str
    total_files: int
    message: str
    status_url: str
    tasks: Dict[str, str]  # task_id -> file_name

@router.post("/upload/batch")
async def batch_upload(
        request: BatchUploadRequest,
        authorization: Optional[str] = Header(None)
):
    """
    批量上传文档 - 使用上传者的等级
    """
    # ========== 获取上传者等级（关键） ==========
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
                user_level = user.role.value  # 只使用用户等级，不看文件名
                username = user.username
                logger.info(f"用户 {username} (等级={user_level}) 发起批量上传，共 {len(request.files)} 个文件")

    # ========== 验证文件存在（文件必须已存在） ==========
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
        return {
            "success": False,
            "message": "没有有效的文件可上传"
        }

    # ========== 提交任务，使用上传者等级（而不是文件名判断） ==========
    processor = get_async_processor()
    tasks = {}

    for file_name in valid_files:
        file_path = UPLOAD_DIR / file_name

        try:
            task_id = await processor.submit_task(
                file_path=str(file_path),
                file_name=file_name,
                user_level=user_level,  # ← 关键：使用上传者等级，不看文件名
                chunk_size=request.chunk_size or settings.chunk_size,
                from_page=request.from_page,
                to_page=request.to_page or settings.max_pages,
                enable_vectorization=request.enable_vectorization,
                enable_storage=request.enable_storage
            )
            tasks[task_id] = file_name
            logger.info(f"任务已提交: {task_id} -> {file_name} (文档等级={user_level})")

        except Exception as e:
            logger.error(f"提交任务失败 {file_name}: {e}")

    return {
        "success": True,
        "total_files": len(valid_files),
        "tasks": tasks,
        "user_level": user_level,  # 返回使用的等级
        "message": f"已提交 {len(tasks)} 个文件，文档等级={user_level}"
    }

@router.get("/upload/batch/status/{batch_id}")
async def get_batch_status(
        batch_id: str,
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    获取批量上传状态
    """
    # 这里需要维护批次与任务的关系
    # 简化实现：返回一个状态
    return {
        "success": True,
        "batch_id": batch_id,
        "status": "processing",
        "completed": 0,
        "total": 0,
        "message": "批量处理中..."
    }


@router.get("/upload/task/{task_id}")
async def get_task_status(
        task_id: str,
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    获取单个任务状态
    """
    processor = get_async_processor()
    status = await processor.get_task_status(task_id)

    if not status:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    return status