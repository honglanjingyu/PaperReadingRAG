# app/api/routes/upload.py
"""
文档上传路由
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib
import time
import shutil

from app.api.config import UPLOAD_DIR, SUPPORTED_EXTENSIONS, settings, processing_status
from app.api.dependencies import get_document_service

router = APIRouter()


@router.post("/upload")
async def upload_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        chunk_size: Optional[int] = None,
        enable_vectorization: bool = True,
        enable_storage: bool = True,
        from_page: int = 0,
        to_page: Optional[int] = None
) -> Dict[str, Any]:
    """
    上传并处理文档

    流程：
    1. 数据加载 -> 布局识别 -> 连接跨页内容 -> 数据清洗
    2. 智能分块 -> 向量化 -> 向量存储
    """
    # 验证文件类型
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}。支持的类型: {list(SUPPORTED_EXTENSIONS.keys())}"
        )

    # 保存文件
    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    # 生成处理ID
    process_id = hashlib.md5(f"{file.filename}_{time.time()}".encode()).hexdigest()[:16]

    # 使用默认配置
    if chunk_size is None:
        chunk_size = settings.chunk_size

    if to_page is None:
        to_page = settings.max_pages

    # 记录处理状态
    processing_status[process_id] = {
        "status": "processing",
        "filename": file.filename,
        "progress": 0,
        "message": "开始处理文档..."
    }

    # 启动后台任务
    document_service = get_document_service()
    background_tasks.add_task(
        document_service.process_document_task,
        process_id,
        str(file_path),
        chunk_size,
        enable_vectorization,
        enable_storage,
        from_page,
        to_page
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


@router.get("/upload/list")
async def list_documents() -> Dict[str, Any]:
    """列出已上传的文档"""
    documents = []
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            documents.append({
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "created": file_path.stat().st_ctime
            })

    return {
        "success": True,
        "total": len(documents),
        "documents": documents
    }


@router.delete("/upload/{filename}")
async def delete_document(filename: str) -> Dict[str, Any]:
    """删除文档"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")

    try:
        file_path.unlink()
        return {
            "success": True,
            "message": f"文档已删除: {filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")