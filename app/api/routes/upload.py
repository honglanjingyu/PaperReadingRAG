# app/api/routes/upload.py
"""
文档上传路由
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Header
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


# app/api/routes/upload.py - 修改 delete_document 函数

@router.delete("/upload/{filename}")
async def delete_document(
        filename: str,
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """删除文档（同时使缓存失效）"""
    from app.service.core.cache import get_document_cache

    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")

    try:
        # 删除文件
        file_path.unlink()

        # 使文档等级缓存失效
        doc_cache = get_document_cache()
        doc_cache.delete_document_level(filename)
        logger.info(f"文档 {filename} 缓存已清除")

        return {
            "success": True,
            "message": f"文档已删除: {filename}"
        }
    except Exception as e:
        logger.error(f"删除失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")