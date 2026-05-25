# app/service/core/rag/async_processor.py

"""异步文档处理器 - Redis 队列（增强多模态支持）"""

import os
import json
import hashlib
import asyncio
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

import redis

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DocumentTask:
    task_id: str
    file_path: str
    file_name: str
    original_filename: str = ""  # 原始文件名
    media_type: str = "document"  # 媒体类型: document, image, audio, video
    extracted_length: int = 0  # 提取的文字长度
    extracted_text: str = ""  # 已提取的文字内容（用于多模态文件）
    user_level: str = "normal"
    chunk_size: int = 256
    from_page: int = 0
    to_page: int = 100000
    enable_vectorization: bool = True
    enable_storage: bool = True
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = None
    started_at: float = None
    completed_at: float = None
    error: Optional[str] = None
    result: Optional[Any] = None
    progress: int = 0  # 进度百分比
    message: str = ""  # 状态消息

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if not self.original_filename:
            self.original_filename = self.file_name

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "original_filename": self.original_filename,
            "media_type": self.media_type,
            "extracted_length": self.extracted_length,
            "extracted_text": self.extracted_text[:500] if self.extracted_text else "",  # 只保存前500字符
            "user_level": self.user_level,
            "chunk_size": self.chunk_size,
            "from_page": self.from_page,
            "to_page": self.to_page,
            "enable_vectorization": self.enable_vectorization,
            "enable_storage": self.enable_storage,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "result": self.result,
            "progress": self.progress,
            "message": self.message
        }


class AsyncDocumentProcessor:
    """异步文档处理器"""

    def __init__(self):
        self.max_workers = int(os.getenv("ASYNC_PROCESSOR_MAX_WORKERS", "3"))
        self.task_timeout = int(os.getenv("ASYNC_PROCESSOR_TASK_TIMEOUT", "300"))
        self._task_prefix = "rag:async_task:"

        # 连接 Redis
        try:
            self._redis = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                password=os.getenv("REDIS_PASSWORD") or None,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            self._redis.ping()
            logger.info("AsyncDocumentProcessor Redis 连接成功")
        except Exception as e:
            logger.error(f"AsyncDocumentProcessor Redis 连接失败: {e}")
            raise

        self._task_queue = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._running = False
        self._workers = []

    def _task_key(self, task_id: str) -> str:
        return f"{self._task_prefix}{task_id}"

    def _save_task(self, task: DocumentTask):
        """保存任务到 Redis"""
        try:
            self._redis.setex(self._task_key(task.task_id), 3600, json.dumps(task.to_dict()))
            logger.debug(f"任务已保存: {task.task_id}, status={task.status.value}, progress={task.progress}")
        except Exception as e:
            logger.error(f"保存任务失败 {task.task_id}: {e}")

    def _load_task(self, task_id: str) -> Optional[DocumentTask]:
        """从 Redis 加载任务"""
        try:
            data = self._redis.get(self._task_key(task_id))
            if data:
                d = json.loads(data)
                return DocumentTask(
                    task_id=d.get('task_id'),
                    file_path=d.get('file_path', ''),
                    file_name=d.get('file_name', ''),
                    original_filename=d.get('original_filename', d.get('file_name', '')),
                    media_type=d.get('media_type', 'document'),
                    extracted_length=d.get('extracted_length', 0),
                    extracted_text=d.get('extracted_text', ''),
                    user_level=d.get('user_level', 'normal'),
                    chunk_size=d.get('chunk_size', 256),
                    from_page=d.get('from_page', 0),
                    to_page=d.get('to_page', 100000),
                    enable_vectorization=d.get('enable_vectorization', True),
                    enable_storage=d.get('enable_storage', True),
                    status=TaskStatus(d.get('status', 'pending')),
                    created_at=d.get('created_at'),
                    started_at=d.get('started_at'),
                    completed_at=d.get('completed_at'),
                    error=d.get('error'),
                    result=d.get('result'),
                    progress=d.get('progress', 0),
                    message=d.get('message', '')
                )
        except Exception as e:
            logger.error(f"加载任务失败 {task_id}: {e}")
        return None

    async def start(self):
        """启动处理器"""
        if self._running:
            return
        self._running = True
        for i in range(self.max_workers):
            self._workers.append(asyncio.create_task(self._worker(i)))

        # 恢复未完成任务
        try:
            for key in self._redis.keys(f"{self._task_prefix}*"):
                task_id = key.split(':')[-1]
                task = self._load_task(task_id)
                if task and task.status in (TaskStatus.PENDING, TaskStatus.PROCESSING):
                    await self._task_queue.put(task)
                    logger.info(f"恢复未完成任务: {task_id}")
        except Exception as e:
            logger.warning(f"恢复任务失败: {e}")

        logger.info(f"异步处理器启动, workers={self.max_workers}")

    async def stop(self):
        """停止处理器"""
        self._running = False
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._executor.shutdown()
        logger.info("异步处理器已停止")

    async def submit_task(
        self,
        file_path: str,
        file_name: str,
        original_filename: str = None,
        media_type: str = "document",
        extracted_length: int = 0,
        user_level: str = "normal",
        chunk_size: int = 256,
        from_page: int = 0,
        to_page: int = 100000,
        enable_vectorization: bool = True,
        enable_storage: bool = True
    ) -> str:
        """提交异步处理任务（文档类型）"""
        task_id = hashlib.md5(f"{file_path}_{time.time()}".encode()).hexdigest()[:16]
        task = DocumentTask(
            task_id=task_id,
            file_path=file_path,
            file_name=file_name,
            original_filename=original_filename or file_name,
            media_type=media_type,
            extracted_length=extracted_length,
            extracted_text="",  # 文档类型没有预提取的文字
            user_level=user_level,
            chunk_size=chunk_size,
            from_page=from_page,
            to_page=to_page,
            enable_vectorization=enable_vectorization,
            enable_storage=enable_storage,
            message=self._get_init_message(media_type)
        )
        self._save_task(task)
        await self._task_queue.put(task)
        logger.info(f"任务提交: {task_id} - {file_name} (类型={media_type})")
        return task_id

    async def submit_task_with_text(
        self,
        extracted_text: str,
        file_name: str,
        original_filename: str = None,
        media_type: str = "image",
        extracted_length: int = 0,
        user_level: str = "normal",
        chunk_size: int = 256,
        enable_vectorization: bool = True,
        enable_storage: bool = True
    ) -> str:
        """
        提交异步处理任务（多模态类型，已提取文字）
        """
        task_id = hashlib.md5(f"{file_name}_{time.time()}".encode()).hexdigest()[:16]
        task = DocumentTask(
            task_id=task_id,
            file_path="",
            file_name=file_name,
            original_filename=original_filename or file_name,
            media_type=media_type,
            extracted_length=extracted_length or len(extracted_text),
            extracted_text=extracted_text,
            user_level=user_level,
            chunk_size=chunk_size,
            from_page=0,
            to_page=100000,
            enable_vectorization=enable_vectorization,
            enable_storage=enable_storage,
            message=self._get_init_message(media_type)
        )
        self._save_task(task)
        await self._task_queue.put(task)
        logger.info(f"任务提交: {task_id} - {file_name} (类型={media_type}, 文字长度={len(extracted_text)})")
        return task_id

    def _get_init_message(self, media_type: str) -> str:
        """获取初始状态消息"""
        if media_type == "image":
            return "等待 OCR 识别..."
        elif media_type == "audio":
            return "等待语音转文字..."
        elif media_type == "video":
            return "等待视频处理..."
        else:
            return "等待处理..."

    async def get_task_status(self, task_id: str) -> Optional[Dict]:
        """获取任务状态"""
        task = self._load_task(task_id)
        if not task:
            return None

        return {
            "task_id": task.task_id,
            "file_name": task.original_filename or task.file_name,
            "status": task.status.value,
            "error": task.error,
            "progress": task.progress,
            "message": task.message or self._get_status_message(task),
            "media_type": task.media_type,
            "extracted_length": task.extracted_length,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at
        }

    def _get_status_message(self, task: DocumentTask) -> str:
        """根据任务状态和媒体类型返回友好的状态消息"""
        if task.status == TaskStatus.PENDING:
            if task.media_type == "image":
                return "等待 OCR 识别..."
            elif task.media_type == "audio":
                return "等待语音转文字..."
            elif task.media_type == "video":
                return "等待视频处理..."
            return "等待处理..."
        elif task.status == TaskStatus.PROCESSING:
            if task.media_type == "image":
                return f"正在 OCR 识别图片文字... ({task.progress}%)"
            elif task.media_type == "audio":
                return f"正在通过 ASR 转写音频... ({task.progress}%)"
            elif task.media_type == "video":
                return f"正在处理视频（提取音频+关键帧OCR）... ({task.progress}%)"
            return f"正在处理文档... ({task.progress}%)"
        elif task.status == TaskStatus.COMPLETED:
            if task.media_type == "image":
                return "图片 OCR 识别完成"
            elif task.media_type == "audio":
                return "音频转文字完成"
            elif task.media_type == "video":
                return "视频处理完成"
            return "文档处理完成"
        elif task.status == TaskStatus.FAILED:
            return f"处理失败: {task.error}" if task.error else "处理失败"
        return "未知状态"

    def _update_task_progress(self, task: DocumentTask, progress: int, message: str = None):
        """更新任务进度"""
        task.progress = min(100, max(0, progress))
        if message:
            task.message = message
        self._save_task(task)

    def _process_document_with_text(self, task: DocumentTask) -> List:
        """
        处理已提取文字的多模态文件（同步函数，在线程池中执行）
        """
        from app.service.core.rag import process_document_with_text

        logger.info(f"开始处理多模态任务: task_id={task.task_id}, 文件={task.file_name}, 文字长度={len(task.extracted_text)}")

        try:
            result = process_document_with_text(
                text_content=task.extracted_text,
                file_name=task.original_filename or task.file_name,
                chunk_size=task.chunk_size,
                enable_vectorization=task.enable_vectorization,
                enable_storage=task.enable_storage,
                user_level=task.user_level,
                verbose=False
            )

            logger.info(f"process_document_with_text 返回: type={type(result)}, len={len(result) if result else 0}")

            if result is None:
                logger.error(f"process_document_with_text 返回 None")
                return []

            return result

        except Exception as e:
            logger.error(f"process_document_with_text 执行失败: {e}", exc_info=True)
            raise

    def _process_document_file(self, task: DocumentTask) -> List:
        """
        处理文档文件（同步函数，在线程池中执行）
        """
        from app.service.core.rag import process_document

        logger.info(f"开始处理文档任务: task_id={task.task_id}, 文件={task.file_name}, 路径={task.file_path}")

        result = process_document(
            file_path=task.file_path,
            chunk_size=task.chunk_size,
            enable_vectorization=task.enable_vectorization,
            enable_storage=task.enable_storage,
            from_page=task.from_page,
            to_page=task.to_page,
            user_level=task.user_level,
            verbose=False
        )

        logger.info(f"process_document_file 返回: type={type(result)}, len={len(result) if result else 0}")
        return result if result else []

    async def _worker(self, worker_id: int):
        """工作协程"""
        while self._running:
            try:
                task = await asyncio.wait_for(self._task_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            logger.info(
                f"Worker {worker_id} 开始处理任务: {task.task_id} (类型={task.media_type}, 文件={task.file_name})")

            # 更新状态为 PROCESSING
            task.status = TaskStatus.PROCESSING
            task.started_at = time.time()
            task.progress = 10
            task.message = self._get_status_message(task)
            self._save_task(task)

            result = None
            try:
                # 根据媒体类型和是否有预提取文字选择处理方式
                if task.media_type in ["image", "audio", "video"] and task.extracted_text:
                    # 多模态文件，已有预提取的文字
                    task.progress = 30
                    task.message = self._get_status_message(task)
                    self._save_task(task)
                    logger.info(f"多模态任务 {task.task_id}: 使用预提取文字，长度={len(task.extracted_text)}")

                    # 执行处理（使用预提取的文字）
                    result = await asyncio.to_thread(
                        self._process_document_with_text,
                        task
                    )
                    logger.info(
                        f"多模态任务 {task.task_id}: _process_document_with_text 返回 result, type={type(result)}, len={len(result) if result else 0}")

                elif task.file_path and os.path.exists(task.file_path):
                    # 普通文档文件
                    task.progress = 20
                    task.message = self._get_status_message(task)
                    self._save_task(task)
                    logger.info(f"文档任务 {task.task_id}: 文件路径={task.file_path}")

                    # 执行文档处理
                    result = await asyncio.to_thread(
                        self._process_document_file,
                        task
                    )
                    logger.info(
                        f"文档任务 {task.task_id}: _process_document_file 返回 result, type={type(result)}, len={len(result) if result else 0}")

                else:
                    raise Exception(f"无效的任务: 没有文件路径也没有预提取文字 (media_type={task.media_type})")

                # ========== 关键修复：检查处理结果并正确更新状态 ==========
                logger.info(f"=== 任务 {task.task_id} 结果检查 ===")
                logger.info(f"result 类型: {type(result)}")
                logger.info(f"result 是否为 None: {result is None}")
                logger.info(f"result 长度: {len(result) if result else 0}")

                # 判断结果是否有效
                is_valid = result is not None and len(result) > 0
                logger.info(f"is_valid: {is_valid}")

                if is_valid:
                    # 处理成功
                    logger.info(f"准备更新任务状态为 COMPLETED: {task.task_id}")

                    task.status = TaskStatus.COMPLETED
                    task.completed_at = time.time()
                    task.progress = 100
                    task.message = self._get_status_message(task)

                    # ========== 关键修复：不保存 result 到 Redis（避免序列化错误） ==========
                    # 保存前将 result 临时设为 None
                    temp_result = task.result
                    task.result = None
                    self._save_task(task)
                    # 恢复 result（如果需要后续使用）
                    task.result = temp_result

                    logger.info(
                        f"✅ 任务完成: {task.task_id} (类型={task.media_type}, 文件={task.file_name}, 块数={len(result)})")

                    # 验证保存
                    saved = self._load_task(task.task_id)
                    if saved:
                        logger.info(f"验证: 任务已保存, status={saved.status.value}, progress={saved.progress}")
                    else:
                        logger.error(f"❌ 任务保存后无法加载: {task.task_id}")
                else:
                    # 处理结果为空，标记为失败
                    error_msg = f"处理结果为空 (result={result}, len={len(result) if result else 0})"
                    logger.error(f"❌ 任务 {task.task_id} 返回空结果: {error_msg}")

                    task.status = TaskStatus.FAILED
                    task.completed_at = time.time()
                    task.error = error_msg
                    task.message = f"处理失败: {error_msg}"
                    task.result = None  # 确保 result 为 None
                    self._save_task(task)

                # 清理临时文件（如果是多模态文件且有临时文件）
                if task.media_type in ["image", "audio", "video"] and task.file_path and task.file_path.endswith(
                        '.extracted.txt'):
                    try:
                        os.unlink(task.file_path)
                        logger.info(f"已清理临时文件: {task.file_path}")
                    except Exception as e:
                        logger.warning(f"清理临时文件失败: {e}")

            except Exception as e:
                logger.error(f"❌ 任务失败: {task.task_id} - {e}", exc_info=True)
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                task.error = str(e)
                task.message = f"处理失败: {str(e)}"
                task.result = None
                self._save_task(task)

            finally:
                self._task_queue.task_done()

# 全局实例
_processor = None


def get_async_processor() -> AsyncDocumentProcessor:
    """获取异步处理器实例"""
    global _processor
    if _processor is None:
        _processor = AsyncDocumentProcessor()
    return _processor


async def init_async_processor():
    """初始化异步处理器"""
    await get_async_processor().start()


async def shutdown_async_processor():
    """关闭异步处理器"""
    await get_async_processor().stop()


__all__ = [
    'AsyncDocumentProcessor',
    'get_async_processor',
    'init_async_processor',
    'shutdown_async_processor'
]