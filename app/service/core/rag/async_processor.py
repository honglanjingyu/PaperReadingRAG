"""异步文档处理器 - Redis 队列"""

import os
import json
import hashlib
import asyncio
import time
import logging
from typing import Dict, Any, Optional
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
    user_level: str
    chunk_size: int
    from_page: int
    to_page: int
    enable_vectorization: bool
    enable_storage: bool
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = None
    started_at: float = None      # 添加
    completed_at: float = None    # 添加
    error: Optional[str] = None
    result: Optional[Any] = None  # 添加

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id, "file_path": self.file_path, "file_name": self.file_name,
            "user_level": self.user_level, "chunk_size": self.chunk_size,
            "from_page": self.from_page, "to_page": self.to_page,
            "enable_vectorization": self.enable_vectorization, "enable_storage": self.enable_storage,
            "status": self.status.value, "created_at": self.created_at,
            "started_at": self.started_at, "completed_at": self.completed_at,
            "error": self.error, "result": self.result
        }


class AsyncDocumentProcessor:
    """异步文档处理器"""

    def __init__(self):
        self.max_workers = int(os.getenv("ASYNC_PROCESSOR_MAX_WORKERS", "3"))
        self.task_timeout = int(os.getenv("ASYNC_PROCESSOR_TASK_TIMEOUT", "300"))
        self._task_prefix = "rag:async_task:"
        self._redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD") or None,
            decode_responses=True
        )
        self._task_queue = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._running = False
        self._workers = []

    def _task_key(self, task_id: str) -> str:
        return f"{self._task_prefix}{task_id}"

    def _save_task(self, task: DocumentTask):
        self._redis.setex(self._task_key(task.task_id), 3600, json.dumps(task.to_dict()))

    def _load_task(self, task_id: str) -> Optional[DocumentTask]:
        data = self._redis.get(self._task_key(task_id))
        if data:
            d = json.loads(data)
            return DocumentTask(
                task_id=d.get('task_id'),
                file_path=d.get('file_path', ''),
                file_name=d.get('file_name', ''),
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
                result=d.get('result')
            )
        return None

    async def start(self):
        if self._running:
            return
        self._running = True
        for i in range(self.max_workers):
            self._workers.append(asyncio.create_task(self._worker(i)))
        # 恢复未完成任务
        for key in self._redis.keys(f"{self._task_prefix}*"):
            task = self._load_task(key.split(':')[-1])
            if task and task.status in (TaskStatus.PENDING, TaskStatus.PROCESSING):
                await self._task_queue.put(task)
        logger.info(f"异步处理器启动, workers={self.max_workers}")

    async def stop(self):
        self._running = False
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._executor.shutdown()
        logger.info("异步处理器已停止")

    async def submit_task(self, file_path: str, file_name: str, user_level: str,
                          chunk_size: int = 256, from_page: int = 0, to_page: int = 100000,
                          enable_vectorization: bool = True, enable_storage: bool = True) -> str:
        task_id = hashlib.md5(f"{file_path}_{time.time()}".encode()).hexdigest()[:16]
        task = DocumentTask(task_id=task_id, file_path=file_path, file_name=file_name,
                            user_level=user_level, chunk_size=chunk_size, from_page=from_page,
                            to_page=to_page, enable_vectorization=enable_vectorization,
                            enable_storage=enable_storage)
        self._save_task(task)
        await self._task_queue.put(task)
        logger.info(f"任务提交: {task_id} - {file_name}")
        return task_id

    async def get_task_status(self, task_id: str) -> Optional[Dict]:
        task = self._load_task(task_id)
        if not task:
            return None
        return {"task_id": task.task_id, "file_name": task.file_name, "status": task.status.value, "error": task.error}

    async def _worker(self, worker_id: int):
        while self._running:
            try:
                task = await asyncio.wait_for(self._task_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            task.status = TaskStatus.PROCESSING
            task.started_at = time.time()  # 添加
            self._save_task(task)

            try:
                from . import process_document
                await asyncio.to_thread(process_document,
                                        task.file_path, task.chunk_size, task.enable_vectorization, task.enable_storage,
                                        task.from_page, task.to_page, user_level=task.user_level)
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()  # 添加
                logger.info(f"任务完成: {task.task_id}")
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()  # 添加
                task.error = str(e)
                logger.error(f"任务失败: {task.task_id} - {e}")
            finally:
                self._save_task(task)
                self._task_queue.task_done()


_processor = None


def get_async_processor() -> AsyncDocumentProcessor:
    global _processor
    if _processor is None:
        _processor = AsyncDocumentProcessor()
    return _processor


async def init_async_processor():
    await get_async_processor().start()


async def shutdown_async_processor():
    await get_async_processor().stop()