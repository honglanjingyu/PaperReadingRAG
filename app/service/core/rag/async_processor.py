# app/service/core/rag/async_processor.py

import asyncio
import os
import hashlib
import time
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import uuid

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DocumentTask:
    """文档处理任务"""
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
    started_at: float = None
    completed_at: float = None
    error: Optional[str] = None
    result: Optional[Any] = None
    callback: Optional[Callable] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class AsyncDocumentProcessor:
    """
    异步文档处理器
    使用队列管理并发，确保数据一致性
    """

    def __init__(
            self,
            max_workers: int = None,
            queue_size: int = None,
            task_timeout: int = None
    ):
        """
        初始化异步处理器

        Args:
            max_workers: 最大并发工作进程数（从环境变量读取）
            queue_size: 队列最大长度（从环境变量读取）
            task_timeout: 单个任务超时时间（秒，从环境变量读取）
        """
        # 从环境变量读取配置，支持参数传入作为备选
        self.max_workers = max_workers or int(os.getenv("ASYNC_PROCESSOR_MAX_WORKERS", "3"))
        self.queue_size = queue_size or int(os.getenv("ASYNC_PROCESSOR_QUEUE_SIZE", "100"))
        self.task_timeout = task_timeout or int(os.getenv("ASYNC_PROCESSOR_TASK_TIMEOUT", "300"))

        # 队列和任务管理
        self._task_queue = asyncio.Queue(maxsize=self.queue_size)
        self._tasks: Dict[str, DocumentTask] = {}
        self._workers: List[asyncio.Task] = []
        self._running = False

        # 线程池用于执行同步的文档处理函数
        # 关键：max_workers 从环境变量读取，控制并发 API 调用数
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # 统计信息
        self._stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_processing_time": 0
        }

        logger.info(
            f"异步文档处理器初始化: max_workers={self.max_workers}, queue_size={self.queue_size}, task_timeout={self.task_timeout}")

    async def start(self):
        """启动处理器"""
        if self._running:
            logger.warning("处理器已经在运行中")
            return

        self._running = True

        # 创建工作进程
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)
            logger.info(f"工作进程 {i} 已启动")

        # 启动监控进程
        self._monitor_task = asyncio.create_task(self._monitor())

        logger.info(f"异步文档处理器已启动，工作进程数: {self.max_workers}")

    async def stop(self):
        """停止处理器"""
        if not self._running:
            return

        logger.info("正在停止异步文档处理器...")
        self._running = False

        # 等待队列清空
        await self._task_queue.join()

        # 取消所有工作进程
        for worker in self._workers:
            worker.cancel()

        # 等待工作进程结束
        await asyncio.gather(*self._workers, return_exceptions=True)

        # 停止监控进程
        self._monitor_task.cancel()

        # 关闭线程池
        self._executor.shutdown(wait=True)

        logger.info("异步文档处理器已停止")

    async def submit_task(
            self,
            file_path: str,
            file_name: str,
            user_level: str,
            chunk_size: int = 256,
            from_page: int = 0,
            to_page: int = 100000,
            enable_vectorization: bool = True,
            enable_storage: bool = True,
            callback: Optional[Callable] = None
    ) -> str:
        """
        提交文档处理任务

        Returns:
            task_id: 任务ID
        """
        # 生成唯一任务ID
        task_id = hashlib.md5(f"{file_path}_{time.time()}_{uuid.uuid4()}".encode()).hexdigest()[:16]

        # 创建任务
        task = DocumentTask(
            task_id=task_id,
            file_path=file_path,
            file_name=file_name,
            user_level=user_level,
            chunk_size=chunk_size,
            from_page=from_page,
            to_page=to_page,
            enable_vectorization=enable_vectorization,
            enable_storage=enable_storage,
            callback=callback
        )

        # 存储任务
        self._tasks[task_id] = task
        self._stats["total_tasks"] += 1

        # 加入队列
        await self._task_queue.put(task)

        logger.info(f"任务 {task_id} 已提交: {file_name} (等级={user_level})")
        return task_id

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        task = self._tasks.get(task_id)
        if not task:
            return None

        return {
            "task_id": task.task_id,
            "file_name": task.file_name,
            "user_level": task.user_level,
            "status": task.status.value,
            "error": task.error,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "processing_time": (task.completed_at - task.started_at) if task.completed_at and task.started_at else None,
            "result": task.result
        }

    async def wait_for_task(self, task_id: str, timeout: int = 300) -> Dict[str, Any]:
        """等待任务完成"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = await self.get_task_status(task_id)
            if not status:
                raise Exception(f"任务 {task_id} 不存在")

            if status["status"] == TaskStatus.COMPLETED.value:
                return status
            elif status["status"] == TaskStatus.FAILED.value:
                raise Exception(f"任务失败: {status['error']}")

            await asyncio.sleep(0.5)

        raise Exception(f"任务 {task_id} 超时")

    async def _worker(self, worker_id: int):
        """
        工作进程
        从队列中取出任务并处理
        """
        logger.info(f"工作进程 {worker_id} 开始运行")

        while self._running:
            try:
                # 从队列获取任务（带超时，避免阻塞）
                task = await asyncio.wait_for(self._task_queue.get(), timeout=1.0)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"工作进程 {worker_id} 获取任务失败: {e}")
                continue

            # 处理任务
            await self._process_task(worker_id, task)

            # 标记任务完成
            self._task_queue.task_done()

        logger.info(f"工作进程 {worker_id} 已停止")

    async def _process_task(self, worker_id: int, task: DocumentTask):
        """
        处理单个任务
        """
        logger.info(f"工作进程 {worker_id} 开始处理任务 {task.task_id}: {task.file_name}")

        # 更新任务状态
        task.status = TaskStatus.PROCESSING
        task.started_at = time.time()

        try:
            # 导入文档处理函数
            from app.service.core.rag.processor import process_document

            # 在线程池中执行同步的文档处理函数
            # 使用 asyncio.to_thread (Python 3.9+)
            # 这会在 ThreadPoolExecutor 中执行，不会阻塞事件循环
            result = await asyncio.to_thread(
                self._process_document_sync,
                task
            )

            # 更新任务状态为成功
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.result = result

            # 更新统计
            self._stats["completed_tasks"] += 1
            processing_time = task.completed_at - task.started_at
            self._update_avg_time(processing_time)

            logger.info(
                f"工作进程 {worker_id} 完成任务 {task.task_id}: "
                f"{task.file_name}, 耗时: {processing_time:.2f}s"
            )

            # 执行回调（如果有）
            if task.callback:
                try:
                    if asyncio.iscoroutinefunction(task.callback):
                        await task.callback(task.task_id, True, result)
                    else:
                        task.callback(task.task_id, True, result)
                except Exception as e:
                    logger.error(f"任务 {task.task_id} 回调执行失败: {e}")

        except Exception as e:
            # 任务处理失败
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            task.error = str(e)

            self._stats["failed_tasks"] += 1

            logger.error(
                f"工作进程 {worker_id} 任务 {task.task_id} 失败: "
                f"{task.file_name}, 错误: {e}",
                exc_info=True
            )

            # 执行回调（如果有）
            if task.callback:
                try:
                    if asyncio.iscoroutinefunction(task.callback):
                        await task.callback(task.task_id, False, str(e))
                    else:
                        task.callback(task.task_id, False, str(e))
                except Exception as cb_e:
                    logger.error(f"任务 {task.callback} 回调执行失败: {cb_e}")

    def _process_document_sync(self, task: DocumentTask) -> Any:
        """
        同步的文档处理函数（在线程池中执行）
        这里包含 MinerU API 调用，会被放到线程池中执行，避免阻塞事件循环
        """
        from app.service.core.rag.processor import process_document

        logger.info(f"开始处理文档: {task.file_name} (工作进程)")

        # 调用原有的文档处理函数
        # 这个函数内部会调用 MinerU API，是同步阻塞的
        # 但由于在 ThreadPoolExecutor 中运行，不会阻塞主事件循环
        result = process_document(
            file_path=task.file_path,
            chunk_size=task.chunk_size,
            enable_vectorization=task.enable_vectorization,
            enable_storage=task.enable_storage,
            from_page=task.from_page,
            to_page=task.to_page,
            verbose=False,
            user_level=task.user_level
        )

        logger.info(f"文档处理完成: {task.file_name}, 生成了 {len(result) if result else 0} 个分块")
        return result

    def _update_avg_time(self, new_time: float):
        """更新平均处理时间"""
        total = self._stats["completed_tasks"]
        if total == 1:
            self._stats["avg_processing_time"] = new_time
        else:
            self._stats["avg_processing_time"] = (
                    (self._stats["avg_processing_time"] * (total - 1) + new_time) / total
            )

    async def _monitor(self):
        """
        监控进程
        定期输出统计信息，清理过期任务
        """
        logger.info("监控进程已启动")

        while self._running:
            try:
                await asyncio.sleep(30)  # 每30秒输出一次

                # 输出统计信息
                queue_size = self._task_queue.qsize()
                logger.info(
                    f"处理器统计: 总任务={self._stats['total_tasks']}, "
                    f"完成={self._stats['completed_tasks']}, "
                    f"失败={self._stats['failed_tasks']}, "
                    f"队列长度={queue_size}, "
                    f"平均处理时间={self._stats['avg_processing_time']:.2f}s"
                )

                # 清理超过1小时的任务记录（可选）
                current_time = time.time()
                to_delete = []
                for task_id, task in self._tasks.items():
                    if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                        if current_time - task.completed_at > 3600:  # 1小时
                            to_delete.append(task_id)

                for task_id in to_delete:
                    del self._tasks[task_id]

                if to_delete:
                    logger.info(f"清理了 {len(to_delete)} 个过期任务")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控进程异常: {e}")

        logger.info("监控进程已停止")


# 全局处理器实例
_async_processor: Optional[AsyncDocumentProcessor] = None


def get_async_processor() -> AsyncDocumentProcessor:
    """获取异步处理器单例"""
    global _async_processor
    if _async_processor is None:
        _async_processor = AsyncDocumentProcessor()
    return _async_processor


async def init_async_processor():
    """初始化异步处理器（在应用启动时调用）"""
    processor = get_async_processor()
    await processor.start()


async def shutdown_async_processor():
    """关闭异步处理器（在应用关闭时调用）"""
    processor = get_async_processor()
    await processor.stop()