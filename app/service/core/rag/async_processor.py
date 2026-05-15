# app/service/core/rag/async_processor.py

import asyncio
import os
import hashlib
import time
import logging
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import uuid

logger = logging.getLogger(__name__)

# 尝试导入 Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis 模块未安装，异步任务状态将无法跨实例共享")


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

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于 Redis 存储）"""
        data = asdict(self)
        data['status'] = self.status.value
        # 只保留 result 的基本信息，避免存储过大的数据
        if data.get('result') and isinstance(data['result'], list):
            data['result'] = {
                'chunks_count': len(data['result']),
                'preview': str(data['result'][:2]) if data['result'] else []
            }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentTask':
        """从字典创建任务"""
        status = TaskStatus(data.get('status', 'pending'))
        data['status'] = status
        return cls(**data)


class AsyncDocumentProcessor:
    """
    异步文档处理器 - 使用 Redis 存储任务状态
    支持多实例部署，任务状态跨实例共享
    """

    def __init__(
            self,
            max_workers: int = None,
            queue_size: int = None,
            task_timeout: int = None,
            redis_client: redis.Redis = None
    ):
        """
        初始化异步处理器

        Args:
            max_workers: 最大并发工作进程数（从环境变量读取）
            queue_size: 队列最大长度（从环境变量读取）
            task_timeout: 单个任务超时时间（秒，从环境变量读取）
            redis_client: Redis 客户端实例（可选，不传则自动创建）
        """
        # 从环境变量读取配置
        self.max_workers = max_workers or int(os.getenv("ASYNC_PROCESSOR_MAX_WORKERS", "3"))
        self.queue_size = queue_size or int(os.getenv("ASYNC_PROCESSOR_QUEUE_SIZE", "100"))
        self.task_timeout = task_timeout or int(os.getenv("ASYNC_PROCESSOR_TASK_TIMEOUT", "300"))

        # Redis key 前缀
        self._task_prefix = "rag:async_task:"
        self._queue_prefix = "rag:task_queue"
        self._stats_prefix = "rag:processor_stats"

        # Redis 连接
        if redis_client:
            self.redis_client = redis_client
        elif REDIS_AVAILABLE:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                password=os.getenv("REDIS_PASSWORD") or None,
                db=int(os.getenv("REDIS_DB", 0)),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            try:
                self.redis_client.ping()
                logger.info("Redis 连接成功，异步任务状态将使用 Redis 存储")
            except Exception as e:
                logger.error(f"Redis 连接失败: {e}")
                self.redis_client = None
                raise RuntimeError(f"Redis 连接失败，无法使用异步任务处理器: {e}")
        else:
            self.redis_client = None
            raise ImportError("redis 模块未安装且无可用 Redis 连接，请安装 redis 模块并确保 Redis 服务已启动")

        # 队列和任务管理
        self._task_queue = asyncio.Queue(maxsize=self.queue_size)
        self._workers: List[asyncio.Task] = []
        self._running = False

        # 线程池用于执行同步的文档处理函数
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # 统计信息（存储在 Redis 中）
        self._init_stats()

        logger.info(
            f"异步文档处理器初始化: max_workers={self.max_workers}, queue_size={self.queue_size}, "
            f"task_timeout={self.task_timeout}, redis_available={self.redis_client is not None}"
        )

    def _init_stats(self):
        """初始化统计信息"""
        if self.redis_client:
            if not self.redis_client.exists(self._stats_prefix):
                self.redis_client.hset(self._stats_prefix, mapping={
                    "total_tasks": 0,
                    "completed_tasks": 0,
                    "failed_tasks": 0,
                    "avg_processing_time": 0
                })

    def _get_task_key(self, task_id: str) -> str:
        """获取任务的 Redis key"""
        return f"{self._task_prefix}{task_id}"

    def _save_task(self, task: DocumentTask):
        """保存任务到 Redis"""
        if self.redis_client:
            key = self._get_task_key(task.task_id)
            self.redis_client.setex(key, self.task_timeout + 3600, json.dumps(task.to_dict(), ensure_ascii=False))

    def _load_task(self, task_id: str) -> Optional[DocumentTask]:
        """从 Redis 加载任务"""
        if not self.redis_client:
            return None

        key = self._get_task_key(task_id)
        data = self.redis_client.get(key)
        if data:
            try:
                return DocumentTask.from_dict(json.loads(data))
            except Exception as e:
                logger.error(f"加载任务失败 {task_id}: {e}")
        return None

    def _delete_task(self, task_id: str):
        """从 Redis 删除任务"""
        if self.redis_client:
            key = self._get_task_key(task_id)
            self.redis_client.delete(key)

    def _update_stats(self, field: str, delta: int = 1):
        """更新统计信息"""
        if self.redis_client:
            self.redis_client.hincrby(self._stats_prefix, field, delta)

    def _update_avg_time(self, new_time: float):
        """更新平均处理时间"""
        if self.redis_client:
            completed = int(self.redis_client.hget(self._stats_prefix, "completed_tasks") or 0)
            if completed == 0:
                self.redis_client.hset(self._stats_prefix, "avg_processing_time", new_time)
            else:
                current_avg = float(self.redis_client.hget(self._stats_prefix, "avg_processing_time") or 0)
                new_avg = (current_avg * completed + new_time) / (completed + 1)
                self.redis_client.hset(self._stats_prefix, "avg_processing_time", new_avg)

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

        # 恢复未完成的任务
        await self._recover_pending_tasks()

        logger.info(f"异步文档处理器已启动，工作进程数: {self.max_workers}")

    async def _recover_pending_tasks(self):
        """恢复未完成的任务（服务重启时）"""
        if not self.redis_client:
            return

        try:
            # 查找所有 pending 或 processing 状态的任务
            keys = self.redis_client.keys(f"{self._task_prefix}*")
            recovered_count = 0

            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    task_data = json.loads(data)
                    status = task_data.get('status')
                    if status in ['pending', 'processing']:
                        # 重新加入队列
                        task = DocumentTask.from_dict(task_data)
                        await self._task_queue.put(task)
                        recovered_count += 1
                        logger.info(f"恢复任务: {task.task_id} ({task.file_name})")

            if recovered_count > 0:
                logger.info(f"恢复 {recovered_count} 个未完成任务")

        except Exception as e:
            logger.error(f"恢复任务失败: {e}")

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
            enable_storage=enable_storage
        )

        # 保存任务到 Redis
        self._save_task(task)

        # 更新统计
        self._update_stats("total_tasks")

        # 加入队列
        await self._task_queue.put(task)

        logger.info(f"任务 {task_id} 已提交: {file_name} (等级={user_level})")
        return task_id

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        task = self._load_task(task_id)
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
        self._save_task(task)

        try:
            # 导入文档处理函数
            from app.service.core.rag.processor import process_document

            # 在线程池中执行同步的文档处理函数
            result = await asyncio.to_thread(
                self._process_document_sync,
                task
            )

            # 更新任务状态为成功
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.result = result
            self._save_task(task)

            # 更新统计
            self._update_stats("completed_tasks")
            processing_time = task.completed_at - task.started_at
            self._update_avg_time(processing_time)

            logger.info(
                f"工作进程 {worker_id} 完成任务 {task.task_id}: "
                f"{task.file_name}, 耗时: {processing_time:.2f}s"
            )

        except Exception as e:
            # 任务处理失败
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            task.error = str(e)
            self._save_task(task)

            self._update_stats("failed_tasks")

            logger.error(
                f"工作进程 {worker_id} 任务 {task.task_id} 失败: "
                f"{task.file_name}, 错误: {e}",
                exc_info=True
            )

    def _process_document_sync(self, task: DocumentTask) -> Any:
        """
        同步的文档处理函数（在线程池中执行）
        """
        from app.service.core.rag.processor import process_document

        logger.info(f"开始处理文档: {task.file_name} (工作进程)")

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

                if self.redis_client:
                    stats = self.redis_client.hgetall(self._stats_prefix)
                    total = int(stats.get("total_tasks", 0))
                    completed = int(stats.get("completed_tasks", 0))
                    failed = int(stats.get("failed_tasks", 0))
                    avg_time = float(stats.get("avg_processing_time", 0))

                    logger.info(
                        f"处理器统计: 总任务={total}, "
                        f"完成={completed}, "
                        f"失败={failed}, "
                        f"队列长度={queue_size}, "
                        f"平均处理时间={avg_time:.2f}s"
                    )

                # 清理超过1小时的任务记录
                if self.redis_client:
                    keys = self.redis_client.keys(f"{self._task_prefix}*")
                    current_time = time.time()
                    deleted = 0

                    for key in keys:
                        data = self.redis_client.get(key)
                        if data:
                            task_data = json.loads(data)
                            completed_at = task_data.get('completed_at')
                            if completed_at and current_time - completed_at > 3600:
                                self.redis_client.delete(key)
                                deleted += 1

                    if deleted > 0:
                        logger.info(f"清理了 {deleted} 个过期任务")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控进程异常: {e}")

        logger.info("监控进程已停止")

    async def get_stats(self) -> Dict[str, Any]:
        """获取处理器统计信息"""
        stats = {
            "max_workers": self.max_workers,
            "queue_size": self.queue_size,
            "running": self._running
        }

        if self.redis_client:
            redis_stats = self.redis_client.hgetall(self._stats_prefix)
            stats.update({
                "total_tasks": int(redis_stats.get("total_tasks", 0)),
                "completed_tasks": int(redis_stats.get("completed_tasks", 0)),
                "failed_tasks": int(redis_stats.get("failed_tasks", 0)),
                "avg_processing_time": float(redis_stats.get("avg_processing_time", 0)),
                "redis_available": True
            })
        else:
            stats.update({
                "redis_available": False,
                "error": "Redis 不可用"
            })

        return stats


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


__all__ = [
    'AsyncDocumentProcessor',
    'get_async_processor',
    'init_async_processor',
    'shutdown_async_processor',
    'TaskStatus',
    'DocumentTask'
]