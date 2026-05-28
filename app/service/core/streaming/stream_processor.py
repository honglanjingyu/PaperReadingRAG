# app/service/core/streaming/stream_processor.py
"""
流式处理器 - Kafka 是唯一文档处理器
"""

import os
import logging
import asyncio
import hashlib
import time
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor

from .kafka_producer import get_kafka_producer, is_kafka_enabled
from .kafka_consumer import get_kafka_consumer

logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    流式处理器 - Kafka 是唯一文档处理器

    工作流程：
    1. 上传 API 接收文件，发送到 Kafka
    2. Kafka 消费者处理文档（向量化、存储）
    3. 分布式锁防止重复处理
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.enabled = is_kafka_enabled()
        self._producer = None
        self._consumer = None
        self._executor = ThreadPoolExecutor(max_workers=4)

        # 锁 TTL（秒），防止死锁
        self._lock_ttl = int(os.getenv("PROCESSING_LOCK_TTL", "300"))

        # 是否启用消费者
        self._consumer_enabled = os.getenv("KAFKA_CONSUMER_ENABLED", "true").lower() == "true"

        if self.enabled:
            self._init_components()

        logger.info(f"StreamProcessor 初始化: enabled={self.enabled}, "
                    f"consumer_enabled={self._consumer_enabled}, lock_ttl={self._lock_ttl}s")

    def _init_components(self):
        """初始化 Kafka 组件"""
        try:
            self._producer = get_kafka_producer()

            if self._consumer_enabled:
                self._consumer = get_kafka_consumer()

                # 注册处理器
                if self._consumer:
                    self._consumer.register_handlers(
                        upsert_handler=self._handle_upsert,
                        delete_handler=self._handle_delete
                    )

            logger.info("Kafka 组件初始化成功")
        except Exception as e:
            logger.error(f"Kafka 组件初始化失败: {e}")
            self.enabled = False

    def _get_redis_client(self):
        """获取 Redis 客户端（用于分布式锁）"""
        try:
            import redis
            return redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                password=os.getenv("REDIS_PASSWORD") or None,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
        except Exception as e:
            logger.error(f"获取 Redis 客户端失败: {e}")
            return None

    def _get_process_lock(self, filename: str) -> bool:
        """
        获取文档处理分布式锁

        Args:
            filename: 文件名

        Returns:
            True: 获取锁成功，可以处理
            False: 获取锁失败，有其他进程正在处理
        """
        redis_client = self._get_redis_client()
        if redis_client is None:
            logger.warning(f"Redis 不可用，跳过锁检查: {filename}")
            return True

        try:
            lock_key = f"rag:processing_lock:{hashlib.md5(filename.encode()).hexdigest()}"

            acquired = redis_client.set(lock_key, str(int(time.time())), nx=True, ex=self._lock_ttl)

            if acquired:
                logger.debug(f"🔒 获取处理锁成功: {filename}")
            else:
                # 检查锁是否过期（防止死锁）
                existing = redis_client.get(lock_key)
                if existing:
                    lock_time = int(existing)
                    if time.time() - lock_time > self._lock_ttl:
                        redis_client.delete(lock_key)
                        acquired = redis_client.set(lock_key, str(int(time.time())), nx=True, ex=self._lock_ttl)
                        if acquired:
                            logger.warning(f"🔒 强制获取过期锁: {filename}")
                        else:
                            logger.info(f"⏭️ 获取锁失败: {filename}")
                    else:
                        logger.info(f"⏭️ 获取锁失败: {filename}")
                else:
                    logger.info(f"⏭️ 获取锁失败: {filename}")

            return bool(acquired)
        except Exception as e:
            logger.error(f"获取锁异常: {filename}, error={e}")
            return True

    def _release_process_lock(self, filename: str):
        """释放文档处理锁"""
        redis_client = self._get_redis_client()
        if redis_client is None:
            return

        try:
            lock_key = f"rag:processing_lock:{hashlib.md5(filename.encode()).hexdigest()}"
            redis_client.delete(lock_key)
            logger.debug(f"🔓 释放处理锁: {filename}")
        except Exception as e:
            logger.warning(f"释放锁失败: {filename}, error={e}")

    def _check_document_exists(self, filename: str, index_name: str = None) -> bool:
        """
        检查文档是否已存在于向量数据库中（幂等性检查）

        Args:
            filename: 文件名
            index_name: 索引名称

        Returns:
            True: 文档已存在
            False: 文档不存在
        """
        try:
            from app.service.core.vector_store import get_vector_store

            index_name = index_name or os.getenv("VECTOR_INDEX_NAME", "rag_documents")
            vector_store = get_vector_store()

            if not vector_store or not vector_store.index_exists(index_name):
                return False

            from pymilvus import Collection
            collection = Collection(index_name)
            collection.load()

            expr = f'docnm == "{filename}"'
            existing = collection.query(
                expr=expr,
                output_fields=["docnm"],
                limit=1
            )

            if existing:
                logger.debug(f"📋 文档已存在于 Milvus: {filename}")
                return True
            return False

        except Exception as e:
            logger.warning(f"检查文档是否存在失败: {filename}, error={e}")
            return False

    async def emit_change_event(
            self,
            filename: str,
            content: str,
            user_level: str = "normal",
            file_path: str = None,
            event_type: str = "upsert"
    ) -> bool:
        """发送变更事件到 Kafka"""
        if not self.enabled or not self._producer:
            logger.warning(f"⚠️ [Kafka] 未启用或生产者不可用: enabled={self.enabled}")
            return False

        logger.info(f"🚀 [Kafka] 发送 {event_type} 事件: filename={filename}, "
                    f"user_level={user_level}, content_length={len(content) if content else 0}")

        try:
            import uuid

            event_data = {
                "event_id": str(uuid.uuid4()),
                "event_type": event_type,
                "filename": filename,
                "user_level": user_level,
                "timestamp": time.time(),
                "content": content,
                "file_path": file_path,
                "new_content_hash": hashlib.md5(content.encode()).hexdigest() if content else None,
                "metadata": {
                    "source": "stream_processor",
                    "content_length": len(content) if content else 0
                }
            }

            if event_type == "upsert":
                topic = self._producer.TOPIC_DOCUMENT_UPSERT
            elif event_type == "delete":
                topic = self._producer.TOPIC_DOCUMENT_DELETE
            else:
                logger.warning(f"⚠️ [Kafka] 未知事件类型: {event_type}")
                return False

            future = await asyncio.to_thread(
                self._producer._producer.send,
                topic=topic,
                key=filename,
                value=event_data
            )

            try:
                await asyncio.to_thread(future.get, timeout=10)
                logger.info(f"✅ [Kafka] 事件发送成功: event_type={event_type}, filename={filename}")
                return True
            except Exception as e:
                logger.error(f"❌ [Kafka] 事件发送确认失败: {filename}, error={e}")
                return False

        except Exception as e:
            logger.error(f"❌ [Kafka] 发送失败: {event_type}, filename={filename}, error={e}", exc_info=True)
            return False

    def _process_document(self, filename: str, content: str, user_level: str, file_path: str) -> int:
        """
        实际处理文档（同步函数，在线程池中执行）

        Returns:
            处理的块数
        """
        from app.service.core.rag import process_document, process_document_with_text
        from app.service.core.cache import get_document_cache
        from app.service.core.graphrag import get_graph_rag_service

        start_time = time.time()

        if file_path and os.path.exists(file_path):
            logger.info(f"📄 [Kafka消费者] 使用文件路径处理: {file_path}")
            result = process_document(
                file_path=file_path,
                enable_vectorization=True,
                enable_storage=True,
                user_level=user_level
            )
        else:
            logger.info(f"📄 [Kafka消费者] 使用文本内容处理: {filename}, content_length={len(content)}")
            result = process_document_with_text(
                text_content=content,
                file_name=filename,
                enable_vectorization=True,
                enable_storage=True,
                user_level=user_level
            )

        elapsed = time.time() - start_time
        chunk_count = len(result) if result else 0

        logger.info(f"✅ [Kafka消费者] 文档处理完成: {filename}, "
                    f"chunks={chunk_count}, elapsed={elapsed:.2f}s")

        if chunk_count > 0:
            doc_cache = get_document_cache()
            doc_cache.set_document_level(filename, user_level)
            logger.debug(f"💾 [Kafka消费者] 文档等级已缓存: {filename} -> {user_level}")

            graph_service = get_graph_rag_service()
            graph_service.invalidate_cache(user_level)
            logger.debug(f"🗑️ [Kafka消费者] GraphRAG 缓存已失效")

        return chunk_count

    def _handle_upsert(self, filename: str, content: str, user_level: str, file_path: str):
        """
        处理 upsert 事件 - Kafka 是唯一处理器
        """
        from app.service.core.cache import get_cache_manager
        import hashlib

        cache = get_cache_manager()

        # ========== 修复：根据文件名生成一致的 task_id ==========
        task_id = hashlib.md5(f"{filename}_{user_level}".encode()).hexdigest()[:16]

        # 更新任务状态：开始处理
        cache.set("task", task_id, {
            "status": "processing",
            "filename": filename,
            "user_level": user_level,
            "progress": 30,
            "message": "正在处理文档...",
            "updated_at": time.time()
        }, ttl=3600)

        logger.info(f"🔄 [Kafka消费者] 收到 upsert 事件: filename={filename}, "
                    f"user_level={user_level}, task_id={task_id}")

        if not self._get_process_lock(filename):
            logger.info(f"⏭️ [Kafka消费者] 文档正在被其他实例处理，跳过: {filename}")
            return

        try:
            # 更新状态：向量化中
            cache.set("task", task_id, {
                "status": "processing",
                "filename": filename,
                "progress": 50,
                "message": "正在向量化...",
                "updated_at": time.time()
            }, ttl=3600)

            # 幂等性检查 - 文档是否已存在
            if self._check_document_exists(filename):
                logger.info(f"⏭️ [Kafka消费者] 文档已存在，跳过重复处理: {filename}")
                cache.set("task", task_id, {
                    "status": "completed",
                    "filename": filename,
                    "progress": 100,
                    "message": "文档已存在，跳过处理",
                    "updated_at": time.time()
                }, ttl=3600)
                return

            # 更新状态：存储中
            cache.set("task", task_id, {
                "status": "processing",
                "filename": filename,
                "progress": 70,
                "message": "正在存储到向量数据库...",
                "updated_at": time.time()
            }, ttl=3600)

            # 实际处理文档
            chunk_count = self._process_document(filename, content, user_level, file_path)

            # 更新状态：完成
            if chunk_count > 0:
                cache.set("task", task_id, {
                    "status": "completed",
                    "filename": filename,
                    "progress": 100,
                    "message": f"处理完成，共 {chunk_count} 个分块",
                    "result": {"chunk_count": chunk_count},
                    "updated_at": time.time()
                }, ttl=3600)
                logger.info(f"✅ [Kafka消费者] 任务状态已更新为 completed: {task_id}")
            else:
                cache.set("task", task_id, {
                    "status": "failed",
                    "filename": filename,
                    "progress": 0,
                    "message": "处理失败，未生成分块",
                    "error": "No chunks generated",
                    "updated_at": time.time()
                }, ttl=3600)

        except Exception as e:
            logger.error(f"❌ [Kafka消费者] upsert 处理失败: {filename}, error={e}", exc_info=True)
            cache.set("task", task_id, {
                "status": "failed",
                "filename": filename,
                "progress": 0,
                "message": f"处理失败: {str(e)}",
                "error": str(e),
                "updated_at": time.time()
            }, ttl=3600)
        finally:
            self._release_process_lock(filename)

    def _handle_delete(self, filename: str, user_level: str, chunk_ids: List[str]):
        """
        处理 delete 事件

        流程：
        1. 获取分布式锁
        2. 执行删除操作
        3. 释放锁
        """
        logger.info(f"🔄 [Kafka消费者] 收到 delete 事件: filename={filename}, "
                    f"user_level={user_level}, chunk_count={len(chunk_ids) if chunk_ids else 0}")

        if not self._get_process_lock(filename):
            logger.info(f"⏭️ [Kafka消费者] 文档正在被其他实例处理，跳过删除: {filename}")
            return

        try:
            from app.api.routes.delete import _delete_from_milvus, _delete_from_elasticsearch
            from app.api.routes.delete import _delete_from_redis_and_memory
            from app.service.core.cache import get_document_cache
            from app.service.core.graphrag import get_graph_rag_service

            index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")
            start_time = time.time()

            milvus_deleted = _delete_from_milvus(filename, index_name)
            logger.debug(f"🗑️ [Kafka消费者] Milvus 删除: {milvus_deleted} 条")

            es_deleted = _delete_from_elasticsearch(filename, index_name)
            logger.debug(f"🗑️ [Kafka消费者] Elasticsearch 删除: {es_deleted} 条")

            redis_deleted = _delete_from_redis_and_memory(filename)
            logger.debug(f"🗑️ [Kafka消费者] Redis 清理: {redis_deleted} 条")

            doc_cache = get_document_cache()
            doc_cache.delete_document_level(filename)
            logger.debug(f"🗑️ [Kafka消费者] 文档等级缓存已删除: {filename}")

            graph_service = get_graph_rag_service()
            graph_service.invalidate_cache(user_level)
            logger.debug(f"🗑️ [Kafka消费者] GraphRAG 缓存已失效")

            elapsed = time.time() - start_time
            logger.info(f"✅ [Kafka消费者] delete 处理完成: filename={filename}, "
                        f"milvus={milvus_deleted}, es={es_deleted}, redis={redis_deleted}, "
                        f"elapsed={elapsed:.2f}s")

        except Exception as e:
            logger.error(f"❌ [Kafka消费者] delete 处理失败: {filename}, error={e}", exc_info=True)
        finally:
            self._release_process_lock(filename)

    def start_consumer(self):
        """启动 Kafka 消费者"""
        if self.enabled and self._consumer and self._consumer_enabled:
            logger.info("=" * 60)
            logger.info("🚀 [Kafka] 正在启动 Kafka 消费者...")
            logger.info(f"📋 [Kafka] 配置信息:")
            logger.info(f"   - bootstrap_servers: {os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')}")
            logger.info(f"   - group_id: {os.getenv('KAFKA_CONSUMER_GROUP_ID', 'rag-consumer-group')}")
            logger.info(f"   - upsert topic: {self._consumer.TOPIC_DOCUMENT_UPSERT}")
            logger.info(f"   - delete topic: {self._consumer.TOPIC_DOCUMENT_DELETE}")
            logger.info(f"   - workers: {self._consumer.max_workers}")
            logger.info(f"   - processing_lock_ttl: {self._lock_ttl}s")
            logger.info(f"   - mode: 唯一处理器模式")
            logger.info("=" * 60)

            self._consumer.start()
            logger.info("✅ [Kafka] Kafka 消费者启动成功（唯一处理器模式）")
        else:
            if not self.enabled:
                logger.info("📡 [Kafka] Kafka 未启用，消费者未启动")
            elif not self._consumer_enabled:
                logger.info("📡 [Kafka] Kafka 消费者已禁用")
            else:
                logger.warning(f"⚠️ [Kafka] 消费者未启动: consumer={self._consumer is not None}")

    def stop_consumer(self):
        """停止 Kafka 消费者"""
        if self.enabled and self._consumer:
            self._consumer.stop()
            logger.info("Kafka 消费者已停止")

    def get_status(self) -> Dict[str, Any]:
        """获取处理器状态"""
        status = {
            "enabled": self.enabled,
            "kafka_available": self._producer is not None,
            "consumer_enabled": self._consumer_enabled,
            "lock_ttl": self._lock_ttl,
            "mode": "unique_processor"
        }

        if self.enabled and self._consumer:
            try:
                status["consumer_lag"] = self._consumer.get_consumer_lag()
            except:
                status["consumer_lag"] = "unknown"

        return status


# 全局单例
_stream_processor = None


def get_stream_processor() -> StreamProcessor:
    """获取流式处理器实例"""
    global _stream_processor
    if _stream_processor is None:
        _stream_processor = StreamProcessor()
    return _stream_processor


async def emit_document_change(
        filename: str,
        content: str,
        user_level: str = "normal",
        file_path: str = None,
        event_type: str = "upsert"
) -> bool:
    """便捷函数：发送文档变更事件"""
    processor = get_stream_processor()
    return await processor.emit_change_event(
        filename, content, user_level, file_path, event_type
    )


__all__ = [
    'StreamProcessor',
    'get_stream_processor',
    'emit_document_change'
]