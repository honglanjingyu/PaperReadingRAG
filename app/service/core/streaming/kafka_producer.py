# app/service/core/streaming/kafka_producer.py

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from kafka import KafkaProducer, KafkaAdminClient
    from kafka.admin import NewTopic
    from kafka.errors import TopicAlreadyExistsError, KafkaError

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.warning("kafka-python 未安装，请运行: pip install kafka-python")


class ChangeEventType(str, Enum):
    """变更事件类型"""
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    UPSERT = "upsert"


@dataclass
class DocumentChangeEvent:
    """文档变更事件"""
    event_id: str
    event_type: ChangeEventType
    filename: str
    user_level: str
    timestamp: float
    # 变更内容
    old_content_hash: Optional[str] = None
    new_content_hash: Optional[str] = None
    old_chunk_ids: Optional[List[str]] = None
    new_chunk_ids: Optional[List[str]] = None
    # 完整内容（用于处理）
    content: Optional[str] = None
    file_path: Optional[str] = None
    # 元数据
    metadata: Dict[str, Any] = None

    def to_json(self) -> str:
        data = asdict(self)
        return json.dumps(data, ensure_ascii=False)


class KafkaChangeProducer:
    """
    Kafka 变更事件生产者

    职责：
    1. 监听文档变更（文件上传、修改、删除）
    2. 将变更事件发送到 Kafka topic
    3. 支持批量发送和异步确认
    4. 自动创建缺失的 topics
    """

    # Topic 配置
    TOPIC_DOCUMENT_CHANGES = os.getenv("KAFKA_TOPIC_DOCUMENT_CHANGES", "rag-document-changes")
    TOPIC_DOCUMENT_UPSERT = os.getenv("KAFKA_TOPIC_DOCUMENT_UPSERT", "rag-document-upsert")
    TOPIC_DOCUMENT_DELETE = os.getenv("KAFKA_TOPIC_DOCUMENT_DELETE", "rag-document-delete")

    # 所有需要创建的 topics
    REQUIRED_TOPICS = [
        TOPIC_DOCUMENT_CHANGES,
        TOPIC_DOCUMENT_UPSERT,
        TOPIC_DOCUMENT_DELETE
    ]

    def __init__(self):
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python 未安装")

        self.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.client_id = os.getenv("KAFKA_CLIENT_ID", "rag-producer")

        # ========== 先定义配置属性 ==========
        # 批量发送配置
        self.batch_size = int(os.getenv("KAFKA_BATCH_SIZE", "100"))
        self.linger_ms = int(os.getenv("KAFKA_LINGER_MS", "10"))

        # 压缩配置（节省带宽）
        self.compression_type = os.getenv("KAFKA_COMPRESSION_TYPE", "snappy")

        # ========== 初始化组件 ==========
        self._producer = None
        self._admin_client = None

        # 先创建 topics
        self._ensure_topics_exist()

        # 再初始化生产者
        self._init_producer()

        logger.info(f"KafkaChangeProducer 初始化: servers={self.bootstrap_servers}, "
                    f"batch_size={self.batch_size}, compression={self.compression_type}")

    def _ensure_topics_exist(self):
        """确保所有必需的 topics 存在，不存在则自动创建"""
        try:
            # 创建 AdminClient
            self._admin_client = KafkaAdminClient(
                bootstrap_servers=self.bootstrap_servers.split(','),
                client_id=f"{self.client_id}-admin",
                request_timeout_ms=30000
            )

            # 获取已存在的 topics
            existing_topics = set(self._admin_client.list_topics())
            logger.info(f"已存在的 Kafka topics: {existing_topics}")

            # 创建缺失的 topics
            topics_to_create = []
            for topic in self.REQUIRED_TOPICS:
                if topic not in existing_topics:
                    topics_to_create.append(NewTopic(
                        name=topic,
                        num_partitions=int(os.getenv("KAFKA_NUM_PARTITIONS", "3")),
                        replication_factor=int(os.getenv("KAFKA_REPLICATION_FACTOR", "1"))
                    ))
                    logger.info(f"需要创建 topic: {topic}")

            if topics_to_create:
                try:
                    self._admin_client.create_topics(new_topics=topics_to_create, validate_only=False)
                    logger.info(f"成功创建 topics: {[t.name for t in topics_to_create]}")
                except TopicAlreadyExistsError:
                    logger.info("Topics 已存在，跳过创建")
                except Exception as e:
                    logger.warning(f"创建 topics 失败（可能权限不足）: {e}")
            else:
                logger.info("所有必需的 topics 已存在")

        except Exception as e:
            logger.warning(f"连接 Kafka Admin 失败，无法自动创建 topics: {e}")
            logger.info("请手动创建以下 topics:")
            for topic in self.REQUIRED_TOPICS:
                logger.info(f"  - {topic}")

    def _init_producer(self):
        """初始化 Kafka 生产者"""
        try:
            # 基础配置参数
            producer_configs = {
                'bootstrap_servers': self.bootstrap_servers.split(','),
                'client_id': self.client_id,
                # 序列化配置
                'value_serializer': lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'),
                'key_serializer': lambda k: k.encode('utf-8') if k else None,
                # 性能配置
                'batch_size': self.batch_size * 1024,  # 转换为字节
                'linger_ms': self.linger_ms,
                'compression_type': self.compression_type,
                # 可靠性配置
                'acks': 'all',  # 等待所有副本确认
                'retries': 3,
                'max_in_flight_requests_per_connection': 5,
                # 缓冲区配置
                'buffer_memory': 33554432,  # 32MB
                # 超时配置（使用正确的参数名）
                'request_timeout_ms': 30000,
                'max_block_ms': 60000,  # 替代 delivery_timeout_ms
            }

            self._producer = KafkaProducer(**producer_configs)
            logger.info("Kafka 生产者初始化成功")
        except Exception as e:
            logger.error(f"Kafka 生产者初始化失败: {e}")
            raise

    def _get_partition_key(self, filename: str) -> str:
        """获取分区键（确保同一文档的事件有序）"""
        return filename

    def send_change_event(self, event: DocumentChangeEvent, topic: str = None) -> Optional[Dict]:
        """发送变更事件到 Kafka"""
        if self._producer is None:
            logger.error("Kafka 生产者未初始化")
            return None

        topic = topic or self.TOPIC_DOCUMENT_CHANGES
        key = self._get_partition_key(event.filename)

        try:
            # 记录发送前的详细信息
            logger.info(f"📤 [Kafka发送] 准备发送事件: topic={topic}, key={key}, "
                        f"event_type={event.event_type.value}, filename={event.filename}, "
                        f"user_level={event.user_level}, content_length={len(event.content) if event.content else 0}")

            value = event.to_json() if hasattr(event, 'to_json') else json.dumps(event.__dict__, ensure_ascii=False)

            # 记录消息大小
            msg_size = len(value.encode('utf-8'))
            logger.debug(f"📤 [Kafka发送] 消息大小: {msg_size} bytes")

            future = self._producer.send(
                topic=topic,
                key=key,
                value=value,
                timestamp_ms=int(event.timestamp * 1000)
            )

            # 添加回调来记录发送结果
            def on_send_success(record_metadata):
                logger.info(f"✅ [Kafka发送成功] topic={record_metadata.topic}, "
                            f"partition={record_metadata.partition}, offset={record_metadata.offset}, "
                            f"filename={event.filename}, event_type={event.event_type.value}")

            def on_send_error(excp):
                logger.error(f"❌ [Kafka发送失败] filename={event.filename}, "
                             f"event_type={event.event_type.value}, error={excp}")

            future.add_callback(on_send_success)
            future.add_errback(on_send_error)

            logger.debug(f"📤 [Kafka发送] 事件已加入发送队列: topic={topic}, key={key}, type={event.event_type.value}")
            return {"topic": topic, "partition": None, "offset": None}

        except Exception as e:
            logger.error(f"❌ [Kafka发送异常] filename={event.filename}, error={e}", exc_info=True)
            return None

    def send_batch_events(
            self,
            events: List[DocumentChangeEvent],
            topic: str = None
    ) -> List[Dict]:
        """批量发送事件"""
        results = []
        for event in events:
            result = self.send_change_event(event, topic)
            results.append(result)

        # 刷新确保所有消息发出
        self.flush()

        success_count = len([r for r in results if r is not None])
        logger.info(f"批量发送完成: {success_count}/{len(events)}")
        return results

    def send_upsert_event(
            self,
            filename: str,
            content: str,
            user_level: str = "normal",
            file_path: str = None
    ) -> Optional[Dict]:
        """
        发送 Upsert 事件（新增或更新）
        """
        import hashlib
        import uuid

        event = DocumentChangeEvent(
            event_id=str(uuid.uuid4()),
            event_type=ChangeEventType.UPSERT,
            filename=filename,
            user_level=user_level,
            timestamp=datetime.now().timestamp(),
            new_content_hash=hashlib.md5(content.encode()).hexdigest() if content else None,
            content=content,
            file_path=file_path,
            metadata={
                "source": "api_upload",
                "content_length": len(content) if content else 0
            }
        )

        return self.send_change_event(event, self.TOPIC_DOCUMENT_UPSERT)

    def send_delete_event(
            self,
            filename: str,
            user_level: str = "normal",
            old_chunk_ids: List[str] = None
    ) -> Optional[Dict]:
        """发送删除事件"""
        import uuid

        event = DocumentChangeEvent(
            event_id=str(uuid.uuid4()),
            event_type=ChangeEventType.DELETE,
            filename=filename,
            user_level=user_level,
            timestamp=datetime.now().timestamp(),
            old_chunk_ids=old_chunk_ids,
            metadata={"source": "api_delete"}
        )

        return self.send_change_event(event, self.TOPIC_DOCUMENT_DELETE)

    def flush(self):
        """刷新所有未发送的消息"""
        if self._producer:
            self._producer.flush()

    def close(self):
        """关闭生产者"""
        if self._producer:
            self.flush()
            self._producer.close()
            logger.info("Kafka 生产者已关闭")
        if self._admin_client:
            self._admin_client.close()


# 全局单例
_kafka_producer = None


def get_kafka_producer() -> Optional[KafkaChangeProducer]:
    """获取 Kafka 生产者实例"""
    global _kafka_producer
    if _kafka_producer is None:
        try:
            _kafka_producer = KafkaChangeProducer()
        except Exception as e:
            logger.error(f"创建 Kafka 生产者失败: {e}")
            return None
    return _kafka_producer


def is_kafka_enabled() -> bool:
    """检查 Kafka 是否启用"""
    enabled = os.getenv("ENABLE_KAFKA_STREAMING", "false").lower() == "true"
    return enabled and KAFKA_AVAILABLE


__all__ = [
    'KafkaChangeProducer',
    'get_kafka_producer',
    'is_kafka_enabled',
    'DocumentChangeEvent',
    'ChangeEventType'
]