# app/service/core/streaming/__init__.py
# 在模块加载时添加日志

"""
流式处理模块 - CDC + Kafka 增量更新
"""

import os
import logging

logger = logging.getLogger(__name__)

# 在模块加载时输出 Kafka 配置状态
KAFKA_ENABLED = os.getenv("ENABLE_KAFKA_STREAMING", "false").lower() == "true"
if KAFKA_ENABLED:
    logger.info("=" * 50)
    logger.info("📡 [Kafka] 流式处理模块已启用")
    logger.info(f"   - Bootstrap Servers: {os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')}")
    logger.info(f"   - Consumer Group: {os.getenv('KAFKA_CONSUMER_GROUP_ID', 'rag-consumer-group')}")
    logger.info(f"   - Upsert Topic: {os.getenv('KAFKA_TOPIC_DOCUMENT_UPSERT', 'rag-document-upsert')}")
    logger.info(f"   - Delete Topic: {os.getenv('KAFKA_TOPIC_DOCUMENT_DELETE', 'rag-document-delete')}")
    logger.info("=" * 50)
else:
    logger.info("📡 [Kafka] 流式处理模块已禁用 (设置 ENABLE_KAFKA_STREAMING=true 启用)")

from .kafka_producer import (
    KafkaChangeProducer,
    get_kafka_producer,
    is_kafka_enabled,
    DocumentChangeEvent,
    ChangeEventType
)

from .kafka_consumer import (
    KafkaChangeConsumer,
    get_kafka_consumer
)

from .stream_processor import (
    StreamProcessor,
    get_stream_processor,
    emit_document_change
)

from .batch_upsert import BatchUpsertOptimizer

__all__ = [
    'KafkaChangeProducer',
    'get_kafka_producer',
    'is_kafka_enabled',
    'DocumentChangeEvent',
    'ChangeEventType',
    'KafkaChangeConsumer',
    'get_kafka_consumer',
    'StreamProcessor',
    'get_stream_processor',
    'emit_document_change',
    'BatchUpsertOptimizer',
]