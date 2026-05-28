# app/service/core/streaming/kafka_consumer.py

"""
Kafka 消费者 - 消费变更事件并处理
"""

import os
import json
import logging
import threading
import time
from typing import Dict, Any, Optional, List, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from kafka import KafkaConsumer, KafkaAdminClient
    from kafka.admin import NewTopic
    from kafka.errors import TopicAlreadyExistsError, KafkaError
    from kafka.structs import TopicPartition

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False


@dataclass
class ConsumerConfig:
    """消费者配置"""
    bootstrap_servers: str
    group_id: str
    topics: List[str]
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = False
    max_poll_records: int = 500
    max_poll_interval_ms: int = 300000
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 3000
    fetch_min_bytes: int = 1
    fetch_max_wait_ms: int = 500


class KafkaChangeConsumer:
    """
    Kafka 变更事件消费者

    职责：
    1. 消费文档变更事件
    2. 调用处理器处理事件
    3. 支持多线程并行处理
    4. 手动提交 offset 确保 at-least-once
    5. 自动确保 topics 存在
    """

    # Topic 配置
    TOPIC_DOCUMENT_UPSERT = os.getenv("KAFKA_TOPIC_DOCUMENT_UPSERT", "rag-document-upsert")
    TOPIC_DOCUMENT_DELETE = os.getenv("KAFKA_TOPIC_DOCUMENT_DELETE", "rag-document-delete")

    REQUIRED_TOPICS = [TOPIC_DOCUMENT_UPSERT, TOPIC_DOCUMENT_DELETE]

    def __init__(self):
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python 未安装")

        self.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.group_id = os.getenv("KAFKA_CONSUMER_GROUP_ID", "rag-consumer-group")

        # Topic 配置
        self.topic_upsert = self.TOPIC_DOCUMENT_UPSERT
        self.topic_delete = self.TOPIC_DOCUMENT_DELETE

        # 消费配置
        self.max_poll_records = int(os.getenv("KAFKA_MAX_POLL_RECORDS", "500"))
        self.max_workers = int(os.getenv("KAFKA_CONSUMER_WORKERS", "4"))

        # 处理函数
        self._upsert_handler: Optional[Callable] = None
        self._delete_handler: Optional[Callable] = None

        # 消费者实例
        self._consumers: List[KafkaConsumer] = []
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._running = False
        self._threads: List[threading.Thread] = []

        # 确保 topics 存在
        self._ensure_topics_exist()

        logger.info(f"KafkaChangeConsumer 初始化: group_id={self.group_id}, "
                    f"workers={self.max_workers}")

    def _ensure_topics_exist(self):
        """确保所有必需的 topics 存在"""
        try:
            admin_client = KafkaAdminClient(
                bootstrap_servers=self.bootstrap_servers.split(','),
                client_id=f"rag-consumer-admin"
            )

            existing_topics = set(admin_client.list_topics())

            topics_to_create = []
            for topic in self.REQUIRED_TOPICS:
                if topic not in existing_topics:
                    topics_to_create.append(NewTopic(
                        name=topic,
                        num_partitions=int(os.getenv("KAFKA_NUM_PARTITIONS", "3")),
                        replication_factor=int(os.getenv("KAFKA_REPLICATION_FACTOR", "1"))
                    ))
                    logger.info(f"消费者需要创建 topic: {topic}")

            if topics_to_create:
                try:
                    admin_client.create_topics(new_topics=topics_to_create, validate_only=False)
                    logger.info(f"消费者成功创建 topics: {[t.name for t in topics_to_create]}")
                except TopicAlreadyExistsError:
                    logger.info("Topics 已存在")
                except Exception as e:
                    logger.warning(f"消费者创建 topics 失败: {e}")
            else:
                logger.debug("消费者检查: 所有必需的 topics 已存在")

            admin_client.close()

        except Exception as e:
            logger.warning(f"消费者无法连接 AdminClient 检查 topics: {e}")

    def register_handlers(
            self,
            upsert_handler: Callable,
            delete_handler: Callable
    ):
        """
        注册事件处理器

        Args:
            upsert_handler: 处理 upsert 事件的函数
                           接收 (filename, content, user_level, file_path)
            delete_handler: 处理 delete 事件的函数
                           接收 (filename, user_level, chunk_ids)
        """
        self._upsert_handler = upsert_handler
        self._delete_handler = delete_handler
        logger.info("事件处理器已注册")

    def _create_consumer(self, topic: str) -> KafkaConsumer:
        """创建 Kafka 消费者"""
        logger.info(f"🔌 [Kafka消费者] 正在创建消费者: topic={topic}, group_id={self.group_id}, "
                    f"bootstrap_servers={self.bootstrap_servers}")

        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap_servers.split(','),
            group_id=self.group_id,
            client_id=f"rag-consumer-{topic}",
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            max_poll_records=self.max_poll_records,
            max_poll_interval_ms=300000,
            session_timeout_ms=30000,
            heartbeat_interval_ms=3000,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')) if v else None,
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            fetch_min_bytes=1,
            fetch_max_wait_ms=500
        )

        # 获取分配的 partitions
        partitions = consumer.assignment()
        logger.info(f"✅ [Kafka消费者] 消费者创建成功: topic={topic}, partitions={partitions}")

        return consumer

    def _process_message(self, message):
        """处理单条消息"""
        try:
            event_data = message.value

            if not event_data:
                return

            # ========== 添加类型检查和兼容处理 ==========
            # 如果 event_data 是字符串，尝试解析为 JSON
            if isinstance(event_data, str):
                try:
                    event_data = json.loads(event_data)
                    logger.debug(f"消息已从字符串解析为 JSON")
                except json.JSONDecodeError as e:
                    logger.error(f"解析 JSON 失败: {e}, 原始消息: {event_data[:200]}")
                    return

            # 确保 event_data 是字典
            if not isinstance(event_data, dict):
                logger.error(f"event_data 不是字典类型: {type(event_data)}")
                return

            event_type = event_data.get('event_type')
            filename = event_data.get('filename')
            user_level = event_data.get('user_level', 'normal')

            logger.info(f"收到 Kafka 消息: event_type={event_type}, filename={filename}, user_level={user_level}")

            if event_type == 'upsert':
                if self._upsert_handler:
                    content = event_data.get('content', '')
                    file_path = event_data.get('file_path')
                    logger.info(f"调用 upsert 处理器: {filename}, content_length={len(content)}")
                    self._upsert_handler(filename, content, user_level, file_path)

            elif event_type == 'delete':
                if self._delete_handler:
                    old_chunk_ids = event_data.get('old_chunk_ids', [])
                    logger.info(f"调用 delete 处理器: {filename}")
                    self._delete_handler(filename, user_level, old_chunk_ids)
            else:
                logger.warning(f"未知的事件类型: {event_type}")

        except Exception as e:
            logger.error(f"处理消息失败: {e}", exc_info=True)

    def _consume_topic(self, topic: str):
        """持续消费单个 topic"""
        consumer = self._create_consumer(topic)
        self._consumers.append(consumer)

        logger.info(f"🔄 [Kafka消费] 开始消费 topic: {topic}, group_id={self.group_id}")

        msg_count = 0
        last_log_time = time.time()

        while self._running:
            try:
                # 批量拉取消息
                messages = consumer.poll(timeout_ms=1000, max_records=self.max_poll_records)

                if not messages:
                    continue

                # 记录拉取到的消息统计
                total_msgs = sum(len(msgs) for msgs in messages.values())
                msg_count += total_msgs

                # 每分钟输出一次消费统计
                if time.time() - last_log_time > 60:
                    logger.info(f"📊 [Kafka消费统计] topic={topic}, 本轮拉取={total_msgs}条, "
                                f"累计消费={msg_count}条")
                    last_log_time = time.time()

                logger.debug(f"📥 [Kafka消费] 拉取到 {total_msgs} 条消息, topic={topic}")

                # 并行处理消息
                futures = []
                for tp, msgs in messages.items():
                    logger.debug(f"📥 [Kafka消费] 处理 partition={tp.partition}, "
                                 f"offset范围={msgs[0].offset}-{msgs[-1].offset}, 数量={len(msgs)}")

                    for msg in msgs:
                        future = self._executor.submit(self._process_message, msg)
                        futures.append(future)

                # 等待当前批次处理完成
                success_count = 0
                fail_count = 0
                for future in futures:
                    try:
                        future.result(timeout=60)
                        success_count += 1
                    except Exception as e:
                        fail_count += 1
                        logger.error(f"❌ [Kafka消费] 处理消息失败: {e}")

                logger.info(f"📊 [Kafka消费] 批次处理完成: topic={topic}, 成功={success_count}, 失败={fail_count}")

                # 手动提交 offset
                consumer.commit()
                logger.debug(f"✅ [Kafka消费] offset已提交: topic={topic}")

            except Exception as e:
                logger.error(f"❌ [Kafka消费异常] topic={topic}, error={e}", exc_info=True)
                time.sleep(1)

        consumer.close()
        logger.info(f"🛑 [Kafka消费] 停止消费 topic: {topic}, 共消费 {msg_count} 条消息")

    def start(self):
        """启动消费者"""
        if self._running:
            logger.warning("消费者已在运行")
            return

        if not self._upsert_handler or not self._delete_handler:
            logger.error("请先注册事件处理器")
            return

        self._running = True

        # 为每个 topic 启动独立的消费线程
        topics = [self.topic_upsert, self.topic_delete]
        for topic in topics:
            thread = threading.Thread(
                target=self._consume_topic,
                args=(topic,),
                name=f"kafka-consumer-{topic}",
                daemon=True
            )
            thread.start()
            self._threads.append(thread)

        logger.info(f"Kafka 消费者已启动，监听 topics: {topics}")

    def stop(self):
        """停止消费者"""
        self._running = False

        for thread in self._threads:
            thread.join(timeout=10)

        self._executor.shutdown(wait=True)

        for consumer in self._consumers:
            try:
                consumer.close()
            except:
                pass

        logger.info("Kafka 消费者已停止")

    def get_consumer_lag(self, topic: str = None) -> Dict[str, int]:
        """获取消费延迟"""
        lag_info = {}

        for consumer in self._consumers:
            for assignment in consumer.assignment():
                topic = assignment.topic
                partition = assignment.partition

                # 获取最新的 offset
                end_offsets = consumer.end_offsets([assignment])
                current_offset = consumer.position(assignment)

                lag = end_offsets.get(assignment, 0) - current_offset
                lag_info[f"{topic}-{partition}"] = lag

        return lag_info


# 全局单例
_kafka_consumer = None


def get_kafka_consumer() -> Optional[KafkaChangeConsumer]:
    """获取 Kafka 消费者实例"""
    global _kafka_consumer
    if _kafka_consumer is None:
        try:
            _kafka_consumer = KafkaChangeConsumer()
        except Exception as e:
            logger.error(f"创建 Kafka 消费者失败: {e}")
            return None
    return _kafka_consumer


__all__ = ['KafkaChangeConsumer', 'get_kafka_consumer']