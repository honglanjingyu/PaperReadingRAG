#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kafka 诊断脚本"""

import os
import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 加载环境变量
from dotenv import load_dotenv

load_dotenv()


def diagnose():
    print("=" * 70)
    print("Kafka 诊断报告")
    print("=" * 70)

    # 1. 检查环境变量
    print("\n1. 环境变量检查:")
    kafka_enabled = os.getenv("ENABLE_KAFKA_STREAMING", "false")
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "not set")
    print(f"   ENABLE_KAFKA_STREAMING = {kafka_enabled}")
    print(f"   KAFKA_BOOTSTRAP_SERVERS = {bootstrap_servers}")

    if kafka_enabled != "true":
        print("   ❌ ENABLE_KAFKA_STREAMING 未设置为 true")
        return

    # 2. 检查 kafka-python 模块
    print("\n2. Kafka-Python 模块检查:")
    try:
        import kafka
        print(f"   ✅ kafka-python 已安装")
        print(f"   版本: {getattr(kafka, '__version__', 'unknown')}")
        print(f"   路径: {kafka.__file__}")
    except ImportError as e:
        print(f"   ❌ kafka-python 未安装: {e}")
        print("   解决方案: pip install kafka-python")
        return

    # 3. 测试 Kafka 连接
    print("\n3. Kafka 连接测试:")
    try:
        from kafka import KafkaProducer
        from kafka.errors import NoBrokersAvailable

        servers = bootstrap_servers.split(',')
        print(f"   尝试连接: {servers}")

        # 尝试创建生产者（快速测试）
        producer = KafkaProducer(
            bootstrap_servers=servers,
            request_timeout_ms=5000,
            max_block_ms=5000,
            api_version_auto_timeout_ms=5000
        )
        print("   ✅ Kafka 连接成功！")
        producer.close()

    except NoBrokersAvailable:
        print("   ❌ 无法连接到 Kafka brokers")
        print("   请检查:")
        print(f"     1. Kafka 服务是否在 {bootstrap_servers} 运行")
        print("     2. 防火墙是否允许 9092 端口")
        print("     3. 使用命令检查: nc -zv 172.20.48.1 9092")
    except Exception as e:
        print(f"   ❌ 连接失败: {type(e).__name__}: {e}")

    # 4. 测试 is_kafka_enabled
    print("\n4. is_kafka_enabled() 测试:")
    try:
        from app.service.core.streaming import is_kafka_enabled
        result = is_kafka_enabled()
        print(f"   is_kafka_enabled() = {result}")
        if result:
            print("   ✅ Kafka 流式处理已启用")
        else:
            print("   ❌ Kafka 流式处理未启用")
    except Exception as e:
        print(f"   ❌ 导入失败: {e}")

    # 5. 测试获取 StreamProcessor
    print("\n5. StreamProcessor 初始化:")
    try:
        from app.service.core.streaming import get_stream_processor
        processor = get_stream_processor()
        print(f"   enabled: {processor.enabled}")
        print(f"   producer: {processor._producer is not None}")
        print(f"   consumer: {processor._consumer is not None}")

        if processor._producer:
            print("   ✅ 生产者已初始化")
        if processor._consumer:
            print("   ✅ 消费者已初始化")

    except Exception as e:
        print(f"   ❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()

    # 6. 测试创建 topic
    print("\n6. Topic 创建测试:")
    try:
        from kafka.admin import KafkaAdminClient, NewTopic

        admin = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers.split(','),
            client_id='diagnose'
        )

        topics = admin.list_topics()
        print(f"   现有 topics: {list(topics.keys())}")

        # 检查必需的 topics
        required_topics = [
            'rag-document-upsert',
            'rag-document-delete',
            'rag-document-changes'
        ]

        for topic in required_topics:
            if topic in topics:
                print(f"   ✅ {topic} 已存在")
            else:
                print(f"   ⚠️  {topic} 不存在（会自动创建）")

        admin.close()

    except Exception as e:
        print(f"   ⚠️ 无法获取 topic 列表: {e}")

    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)


if __name__ == "__main__":
    diagnose()