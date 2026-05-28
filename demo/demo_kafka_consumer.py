#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 Kafka 消费者是否真正工作"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()


async def demo_kafka_consumer():
    print("=" * 60)
    print("测试 Kafka 消费者")
    print("=" * 60)

    from app.service.core.streaming import get_stream_processor

    processor = get_stream_processor()
    print(f"\n1. StreamProcessor 状态:")
    print(f"   enabled: {processor.enabled}")
    print(f"   producer: {processor._producer is not None}")
    print(f"   consumer: {processor._consumer is not None}")

    # 2. 启动消费者
    print("\n2. 启动 Kafka 消费者...")
    processor.start_consumer()
    print("   消费者已启动")

    # 3. 发送测试消息
    print("\n3. 发送测试 Upsert 事件...")
    result = await processor.emit_change_event(
        filename="test_kafka_doc.txt",
        content="这是一个测试文档，用于验证Kafka流式处理是否正常工作。",
        user_level="normal",
        file_path=None,
        event_type="upsert"
    )
    print(f"   发送结果: {result}")

    # 4. 等待消费者处理
    print("\n4. 等待消费者处理（5秒）...")
    await asyncio.sleep(5)

    # 5. 检查是否有消息被消费
    print("\n5. 检查消费者状态:")
    if processor._consumer:
        status = processor.get_status()
        print(f"   状态: {status}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("请检查 logs/rag_*.log 文件查看消费者是否处理了消息")
    print("=" * 60)


async def demo_document_upload_with_kafka():
    """测试文档上传是否会触发 Kafka 事件"""
    print("\n" + "=" * 60)
    print("测试文档上传触发 Kafka 事件")
    print("=" * 60)

    from app.service.core.streaming import get_stream_processor
    import time

    processor = get_stream_processor()

    # 模拟文档上传
    test_filename = f"test_upload_{int(time.time())}.txt"
    test_content = "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。"

    print(f"\n1. 发送文档上传事件: {test_filename}")
    result = await processor.emit_change_event(
        filename=test_filename,
        content=test_content,
        user_level="normal",
        file_path=None,
        event_type="upsert"
    )
    print(f"   发送结果: {result}")

    if result:
        print("   ✅ 事件已发送到 Kafka")
    else:
        print("   ❌ 事件发送失败")

    # 等待处理
    print("\n2. 等待 Kafka 消费者处理（3秒）...")
    await asyncio.sleep(3)

    print("\n3. 检查是否处理完成:")
    print("   请查看日志中的 '处理 upsert 事件' 消息")
    print(f"   文件: {test_filename}")
    print(f"   内容长度: {len(test_content)}")


def check_logs():
    """检查日志文件中是否有 Kafka 相关的处理记录"""
    import os
    from pathlib import Path

    print("\n" + "=" * 60)
    print("检查日志文件")
    print("=" * 60)

    log_dir = Path(__file__).parent.parent / "logs"
    if not log_dir.exists():
        print("日志目录不存在，请先运行服务产生日志")
        return

    log_files = list(log_dir.glob("rag_*.log"))
    if not log_files:
        print("未找到日志文件")
        return

    latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
    print(f"\n最新日志: {latest_log}")

    # 搜索 Kafka 相关日志
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            content = f.read()

            kafka_lines = []
            for line in content.split('\n'):
                if 'Kafka' in line or 'kafka' in line or '流式' in line or 'upsert' in line:
                    kafka_lines.append(line)

            if kafka_lines:
                print(f"\n找到 {len(kafka_lines)} 条 Kafka 相关日志:")
                for line in kafka_lines[-10:]:  # 显示最后10条
                    print(f"  {line[:150]}")
            else:
                print("\n⚠️ 未找到 Kafka 相关日志")
                print("可能原因:")
                print("1. 消费者未真正启动")
                print("2. 处理函数未正确注册")
                print("3. 日志级别太高")
    except Exception as e:
        print(f"读取日志失败: {e}")


async def main():
    # 1. 先检查日志
    check_logs()

    # 2. 测试基本功能
    await demo_kafka_consumer()

    # 3. 测试文档上传事件
    await demo_document_upload_with_kafka()

    # 4. 再次检查日志
    await asyncio.sleep(2)
    check_logs()


if __name__ == "__main__":
    asyncio.run(main())