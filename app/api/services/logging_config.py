# app/api/services/logging_config.py
"""
统一日志配置模块 - 输出详细日志到文件，控制台只保留重要信息
"""

import os
import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

# 创建 logs 目录
LOGS_DIR = Path(__file__).parent.parent.parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# 日志文件名（按日期分割）
log_filename = LOGS_DIR / f"rag_{datetime.now().strftime('%Y%m%d')}.log"
error_log_filename = LOGS_DIR / f"rag_error_{datetime.now().strftime('%Y%m%d')}.log"


def setup_file_logging():
    """配置文件日志 - 记录详细日志到文件（带轮转）"""

    # 主日志文件处理器（记录所有级别）
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)

    # 详细日志格式
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # 错误日志文件处理器（只记录 ERROR 及以上）
    error_handler = RotatingFileHandler(
        error_log_filename,
        maxBytes=20 * 1024 * 1024,  # 20MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)

    # 配置各个模块的日志到文件
    loggers_to_configure = [
        'app',
        'app.service',
        'app.service.core',
        'app.service.core.deepdoc',
        'app.service.core.embedding',
        'app.service.core.retrieval',
        'app.service.core.rag',
        'app.service.core.vector_store',
        'app.service.core.streaming',  # 添加 streaming 模块
        'app.api',
        'app.api.routes',
        'app.api.services',
        'app.db',
        'app.auth',
        'app.a2a',
        'urllib3',
        'elasticsearch',
        'httpx',
        'httpcore',
        'pymilvus',
        'jieba',
    ]

    for logger_name in loggers_to_configure:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        # 避免重复添加处理器
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_filename) for h in logger.handlers):
            logger.addHandler(file_handler)
            logger.addHandler(error_handler)
        logger.propagate = False  # 不传播到根日志器

    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not any(
            isinstance(h, logging.FileHandler) and h.baseFilename == str(log_filename) for h in root_logger.handlers):
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)


def setup_console_logging():
    """
    配置控制台日志 - 只输出 WARNING 及以上级别，过滤 Kafka 详细日志
    """
    # 创建控制台处理器 - 只输出 WARNING 及以上（改为 WARNING 级别）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # 改为 WARNING 级别，过滤 INFO 和 DEBUG

    # 简洁格式
    console_formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    console_handler.setFormatter(console_formatter)

    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)  # 根日志器也设置为 WARNING

    # 检查是否已有控制台处理器
    has_console = any(
        isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
        for h in root_logger.handlers
    )
    if not has_console:
        root_logger.addHandler(console_handler)

    # 为 uvicorn 的访问日志单独配置控制台输出（保持简洁）
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.propagate = True
    uvicorn_access_logger.setLevel(logging.WARNING)  # 改为 WARNING


def suppress_noisy_loggers():
    """压制第三方库的日志输出（包括 Kafka）"""
    noisy_loggers = [
        'urllib3.connectionpool',
        'requests.packages.urllib3',
        'httpcore.connection',
        'httpcore.http11',
        'elastic_transport.transport',
        'elasticsearch',
        'neo4j',
        'neo4j.notifications',
        # ========== 添加 Kafka 相关日志器 ==========
        'kafka',
        'kafka.conn',
        'kafka.client',
        'kafka.consumer',
        'kafka.consumer.subscription_state',
        'kafka.coordinator',
        'kafka.coordinator.consumer',
        'kafka.cluster',
        'kafka.producer',
        'kafka.producer.sender',
    ]

    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)  # 只显示 ERROR 及以上
        # 确保不传播到根日志器
        logger.propagate = False

    # 单独设置 neo4j 通知
    neo4j_notifications = logging.getLogger("neo4j.notifications")
    neo4j_notifications.setLevel(logging.ERROR)

    # 额外设置 kafka 相关日志器，防止它们在控制台输出
    for name in logging.root.manager.loggerDict:
        if 'kafka' in name.lower():
            logger = logging.getLogger(name)
            logger.setLevel(logging.ERROR)
            logger.propagate = False


def init_logging():
    """初始化日志系统"""
    # 先设置控制台日志（在文件日志之前，确保控制台级别正确）
    setup_console_logging()
    setup_file_logging()
    suppress_noisy_loggers()

    # 记录启动日志到文件
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("日志系统初始化完成")
    logger.info(f"日志目录: {LOGS_DIR}")
    logger.info("=" * 60)


__all__ = ['init_logging', 'LOGS_DIR']