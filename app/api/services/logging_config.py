# app/api/services/logging_config.py
"""
日志配置模块 - 只保留简洁的控制台输出和文件日志
"""

import os
import logging
import sys
from pathlib import Path
from datetime import datetime

# 创建 logs 目录
LOGS_DIR = Path(__file__).parent.parent.parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# 日志文件名（按日期分割）
log_filename = LOGS_DIR / f"rag_{datetime.now().strftime('%Y%m%d')}.log"


def setup_file_logging():
    """配置文件日志 - 记录详细日志到文件"""
    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # 文件日志格式（包含详细信息）
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # 配置各个模块的日志到文件
    loggers_to_configure = [
        'app.service.core.deepdoc',
        'app.service.core.embedding',
        'app.service.core.retrieval',
        'app.service.core.rag',
        'app.api.services',
    ]

    for logger_name in loggers_to_configure:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        # 移除已有的处理器，避免重复
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.addHandler(file_handler)
        logger.propagate = False  # 不传播到根日志器，避免控制台重复输出

    # 配置根日志器（只输出到文件，不输出到控制台）
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # 移除所有默认处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)


def setup_console_logging():
    """
    配置控制台日志 - 只输出错误级别以上，避免干扰 uvicorn 访问日志
    uvicorn 的访问日志格式: INFO:     127.0.0.1:33396 - "GET /api/health HTTP/1.1" 200 OK
    """
    # 创建控制台处理器（只输出 WARNING 及以上级别）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)

    # 简洁格式（只显示消息）
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)

    # 配置根日志器（控制台只输出警告和错误）
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    # 检查是否已有控制台处理器
    has_console = any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
                      for h in root_logger.handlers)
    if not has_console:
        root_logger.addHandler(console_handler)


def suppress_noisy_loggers():
    """压制部分第三方库的日志输出"""
    noisy_loggers = [
        'elasticsearch',
        'urllib3.connectionpool',
        'requests.packages.urllib3',
        'openai',
        'httpcore',
        'httpx',
        'pymilvus',  # 添加 Milvus 的日志压制
        'sentence_transformers',
    ]

    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False


def init_logging():
    """初始化日志系统 - 只保留文件详细日志，控制台保持干净"""
    setup_file_logging()
    setup_console_logging()
    suppress_noisy_loggers()

    # 打印日志文件位置
    print(f"📝 详细日志文件: {log_filename}")


__all__ = ['init_logging', 'LOGS_DIR']