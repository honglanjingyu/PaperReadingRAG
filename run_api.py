# run_api.py
"""
RAG系统FastAPI入口文件
启动文件: python run_api.py
访问地址: http://localhost:8000
"""

import uvicorn
import os
import signal
import sys
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置根日志器级别为 WARNING，减少控制台输出
logging.basicConfig(level=logging.WARNING)


# ============================================================
# 信号处理 - 解决PyCharm停止问题
# ============================================================

def signal_handler(signum, frame):
    """处理中断信号"""
    print("\n\n收到停止信号，正在关闭服务...")
    sys.exit(0)


# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================
# 启动入口
# ============================================================

def main():
    """启动服务"""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"

    print("=" * 60)
    print("RAG系统API服务启动")
    print("=" * 60)
    print(f"访问地址: http://{host}:{port}")
    print(f"API文档: http://{host}:{port}/docs")
    print(f"上传页面: http://{host}:{port}/upload")
    print(f"聊天页面: http://{host}:{port}/chat")
    print("=" * 60)
    print("提示：按 Ctrl+C 停止服务")
    print("=" * 60)

    # 配置 uvicorn 日志级别
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = '%(asctime)s - %(levelname)s - %(message)s'
    log_config["loggers"]["uvicorn"]["level"] = "INFO"
    log_config["loggers"]["uvicorn.access"]["level"] = "INFO"

    # 导入并运行应用
    from app.api.main import app

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        loop="asyncio",
        workers=1,
        access_log=True  # 保留访问日志
    )


if __name__ == "__main__":
    main()