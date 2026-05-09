# run_api.py
"""
RAG系统FastAPI入口文件
启动文件: python run_api.py
访问地址: http://localhost:8001
"""

import uvicorn
import signal
import sys
import logging
from dotenv import load_dotenv

from app.a2a.server import integrate_a2a_to_app
from app.a2a.server import run_a2a_server as run_standalone_a2a

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
    import argparse
    parser = argparse.ArgumentParser(description="PaperReadingRAG 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8001, help="API 端口")
    parser.add_argument("--a2a-port", type=int, default=8004, help="A2A 服务端口")
    parser.add_argument("--a2a-only", action="store_true", help="仅启动 A2A 服务")
    parser.add_argument("--reload", action="store_true", help="热重载模式")
    args = parser.parse_args()

    if args.a2a_only:
        # 只启动 A2A 服务
        print("=" * 60)
        print("PaperReadingRAG - A2A 服务模式")
        print("=" * 60)
        print(f"A2A 地址: http://{args.host}:{args.a2a_port}")
        print("=" * 60)

        import asyncio
        asyncio.run(run_standalone_a2a(args.host, args.a2a_port))
        return

    # 正常启动 API 服务（包含 A2A 路由）
    print("=" * 60)
    print("PaperReadingRAG 服务启动")
    print("=" * 60)
    print(f"API 地址: http://{args.host}:{args.port}")
    print(f"API 文档: http://{args.host}:{args.port}/docs")
    print(f"A2A 地址: http://{args.host}:{args.port}/a2a")
    print("=" * 60)

    from app.api.main import app

    # 集成 A2A 路由
    integrate_a2a_to_app(app)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()