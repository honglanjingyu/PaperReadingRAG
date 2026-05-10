# run_api.py
import uvicorn
import signal
import sys
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置根日志器 - 只保留 uvicorn 日志
logging.basicConfig(level=logging.WARNING)

# 抑制第三方库的详细日志
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# 导入app
from app.api.main import app
from app.a2a.server import integrate_a2a_to_app


def signal_handler(signum, frame):
    """处理中断信号"""
    print("\n\n收到停止信号，正在关闭服务...")
    sys.exit(0)


# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


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
        print(f"Agent Discovery: http://{args.host}:{args.a2a_port}/.well-known/agent.json")
        print("=" * 60)

        import asyncio
        from app.a2a.server import run_a2a_server
        asyncio.run(run_a2a_server(args.host, args.a2a_port))
        return

    # 正常启动 API 服务（包含 A2A 路由）
    print("=" * 60)
    print("PaperReadingRAG 服务启动")
    print("=" * 60)
    print(f"API 地址: http://{args.host}:{args.port}")
    print(f"API 文档: http://{args.host}:{args.port}/docs")
    print(f"A2A 地址: http://{args.host}:{args.port}/a2a")
    print(f"Agent Discovery: http://{args.host}:{args.port}/.well-known/agent.json")
    print("=" * 60)

    # 集成 A2A 路由（包含发现端点）
    integrate_a2a_to_app(app)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        access_log=True,  # 启用访问日志
    )


if __name__ == "__main__":
    main()