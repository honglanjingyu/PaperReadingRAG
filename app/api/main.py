# app/api/main.py
"""
FastAPI应用创建
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.api.routes import health, upload, chat
from app.api.config import settings
from app.api.services import init_logging


def create_app() -> FastAPI:
    """创建FastAPI应用"""
    # 初始化日志系统
    init_logging()

    app = FastAPI(
        title="RAG文档问答系统",
        description="支持文档上传、智能分块、向量检索和智能问答",
        version="1.0.0"
    )

    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(health.router, prefix="/api", tags=["健康检查"])
    app.include_router(upload.router, prefix="/api", tags=["文档上传"])
    app.include_router(chat.router, prefix="/api", tags=["智能问答"])

    return app


def configure_static_routes(app: FastAPI):
    """配置静态文件路由"""
    web_dir = Path(__file__).parent.parent / "web"

    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/")
        async def root():
            return FileResponse(str(web_dir / "upload.html"))

        @app.get("/upload")
        async def upload_page():
            return FileResponse(str(web_dir / "upload.html"))

        @app.get("/chat")
        async def chat_page():
            return FileResponse(str(web_dir / "chat.html"))


# 创建应用实例
app = create_app()
configure_static_routes(app)

__all__ = ['app']