# app/api/main.py
"""
FastAPI应用创建 - 优化异步配置
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import logging
from contextlib import asynccontextmanager
import asyncio

from app.api.routes import health, chat
from app.api.routes.upload import router as upload_router
from app.api.routes.delete import router as delete_router
from app.api.routes.graph_rag import router as graph_rag_router
from app.api.config import settings
from app.api.auth_routes import router as auth_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理 - 优化异步处理"""
    logger.info("应用启动中...")

    # 初始化异步文档处理器
    from app.service.core.rag.async_processor import init_async_processor, shutdown_async_processor
    await init_async_processor()
    logger.info("异步文档处理器已启动")

    yield

    logger.info("应用关闭中...")
    await shutdown_async_processor()
    logger.info("异步文档处理器已关闭")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG文档问答系统",
        description="支持文档上传、智能分块、向量检索和智能问答",
        version="1.0.0",
        lifespan=lifespan
    )

    # CORS 配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(health.router, prefix="/api", tags=["健康检查"])
    app.include_router(upload_router, prefix="/api", tags=["文档上传"])
    app.include_router(delete_router, prefix="/api", tags=["文档删除"])
    app.include_router(chat.router, prefix="/api", tags=["智能问答"])
    app.include_router(auth_router)
    app.include_router(graph_rag_router, prefix="/api", tags=["GraphRAG管理"])

    logger.info("FastAPI 应用创建完成")
    return app


def configure_static_routes(app: FastAPI):
    """配置静态文件路由"""
    web_dir = Path(__file__).parent.parent / "web"

    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        css_dir = web_dir / "css"
        if css_dir.exists():
            app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")

        js_dir = web_dir / "js"
        if js_dir.exists():
            app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")

        @app.get("/")
        async def root():
            return FileResponse(str(web_dir / "chat.html"))

        @app.get("/upload")
        async def upload_page():
            return FileResponse(str(web_dir / "upload.html"))

        @app.get("/chat")
        async def chat_page():
            return FileResponse(str(web_dir / "chat.html"))

        @app.get("/login")
        async def login_page():
            return FileResponse(str(web_dir / "login.html"))

        @app.get("/login.html")
        async def login_html():
            return FileResponse(str(web_dir / "login.html"))

        @app.get("/graph")
        async def graph_page():
            return FileResponse(str(web_dir / "graph.html"))


app = create_app()
configure_static_routes(app)

__all__ = ['app']