# app/api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import logging
from contextlib import asynccontextmanager

from app.api.routes import health, chat
from app.api.routes.upload import router as upload_router
from app.api.routes.delete import router as delete_router
from app.api.routes.graph_rag import router as graph_rag_router
from app.api.config import settings
from app.api.auth_routes import router as auth_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("应用启动中...")

    # ========== 移除异步文档处理器初始化 ==========
    # 不再需要 AsyncDocumentProcessor

    # ========== 启动 Kafka 流式消费者（唯一处理器） ==========
    from app.service.core.streaming import get_stream_processor
    stream_processor = get_stream_processor()
    stream_processor.start_consumer()
    logger.info("Kafka 流式消费者已启动（唯一文档处理器）")

    yield

    logger.info("应用关闭中...")
    # 停止流式消费者
    stream_processor.stop_consumer()
    logger.info("Kafka 流式消费者已停止")


def configure_uvicorn_logging():
    """配置 uvicorn 日志，减少控制台输出"""
    import logging

    uvicorn_loggers = [
        'uvicorn',
        'uvicorn.error',
        'uvicorn.access',
        'uvicorn.asgi',
        'uvicorn.lifespan'
    ]

    for logger_name in uvicorn_loggers:
        log = logging.getLogger(logger_name)
        log.setLevel(logging.WARNING)
        log.propagate = False


def create_app() -> FastAPI:
    configure_uvicorn_logging()

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