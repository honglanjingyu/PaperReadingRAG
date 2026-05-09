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
from app.api.auth_routes import router as auth_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG文档问答系统",
        description="支持文档上传、智能分块、向量检索和智能问答",
        version="1.0.0"
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
    app.include_router(upload.router, prefix="/api", tags=["文档上传"])
    app.include_router(chat.router, prefix="/api", tags=["智能问答"])
    app.include_router(auth_router)  # 添加认证路由

    return app


# app/api/main.py
# 在 configure_static_routes 函数中确保 CSS 目录被正确挂载

def configure_static_routes(app: FastAPI):
    """配置静态文件路由"""
    web_dir = Path(__file__).parent.parent / "web"

    if web_dir.exists():
        # 挂载静态文件目录
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        # 挂载 CSS 目录（直接从 web/css 提供）
        css_dir = web_dir / "css"
        if css_dir.exists():
            app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")

        # 挂载 JS 目录
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


# 创建应用实例
app = create_app()
configure_static_routes(app)

__all__ = ['app']