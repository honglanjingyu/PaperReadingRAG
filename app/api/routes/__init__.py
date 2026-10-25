# app/api/routes/__init__.py
"""API路由模块"""

"""API 路由模块：各 router 的唯一导出入口"""

from app.api.routes import health, chat
from app.api.routes.upload import router as upload_router
from app.api.routes.delete import router as delete_router
from app.api.routes.graph_rag import router as graph_rag_router
from app.api.auth_routes import router as auth_router

__all__ = ['health', 'chat', 'upload_router', 'delete_router', 'graph_rag_router', 'auth_router']