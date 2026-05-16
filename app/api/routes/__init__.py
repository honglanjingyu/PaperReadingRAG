# app/api/routes/__init__.py
"""API路由模块"""

from app.api.routes import health, chat
from app.api.routes.upload import router as upload_router
from app.api.routes.delete import router as delete_router
from app.api.routes.graph_rag import router as graph_rag_router

__all__ = ['health', 'chat', 'upload_router', 'delete_router', 'graph_rag_router']