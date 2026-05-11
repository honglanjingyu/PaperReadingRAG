# app/api/routes/__init__.py
"""API路由模块"""

from app.api.routes import health, upload, chat, upload_batch
from app.api.routes.delete import router as delete_router
from app.api.routes.delete_batch import router as delete_batch_router

__all__ = ['health', 'upload', 'chat', 'upload_batch', 'delete_router', 'delete_batch_router']