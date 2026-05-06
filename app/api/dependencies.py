# app/api/dependencies.py
"""
依赖注入模块
"""

from app.api.services.document_service import DocumentService
from app.api.services.chat_service import ChatService

# 服务单例
_document_service = None
_chat_service = None


def get_document_service() -> DocumentService:
    """获取文档服务实例"""
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service


def get_chat_service() -> ChatService:
    """获取聊天服务实例"""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service


__all__ = ['get_document_service', 'get_chat_service']