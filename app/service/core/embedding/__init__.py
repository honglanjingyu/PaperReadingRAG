"""
向量化服务模块
提供文本向量嵌入生成功能，支持远程API和本地模型
"""

from .base_embedding import BaseEmbeddingModel
from .remote_embedding import RemoteEmbeddingModel
from .local_embedding import LocalEmbeddingModel
from .embedding_manager import EmbeddingManager, EmbeddingType, get_embedding_manager
from .embedding_service import get_embedding_service

# 保持向后兼容的别名
def get_embedding_service():
    """获取EmbeddingService（向后兼容）"""
    return get_embedding_manager()

__all__ = [
    'BaseEmbeddingModel',
    'RemoteEmbeddingModel',
    'LocalEmbeddingModel',
    'EmbeddingManager',
    'EmbeddingType',
    'get_embedding_manager',
    'get_embedding_service'
]