# app/service/core/chunking/__init__.py (更新后 - 删除原有递归分块导出，保留父子分块为主)

"""
分块模块 - 主要使用父子分块策略
"""

# 父子分块（主推）
from .parent_child_chunker import (
    ParentChildChunker,
    ParentChildDocument,
    ParentChunk,
    ChildChunk,
    chunk_document_parent_child
)

# 保留其他分块策略作为备选
from .chunk_types import Chunk
from .chunk_manager import ChunkManager
from .chunk_processor import ChunkProcessor
from .chunk_factory import (
    chunk_text_simple,
    get_chunk_statistics
)

__all__ = [
    # 父子分块（主要）
    'ParentChildChunker',
    'ParentChildDocument',
    'ParentChunk',
    'ChildChunk',
    'chunk_document_parent_child',
    # 基础类型
    'Chunk',
    'ChunkManager',
    'ChunkProcessor',
    'chunk_text_simple',
    'get_chunk_statistics'
]