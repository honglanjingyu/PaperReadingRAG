"""
分块模块 - 父子分块策略
"""

from .types import (
    Chunk,
    ChildChunk,
    ParentChunk,
    ParentChildDocument,
)
from .splitter import (
    ParentChildSplitter,
    chunk_document_parent_child,
    chunk_text_simple,
)

__all__ = [
    # 类型
    'Chunk',
    'ChildChunk',
    'ParentChunk',
    'ParentChildDocument',
    # 分块器
    'ParentChildSplitter',
    'chunk_document_parent_child',
    'chunk_text_simple',
]