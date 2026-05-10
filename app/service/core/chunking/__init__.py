# app/service/core/chunking/__init__.py

"""分块模块 - 将长文档按语义或固定长度切分成小块"""

from .chunk_types import Chunk
from .chunk_manager import ChunkManager
from .chunk_strategies import (
    FixedTokenChunker,
    SemanticChunker,
    RecursiveChunker,
    SentenceChunker,
    ParagraphChunker,
    ChunkStrategy,
    BaseChunker
)
from .chunk_processor import ChunkProcessor
from .chunk_factory import (
    create_chunker,
    chunk_text_to_chunks,
    chunk_text_simple,
    get_chunk_statistics
)

# 向后兼容别名
RecursiveChunkerSimple = RecursiveChunker

__all__ = [
    'Chunk',
    'ChunkManager',
    'FixedTokenChunker',
    'SemanticChunker',
    'RecursiveChunker',
    'RecursiveChunkerSimple',
    'SentenceChunker',
    'ParagraphChunker',
    'ChunkStrategy',
    'BaseChunker',
    'ChunkProcessor',
    'create_chunker',
    'chunk_text_to_chunks',
    'chunk_text_simple',
    'get_chunk_statistics'
]