# app/service/core/chunking/__init__.py
"""
分块模块 - 将长文档按语义或固定长度切分成小块
"""
from .chunk_manager import ChunkManager
from .chunk_strategies import (
    FixedTokenChunker,
    SemanticChunker,
    RecursiveChunker,
    SentenceChunker,
    ParagraphChunker,
    Chunk,
    ChunkStrategy,
    BaseChunker
)
from .chunk_processor import ChunkProcessor
from .recursive_chunker import RecursiveChunkerSimple
from .chunk_factory import (
    create_chunker,
    chunk_text_to_chunks,
    chunk_text_simple,
    get_chunk_statistics
)

__all__ = [
    'ChunkManager',
    'FixedTokenChunker',
    'SemanticChunker',
    'RecursiveChunker',
    'SentenceChunker',
    'ParagraphChunker',
    'Chunk',
    'ChunkStrategy',
    'BaseChunker',
    'ChunkProcessor',
    'RecursiveChunkerSimple',
    'create_chunker',
    'chunk_text_to_chunks',
    'chunk_text_simple',
    'get_chunk_statistics'
]