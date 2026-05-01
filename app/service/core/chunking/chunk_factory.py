# app/service/core/chunking/chunk_factory.py
"""
分块工厂 - 提供便捷的分块接口
"""

from typing import List, Dict, Any, Optional
from .chunk_manager import ChunkManager, Chunk
from .recursive_chunker import RecursiveChunkerSimple
import hashlib


def create_chunker(chunk_size: int = 256, strategy: str = 'recursive'):
    """创建分块器"""
    if strategy == 'recursive':
        return RecursiveChunkerSimple(chunk_token_num=chunk_size)
    else:
        return ChunkManager({
            'chunk_token_num': chunk_size,
            'strategy': strategy
        })


def chunk_text_to_chunks(
    text: str,
    chunk_size: int = 256,
    metadata: Optional[Dict] = None,
    strategy: str = 'recursive'
) -> List:
    """
    将文本分块并返回 Chunk 对象列表

    Args:
        text: 输入文本
        chunk_size: 分块大小
        metadata: 元数据
        strategy: 分块策略

    Returns:
        Chunk 对象列表
    """
    if not text:
        return []

    if strategy == 'recursive':
        chunker = RecursiveChunkerSimple(chunk_token_num=chunk_size)
        chunk_texts = chunker.chunk(text)

        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            if chunk_text.strip():
                chunk_id = hashlib.md5(f"{i}_{chunk_text[:100]}".encode()).hexdigest()[:16]
                from .chunk_strategies import Chunk
                chunks.append(Chunk(
                    id=f"chunk_{i}_{chunk_id}",
                    content=chunk_text,
                    metadata=metadata or {},
                    start_idx=0,
                    end_idx=len(chunk_text),
                    token_count=chunker._count_tokens(chunk_text)
                ))
        return chunks
    else:
        manager = ChunkManager({
            'chunk_token_num': chunk_size,
            'strategy': strategy
        })
        return manager.chunk_text(text, metadata)


def chunk_text_simple(text: str, chunk_size: int = 256) -> List[str]:
    """简单分块，只返回文本列表"""
    chunker = RecursiveChunkerSimple(chunk_token_num=chunk_size)
    return chunker.chunk(text)


def get_chunk_statistics(chunks: List) -> Dict[str, Any]:
    """获取分块统计信息"""
    if not chunks:
        return {
            'total_chunks': 0,
            'avg_token_count': 0,
            'min_token_count': 0,
            'max_token_count': 0,
            'total_tokens': 0
        }

    token_counts = [c.token_count for c in chunks]

    return {
        'total_chunks': len(chunks),
        'avg_token_count': sum(token_counts) / len(token_counts),
        'min_token_count': min(token_counts),
        'max_token_count': max(token_counts),
        'total_tokens': sum(token_counts)
    }


__all__ = [
    'create_chunker',
    'chunk_text_to_chunks',
    'chunk_text_simple',
    'get_chunk_statistics'
]