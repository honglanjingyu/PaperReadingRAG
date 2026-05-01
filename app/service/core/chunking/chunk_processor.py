"""
分块处理器 - 整合清洗和分块流程
"""

from typing import List, Dict, Any, Optional, Tuple
from .chunk_manager import ChunkManager, Chunk
from service.core.cleaner.data_cleaner import DataCleaner, NoiseFilter


class ChunkProcessor:
    """
    分块处理器 - 整合数据清洗和分块流程
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化处理器

        Args:
            config: 配置参数
                - clean: 是否启用清洗 (default: True)
                - chunk_token_num: 块大小 (default: 128)
                - strategy: 分块策略 (default: 'recursive')
                - remove_noise: 是否移除噪声 (default: True)
        """
        self.config = {
            'clean': True,
            'chunk_token_num': 128,
            'strategy': 'recursive',
            'remove_noise': True,
            **(config or {})
        }

        # 初始化清洗器
        self.cleaner = DataCleaner({
            'remove_empty_lines': True,
            'normalize_whitespace': True,
            'min_line_length': 2,
            'remove_special_chars': False
        })

        self.noise_filter = NoiseFilter()

        # 初始化分块管理器
        self.chunk_manager = ChunkManager({
            'chunk_token_num': self.config['chunk_token_num'],
            'strategy': self.config['strategy']
        })

    def process_text(self, text: str,
                     metadata: Optional[Dict] = None,
                     enable_clean: bool = True) -> List[Chunk]:
        """
        处理单个文本：清洗 -> 分块

        Args:
            text: 输入文本
            metadata: 元数据
            enable_clean: 是否启用清洗

        Returns:
            分块列表
        """
        if not text:
            return []

        # 1. 清洗文本
        if enable_clean and self.config['clean']:
            text = self.cleaner.clean_text(text)

        # 2. 分块
        chunks = self.chunk_manager.chunk_text(text, metadata)

        # 3. 过滤噪声块
        if self.config['remove_noise']:
            chunks = [c for c in chunks if not self.noise_filter.is_noise_line(c.content)]

        return chunks

    def process_sections(self, sections: List[tuple],
                         metadata: Optional[Dict] = None,
                         enable_clean: bool = True) -> List[Chunk]:
        """
        处理段落列表：清洗 -> 分块

        Args:
            sections: 段落列表 [(text, style), ...]
            metadata: 元数据
            enable_clean: 是否启用清洗

        Returns:
            分块列表
        """
        if not sections:
            return []

        # 1. 提取并清洗文本
        texts = []
        for text, style in sections:
            if text and text.strip():
                if enable_clean and self.config['clean']:
                    text = self.cleaner.clean_text(text)
                if text:
                    texts.append(text)

        full_text = "\n".join(texts)

        # 2. 分块
        return self.chunk_manager.chunk_text(full_text, metadata)

    def process_document(self, document: Dict[str, Any],
                         enable_clean: bool = True) -> List[Chunk]:
        """
        处理文档对象：清洗 -> 分块

        Args:
            document: 文档字典
            enable_clean: 是否启用清洗

        Returns:
            分块列表
        """
        content = document.get('content_with_weight', '')
        if not content:
            return []

        # 1. 清洗内容
        if enable_clean and self.config['clean']:
            content = self.cleaner.clean_text(content)
            document['content_with_weight'] = content

        # 2. 分块
        return self.chunk_manager.chunk_document(document)

    def process_with_original_merge(self, sections: List[tuple],
                                    metadata: Optional[Dict] = None) -> Tuple[List[str], List[Chunk]]:
        """
        使用项目原有的 naive_merge 进行分块，同时生成 Chunk 对象

        Args:
            sections: 段落列表
            metadata: 元数据

        Returns:
            (chunk_texts, chunk_objects): 分块文本列表和 Chunk 对象列表
        """
        # 使用原有的 naive_merge
        chunk_texts = self.chunk_manager.naive_merge(sections)

        # 转换为 Chunk 对象
        chunks = []
        for i, text in enumerate(chunk_texts):
            if text.strip():
                chunk = Chunk(
                    id=f"chunk_{i}",
                    content=text,
                    metadata=metadata or {},
                    start_idx=0,
                    end_idx=len(text),
                    token_count=self.chunk_manager.chunker._count_tokens(text)
                )
                chunks.append(chunk)

        return chunk_texts, chunks


# 便捷函数
def chunk_document(document: Dict[str, Any],
                   chunk_size: int = 128,
                   strategy: str = 'recursive') -> List[Chunk]:
    """
    快速分块文档

    Args:
        document: 文档字典
        chunk_size: 块大小
        strategy: 分块策略

    Returns:
        分块列表
    """
    processor = ChunkProcessor({
        'chunk_token_num': chunk_size,
        'strategy': strategy
    })
    return processor.process_document(document)


def chunk_text(text: str,
               chunk_size: int = 128,
               strategy: str = 'recursive') -> List[Chunk]:
    """
    快速分块文本

    Args:
        text: 输入文本
        chunk_size: 块大小
        strategy: 分块策略

    Returns:
        分块列表
    """
    processor = ChunkProcessor({
        'chunk_token_num': chunk_size,
        'strategy': strategy
    })
    return processor.process_text(text)