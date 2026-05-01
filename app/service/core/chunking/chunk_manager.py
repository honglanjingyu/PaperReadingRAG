# app/service/core/chunking/chunk_manager.py
"""
分块管理器 - 统一分块接口
整合项目原有的 naive_merge 分块逻辑
"""

import re
import sys
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from .chunk_strategies import (
    ChunkStrategy, BaseChunker, Chunk,
    FixedTokenChunker, SemanticChunker,
    RecursiveChunker, SentenceChunker, ParagraphChunker
)


def num_tokens_from_string(text: str) -> int:
    """估算 token 数量（简化版）"""
    if not text:
        return 0
    # 中文约 1.5 字符/token，英文约 4 字符/token
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


class ChunkManager:
    """
    分块管理器 - 统一管理各种分块策略
    整合项目原有的分块逻辑
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化分块管理器

        Args:
            config: 配置参数
                - strategy: 分块策略 (default: 'recursive')
                - chunk_token_num: 块大小 (default: 128)
                - overlap_tokens: 重叠 token 数 (default: 20)
                - min_chunk_size: 最小块大小 (default: 20)
                - delimiter: 分隔符 (default: "\n!?。；！？")
                - sentences_per_chunk: 每块句子数 (用于 sentence 策略)
                - similarity_threshold: 相似度阈值 (用于 semantic 策略)
        """
        self.config = {
            'strategy': 'recursive',
            'chunk_token_num': 128,
            'overlap_tokens': 20,
            'min_chunk_size': 20,
            'delimiter': "\n!?。；！？",
            'sentences_per_chunk': 5,
            'similarity_threshold': 0.7,
            **(config or {})
        }

        # 初始化分块器
        self._init_chunker()

    def _init_chunker(self):
        """根据配置初始化分块器"""
        strategy_map = {
            ChunkStrategy.FIXED_TOKEN.value: FixedTokenChunker,
            ChunkStrategy.SEMANTIC.value: SemanticChunker,
            ChunkStrategy.RECURSIVE.value: RecursiveChunker,
            ChunkStrategy.SENTENCE.value: SentenceChunker,
            ChunkStrategy.PARAGRAPH.value: ParagraphChunker,
        }

        strategy = self.config['strategy']
        if isinstance(strategy, str):
            strategy = strategy.lower()

        chunker_class = strategy_map.get(strategy, RecursiveChunker)
        self.chunker = chunker_class(self.config)

    def set_strategy(self, strategy: Union[str, ChunkStrategy]):
        """切换分块策略"""
        if isinstance(strategy, str):
            strategy = strategy.lower()
        self.config['strategy'] = strategy
        self._init_chunker()

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        对单个文本进行分块

        Args:
            text: 输入文本
            metadata: 元数据

        Returns:
            分块列表
        """
        if not text:
            return []

        return self.chunker.chunk(text, metadata)

    def chunk_sections(self, sections: List[tuple],
                       metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        对段落列表进行分块（兼容项目原有格式）

        Args:
            sections: 段落列表，格式 [(text, style), ...]
            metadata: 元数据

        Returns:
            分块列表
        """
        if not sections:
            return []

        # 提取文本
        texts = [sec[0] for sec in sections if sec[0].strip()]
        full_text = "\n".join(texts)

        return self.chunk_text(full_text, metadata)

    def chunk_document(self, document: Dict[str, Any]) -> List[Chunk]:
        """
        对文档对象进行分块

        Args:
            document: 文档字典，包含 content_with_weight 字段

        Returns:
            分块列表
        """
        content = document.get('content_with_weight', '')
        if not content:
            return []

        metadata = {
            'doc_id': document.get('id', ''),
            'docnm_kwd': document.get('docnm_kwd', ''),
            'kb_id': document.get('kb_id', ''),
        }

        return self.chunk_text(content, metadata)

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        """
        批量分块文档

        Args:
            documents: 文档列表

        Returns:
            分块列表
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def naive_merge(self, sections: List[tuple],
                    chunk_token_num: int = None,
                    delimiter: str = None) -> List[str]:
        """
        项目原有的 naive_merge 分块逻辑

        Args:
            sections: 段落列表，格式 [(text, pos), ...]
            chunk_token_num: 块大小
            delimiter: 分隔符

        Returns:
            分块文本列表
        """
        if not sections:
            return []

        if chunk_token_num is None:
            chunk_token_num = self.config['chunk_token_num']
        if delimiter is None:
            delimiter = self.config['delimiter']

        # 标准化 sections 格式
        if isinstance(sections[0], str):
            sections = [(s, "") for s in sections]

        cks = [""]  # 分块结果
        tk_nums = [0]  # token 计数

        def add_chunk(t: str, pos: str):
            tnum = num_tokens_from_string(t)

            if not pos:
                pos = ""
            if tnum < 8:
                pos = ""

            # 超过限制则创建新块
            if tk_nums[-1] > chunk_token_num:
                if t.find(pos) < 0:
                    t += pos
                cks.append(t)
                tk_nums.append(tnum)
            else:
                if cks[-1].find(pos) < 0:
                    t += pos
                cks[-1] += t
                tk_nums[-1] += tnum

        for sec, pos in sections:
            add_chunk(sec, pos)

        return cks

    def naive_merge_docx(self, sections: List[tuple],
                         chunk_token_num: int = None,
                         delimiter: str = None) -> tuple:
        """
        DOCX 文件的 naive_merge 分块（支持图片）

        Returns:
            (chunks, images): 分块文本列表和对应的图片列表
        """
        if not sections:
            return [], []

        if chunk_token_num is None:
            chunk_token_num = self.config['chunk_token_num']
        if delimiter is None:
            delimiter = self.config['delimiter']

        cks = [""]
        images = [None]
        tk_nums = [0]

        def add_chunk(t: str, image, pos: str = ""):
            tnum = num_tokens_from_string(t)

            if tnum < 8:
                pos = ""

            if tk_nums[-1] > chunk_token_num:
                if t.find(pos) < 0:
                    t += pos
                cks.append(t)
                images.append(image)
                tk_nums.append(tnum)
            else:
                if cks[-1].find(pos) < 0:
                    t += pos
                cks[-1] += t
                tk_nums[-1] += tnum

        for sec, image in sections:
            add_chunk(sec, image, '')

        return cks, images

    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        获取分块统计信息

        Args:
            chunks: 分块列表

        Returns:
            统计信息字典
        """
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