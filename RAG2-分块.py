# RAG2-分块.py
"""
RAG 文档处理 - 数据清洗 + 智能分块
集成了数据清洗、多种分块策略、表格处理等功能
"""

import os
import sys
import re
import unicodedata
import html
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pdfplumber
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ==================== 分块策略枚举 ====================

class ChunkStrategy(Enum):
    """分块策略枚举"""
    FIXED_TOKEN = "fixed_token"  # 固定 token 数
    SEMANTIC = "semantic"  # 语义分块
    RECURSIVE = "recursive"  # 递归分块
    SENTENCE = "sentence"  # 句子级分块
    PARAGRAPH = "paragraph"  # 段落级分块
    NAIVE = "naive"  # 原有的 naive_merge 分块


@dataclass
class Chunk:
    """分块数据结构"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_idx: int = 0
    end_idx: int = 0
    token_count: int = 0
    page_num: int = 0
    chunk_index: int = 0

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'token_count': self.token_count,
            'page_num': self.page_num,
            'chunk_index': self.chunk_index
        }


# ==================== 分块器基类 ====================

class BaseChunker:
    """分块器基类"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = {
            'chunk_token_num': 128,  # 默认块大小（token数）
            'min_chunk_size': 20,  # 最小块大小
            'overlap_tokens': 20,  # 重叠 token 数
            'delimiter': "\n!?。；！？",  # 分隔符
            **(config or {})
        }

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """分块主方法，子类必须实现"""
        raise NotImplementedError

    def _count_tokens(self, text: str) -> int:
        """
        估算 token 数量
        中文约 1.5 字符/token，英文约 4 字符/token
        """
        if not text:
            return 0
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    def _create_chunk(self, text: str, start: int, end: int,
                      metadata: Dict, chunk_id: int, page_num: int = 0) -> Chunk:
        """创建 Chunk 对象"""
        return Chunk(
            id=f"chunk_{chunk_id}",
            content=text[start:end].strip(),
            metadata=metadata or {},
            start_idx=start,
            end_idx=end,
            token_count=self._count_tokens(text[start:end]),
            page_num=page_num,
            chunk_index=chunk_id
        )


class FixedTokenChunker(BaseChunker):
    """
    固定 Token 数分块策略
    按指定的 token 数进行切分，支持重叠
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.chunk_size = self.config.get('chunk_token_num', 128)
        self.overlap = self.config.get('overlap_tokens', 20)

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if not text:
            return []

        chunks = []
        start = 0
        length = len(text)
        chunk_id = 0

        while start < length:
            # 估算结束位置
            estimated_end = min(start + self.chunk_size * 4, length)

            # 尝试在分隔符处切分
            chunk_text = text[start:estimated_end]

            # 寻找最佳切分点
            best_split = estimated_end
            for delim in self.config['delimiter']:
                pos = chunk_text.rfind(delim)
                if pos != -1:
                    split_pos = start + pos + 1
                    if split_pos > start and split_pos - start > self.config['min_chunk_size'] * 4:
                        best_split = split_pos
                        break

            chunk_obj = self._create_chunk(text, start, best_split,
                                           metadata or {}, chunk_id)
            chunks.append(chunk_obj)

            # 移动起始位置（考虑重叠）
            start = best_split - self.overlap * 4
            chunk_id += 1

        return chunks


class RecursiveChunker(BaseChunker):
    """
    递归分块策略
    优先在段落、句子、短语等自然边界处切分
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.separators = [
            "\n\n",  # 段落
            "\n",  # 换行
            "。", "！", "？", "；",  # 中文句子分隔符
            ". ", "! ", "? ", "; ",  # 英文句子分隔符
            "，", ", ",  # 逗号
            " ",  # 空格
        ]

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if not text:
            return []

        return self._recursive_split(text, metadata or {}, 0)

    def _recursive_split(self, text: str, metadata: Dict,
                         depth: int = 0, chunk_id_start: int = 0) -> List[Chunk]:
        """递归分割文本"""
        if self._count_tokens(text) <= self.config['chunk_token_num']:
            return [self._create_chunk(text, 0, len(text), metadata, chunk_id_start)]

        chunks = []
        current_id = chunk_id_start

        # 尝试用不同级别分隔符分割
        for separator in self.separators:
            if separator in text:
                parts = text.split(separator, 1)
                left, right = parts[0], parts[1]

                # 检查左侧是否适合作为独立块
                if self._count_tokens(left) >= self.config['min_chunk_size']:
                    if self._count_tokens(left) <= self.config['chunk_token_num']:
                        chunks.append(self._create_chunk(left, 0, len(left), metadata, current_id))
                        current_id += 1
                    else:
                        # 左侧太大，继续分割
                        sub_chunks = self._recursive_split(left, metadata, depth + 1, current_id)
                        chunks.extend(sub_chunks)
                        current_id += len(sub_chunks)

                    # 处理右侧
                    sub_chunks = self._recursive_split(right, metadata, depth + 1, current_id)
                    chunks.extend(sub_chunks)
                    return chunks

        # 如果无法按分隔符分割，强制按长度分割
        mid = len(text) // 2
        left, right = text[:mid], text[mid:]

        chunks.extend(self._recursive_split(left, metadata, depth + 1, current_id))
        current_id += len(chunks)
        chunks.extend(self._recursive_split(right, metadata, depth + 1, current_id))

        return chunks


class SentenceChunker(BaseChunker):
    """句子级分块策略"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.sentences_per_chunk = config.get('sentences_per_chunk', 5) if config else 5

    def _split_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        sentence_delimiters = r'(?<=[。！？.!?])\s+'
        sentences = re.split(sentence_delimiters, text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if not text:
            return []

        sentences = self._split_sentences(text)
        chunks = []
        chunk_id = 0

        for i in range(0, len(sentences), self.sentences_per_chunk):
            chunk_text = ''.join(sentences[i:i + self.sentences_per_chunk])
            chunk_obj = self._create_chunk(chunk_text, 0, len(chunk_text),
                                           metadata or {}, chunk_id)
            chunks.append(chunk_obj)
            chunk_id += 1

        return chunks


class ParagraphChunker(BaseChunker):
    """段落级分块策略"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.recursive_chunker = RecursiveChunker(config)

    def _split_paragraphs(self, text: str) -> List[str]:
        """按段落分割"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if not text:
            return []

        paragraphs = self._split_paragraphs(text)
        chunks = []
        chunk_id = 0
        current_chunk = ""

        for para in paragraphs:
            # 如果单个段落超过限制，需要进一步切分
            if self._count_tokens(para) > self.config['chunk_token_num']:
                if current_chunk:
                    chunk_obj = self._create_chunk(current_chunk, 0, len(current_chunk),
                                                   metadata or {}, chunk_id)
                    chunks.append(chunk_obj)
                    chunk_id += 1
                    current_chunk = ""

                # 使用递归分块处理长段落
                sub_chunks = self.recursive_chunker.chunk(para, metadata)
                for sc in sub_chunks:
                    sc.id = f"chunk_{chunk_id}"
                    sc.chunk_index = chunk_id
                    chunks.append(sc)
                    chunk_id += 1

            elif self._count_tokens(current_chunk + "\n" + para) > self.config['chunk_token_num']:
                # 当前块已满
                if current_chunk:
                    chunk_obj = self._create_chunk(current_chunk, 0, len(current_chunk),
                                                   metadata or {}, chunk_id)
                    chunks.append(chunk_obj)
                    chunk_id += 1
                current_chunk = para
            else:
                # 合并到当前块
                current_chunk += "\n" + para if current_chunk else para

        # 添加最后一个块
        if current_chunk:
            chunk_obj = self._create_chunk(current_chunk, 0, len(current_chunk),
                                           metadata or {}, chunk_id)
            chunks.append(chunk_obj)

        return chunks


class SemanticChunker(BaseChunker):
    """
    语义分块策略
    基于句子嵌入相似度进行分块
    """

    def __init__(self, config: Optional[Dict] = None, embedding_model=None):
        super().__init__(config)
        self.embedding_model = embedding_model
        self.similarity_threshold = config.get('similarity_threshold', 0.7) if config else 0.7

    def _split_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        sentence_delimiters = r'(?<=[。！？.!?])\s+'
        sentences = re.split(sentence_delimiters, text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_similarity(self, sent1: str, sent2: str) -> float:
        """计算两个句子的相似度（简化版）"""
        # 基于共同词汇的 Jaccard 相似度
        words1 = set(re.findall(r'\w+', sent1.lower()))
        words2 = set(re.findall(r'\w+', sent2.lower()))
        if not words1 or not words2:
            return 0
        return len(words1 & words2) / len(words1 | words2)

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if not text:
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk = sentences[0]
        chunk_id = 0

        for i in range(1, len(sentences)):
            similarity = self._compute_similarity(sentences[i - 1], sentences[i])

            # 如果相似度低于阈值，且当前块已达到最小大小，则切分
            if similarity < self.similarity_threshold and \
                    self._count_tokens(current_chunk) >= self.config['min_chunk_size']:
                chunk_obj = self._create_chunk(current_chunk, 0, len(current_chunk),
                                               metadata or {}, chunk_id)
                chunks.append(chunk_obj)
                current_chunk = sentences[i]
                chunk_id += 1
            else:
                # 合并到当前块
                current_chunk += sentences[i]

        # 添加最后一个块
        if current_chunk:
            chunk_obj = self._create_chunk(current_chunk, 0, len(current_chunk),
                                           metadata or {}, chunk_id)
            chunks.append(chunk_obj)

        return chunks


class NaiveChunker(BaseChunker):
    """
    原有的 naive_merge 分块策略
    保持与项目原有逻辑兼容
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        使用原有逻辑进行分块
        """
        if not text:
            return []

        # 分割成段落
        sections = [(line, "") for line in text.split('\n') if line.strip()]

        # 使用 naive_merge 逻辑
        chunks_text = self._naive_merge(sections)

        # 转换为 Chunk 对象
        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            if chunk_text.strip():
                chunk_obj = self._create_chunk(chunk_text, 0, len(chunk_text),
                                               metadata or {}, i)
                chunks.append(chunk_obj)

        return chunks

    def _naive_merge(self, sections: List[tuple]) -> List[str]:
        """
        项目原有的 naive_merge 分块逻辑
        """
        if not sections:
            return []

        if isinstance(sections[0], str):
            sections = [(s, "") for s in sections]

        cks = [""]
        tk_nums = [0]
        chunk_token_num = self.config['chunk_token_num']

        def add_chunk(t: str, pos: str):
            tnum = self._count_tokens(t)
            if not pos:
                pos = ""
            if tnum < 8:
                pos = ""

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


# ==================== 分块管理器 ====================

class ChunkManager:
    """分块管理器 - 统一管理各种分块策略"""

    def __init__(self, config: Optional[Dict] = None):
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

        self._init_chunker()

    def _init_chunker(self):
        """根据配置初始化分块器"""
        strategy_map = {
            ChunkStrategy.FIXED_TOKEN.value: FixedTokenChunker,
            ChunkStrategy.SEMANTIC.value: SemanticChunker,
            ChunkStrategy.RECURSIVE.value: RecursiveChunker,
            ChunkStrategy.SENTENCE.value: SentenceChunker,
            ChunkStrategy.PARAGRAPH.value: ParagraphChunker,
            ChunkStrategy.NAIVE.value: NaiveChunker,
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
        """对单个文本进行分块"""
        if not text:
            return []
        return self.chunker.chunk(text, metadata)

    def chunk_sections(self, sections: List[tuple],
                       metadata: Optional[Dict] = None) -> List[Chunk]:
        """对段落列表进行分块"""
        if not sections:
            return []

        texts = [sec[0] for sec in sections if sec[0].strip()]
        full_text = "\n".join(texts)

        return self.chunk_text(full_text, metadata)

    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
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


# ==================== 数据清洗模块（增强版） ====================

class DataCleaner:
    """数据清洗器 - 处理各种文档类型的文本清洗"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = {
            'remove_empty_lines': True,
            'remove_special_chars': True,
            'normalize_whitespace': True,
            'max_line_length': None,
            'min_line_length': 2,
            'remove_urls': False,
            'remove_emails': False,
            'remove_numbers': False,
            'language': 'auto',
            **(config or {})
        }

        # 特殊字符模式
        self.special_chars_pattern = re.compile(
            r'[^\u4e00-\u9fff\u3400-\u4dbf\w\s\.\,\!\?\;\:\'\"\(\)\[\]\{\}\<\>\/\\\|\-\=\+\*\&\^\$\#\@\~`]'
        )

        # URL 模式
        self.url_pattern = re.compile(
            r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*|'
            r'www\.[-\w.]+[^\s]*'
        )

        # 邮箱模式
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )

        # 噪声模式
        self.noise_patterns = [
            re.compile(r'^\s*\d+\s*$'),
            re.compile(r'^\s*第\s*\d+\s*页\s*$'),
            re.compile(r'^\s*Page\s+\d+\s*$', re.IGNORECASE),
            re.compile(r'^\s*\d+\s*/\s*\d+\s*$'),
            re.compile(r'^\s*Copyright\s+©?\s*\d{4}\s*$', re.IGNORECASE),
            re.compile(r'^\s*All\s+Rights\s+Reserved\s*$', re.IGNORECASE),
            re.compile(r'^\s*Confidential\s*$', re.IGNORECASE),
            re.compile(r'^[-=_*]{10,}$'),
            re.compile(r'^[─━]{10,}$'),
        ]

    def clean_text(self, text: str, verbose: bool = False) -> str:
        """主清洗函数"""
        if not text:
            return ""

        original_length = len(text)

        # 1. 标准化 Unicode
        text = self._normalize_unicode(text)

        # 2. 移除控制字符
        text = self._remove_control_chars(text)

        # 3. 标准化空白字符
        if self.config['normalize_whitespace']:
            text = self._normalize_whitespace(text)

        # 4. 移除URL
        if self.config['remove_urls']:
            text = self._remove_urls(text)

        # 5. 移除邮箱
        if self.config['remove_emails']:
            text = self._remove_emails(text)

        # 6. 移除特殊字符
        if self.config['remove_special_chars']:
            text = self._remove_special_chars(text)

        # 7. 移除数字
        if self.config['remove_numbers']:
            text = self._remove_numbers(text)

        # 8. 按行清理
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            cleaned_line = self._clean_line(line)
            if cleaned_line:
                cleaned_lines.append(cleaned_line)

        cleaned_text = '\n'.join(cleaned_lines)

        if verbose:
            print(f"  清洗统计: {original_length} -> {len(cleaned_text)} 字符")

        return cleaned_text

    def _normalize_unicode(self, text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        text = text.replace('\u3000', ' ')
        return text

    def _remove_control_chars(self, text: str) -> str:
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r'[ \t]+', ' ', text)
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines)

    def _remove_urls(self, text: str) -> str:
        return self.url_pattern.sub('', text)

    def _remove_emails(self, text: str) -> str:
        return self.email_pattern.sub('', text)

    def _remove_special_chars(self, text: str) -> str:
        return self.special_chars_pattern.sub('', text)

    def _remove_numbers(self, text: str) -> str:
        return re.sub(r'\b\d+(?:\.\d+)?\b', '', text)

    def _clean_line(self, line: str) -> Optional[str]:
        line = line.strip()

        if not line:
            return None if self.config['remove_empty_lines'] else ''

        if len(line) < self.config['min_line_length']:
            return None

        if self.config['max_line_length'] and len(line) > self.config['max_line_length']:
            line = line[:self.config['max_line_length']]

        for pattern in self.noise_patterns:
            if pattern.match(line):
                return None

        return line


class TableCleaner:
    """表格数据清洗器"""

    @staticmethod
    def clean_table_data(table_data: List[List]) -> List[List]:
        if not table_data:
            return []

        cleaned_table = []
        for row in table_data:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append('')
                else:
                    cleaned = str(cell).strip()
                    cleaned = re.sub(r'\s+', ' ', cleaned)
                    cleaned_row.append(cleaned)
            cleaned_table.append(cleaned_row)

        return cleaned_table

    @staticmethod
    def table_to_markdown(table_data: List[List]) -> str:
        if not table_data or len(table_data) < 2:
            return ""

        lines = []
        header = "| " + " | ".join(str(cell) for cell in table_data[0]) + " |"
        lines.append(header)
        separator = "| " + " | ".join(["---"] * len(table_data[0])) + " |"
        lines.append(separator)

        for row in table_data[1:]:
            line = "| " + " | ".join(str(cell) for cell in row) + " |"
            lines.append(line)

        return "\n".join(lines)


class NoiseFilter:
    """噪声内容过滤器"""

    def __init__(self):
        self.noise_keywords = [
            '请勿转载', '版权所有', '翻印必究',
            'confidential', 'proprietary', 'all rights reserved',
            '仅供内部使用', '内部资料', '注意保密',
        ]

    def is_noise_line(self, line: str, threshold: float = 0.3) -> bool:
        if not line:
            return True

        line_lower = line.lower()
        noise_count = sum(1 for kw in self.noise_keywords if kw.lower() in line_lower)
        if noise_count / max(len(line) / 20, 1) > threshold:
            return True

        if re.match(r'^[\W_]+$', line):
            return True

        return False


# ==================== PDF 解析器（增强版） ====================

class PlainParser:
    """PDF 简单解析器 - 仅提取纯文本"""

    def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
        lines = []
        with open(filename, 'rb') as f:
            pdf = pdfplumber.open(BytesIO(f.read()))
            pages = pdf.pages[from_page:to_page] if to_page <= len(pdf.pages) else pdf.pages[from_page:]
            for page in pages:
                text = page.extract_text()
                if text:
                    lines.extend([t for t in text.split("\n") if t.strip()])
        return [(line, "") for line in lines], []


class EnhancedPDFParser:
    """增强版 PDF 解析器 - 包含数据清洗和分块功能"""

    def __init__(self, clean_config: Optional[Dict] = None, chunk_config: Optional[Dict] = None):
        """
        初始化增强解析器

        Args:
            clean_config: 清洗配置参数
            chunk_config: 分块配置参数
        """
        self.text_cleaner = DataCleaner(clean_config)
        self.table_cleaner = TableCleaner()
        self.noise_filter = NoiseFilter()
        self.chunk_manager = ChunkManager(chunk_config)

    def parse(self, pdf_path: str, from_page: int = 0, to_page: int = 100000,
              enable_cleaning: bool = True, enable_chunking: bool = True,
              verbose: bool = True) -> Tuple[List[Chunk], List, Dict]:
        """
        解析 PDF 并可选清洗和分块

        Args:
            pdf_path: PDF 文件路径
            from_page: 起始页
            to_page: 结束页
            enable_cleaning: 是否启用数据清洗
            enable_chunking: 是否启用分块
            verbose: 是否打印详细信息

        Returns:
            (chunks, tables, stats): 分块列表、表格数据、统计信息
        """
        print(f"\n{'=' * 60}")
        print(f"增强版 PDF 解析: {pdf_path}")
        print(f"数据清洗: {'启用' if enable_cleaning else '禁用'}")
        print(f"分块: {'启用' if enable_chunking else '禁用'}")
        print(f"分块策略: {self.chunk_manager.config['strategy']}")
        print(f"块大小: {self.chunk_manager.config['chunk_token_num']} tokens")
        print(f"{'=' * 60}")

        if not os.path.exists(pdf_path):
            print(f"错误: 文件不存在 - {pdf_path}")
            return [], [], {}

        file_size = os.path.getsize(pdf_path) / 1024
        print(f"文件大小: {file_size:.2f} KB")

        all_chunks = []
        all_tables = []
        page_texts = []  # 存储每页的文本，用于分块

        stats = {
            'total_pages': 0,
            'text_pages': 0,
            'table_pages': 0,
            'total_tables': 0,
            'original_chars': 0,
            'cleaned_chars': 0,
            'filtered_lines': 0,
            'total_chunks': 0,
            'avg_chunk_size': 0
        }

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            stats['total_pages'] = total_pages
            print(f"总页数: {total_pages}")

            start_page = max(0, from_page)
            end_page = min(total_pages, to_page)
            print(f"解析页范围: 第 {start_page + 1} - {end_page} 页")

            for page_num in range(start_page, end_page):
                if verbose:
                    print(f"\n--- 第 {page_num + 1} / {total_pages} 页 ---")
                page = pdf.pages[page_num]

                # 1. 提取和清洗文本
                text = page.extract_text()
                page_text = []

                if text:
                    stats['text_pages'] += 1
                    lines = text.split('\n')

                    for line in lines:
                        line = line.strip()
                        if line:
                            if enable_cleaning:
                                cleaned_line = self.text_cleaner.clean_text(line)
                                stats['original_chars'] += len(line)
                                stats['cleaned_chars'] += len(cleaned_line)

                                if not self.noise_filter.is_noise_line(cleaned_line):
                                    if cleaned_line:
                                        page_text.append(cleaned_line)
                                        all_chunks.append(Chunk(
                                            id=f"chunk_{len(all_chunks)}",
                                            content=cleaned_line,
                                            metadata={'page': page_num + 1},
                                            page_num=page_num + 1,
                                            chunk_index=len(all_chunks)
                                        ))
                                else:
                                    stats['filtered_lines'] += 1
                            else:
                                page_text.append(line)
                                all_chunks.append(Chunk(
                                    id=f"chunk_{len(all_chunks)}",
                                    content=line,
                                    metadata={'page': page_num + 1},
                                    page_num=page_num + 1,
                                    chunk_index=len(all_chunks)
                                ))

                    if verbose:
                        print(f"  提取文本: {len(lines)} 行, {len(text)} 字符")

                    # 保存该页文本用于后续分块
                    if page_text:
                        page_texts.append("\n".join(page_text))
                else:
                    if verbose:
                        print(f"  未提取到文本")

                # 2. 提取和清洗表格
                tables = page.extract_tables()
                if tables:
                    stats['table_pages'] += 1
                    if verbose:
                        print(f"  找到 {len(tables)} 个表格")

                    for i, table in enumerate(tables):
                        if table and len(table) > 0:
                            stats['total_tables'] += 1

                            if enable_cleaning:
                                cleaned_table = self.table_cleaner.clean_table_data(table)
                            else:
                                cleaned_table = table

                            try:
                                headers = cleaned_table[0] if cleaned_table[0] else None
                                data = cleaned_table[1:] if len(cleaned_table) > 1 else []
                                df = pd.DataFrame(data, columns=headers)

                                if verbose:
                                    print(f"    表格 {i + 1}: {df.shape[0]} 行 x {df.shape[1]} 列")

                                all_tables.append({
                                    'page': page_num + 1,
                                    'table_index': i,
                                    'data': cleaned_table,
                                    'dataframe': df
                                })
                            except Exception as e:
                                if verbose:
                                    print(f"    表格 {i + 1}: 解析失败 - {e}")
                                all_tables.append({
                                    'page': page_num + 1,
                                    'table_index': i,
                                    'data': cleaned_table
                                })
                elif verbose:
                    print(f"  未找到表格")

        # 3. 如果启用分块，对提取的内容进行重分块
        if enable_chunking and page_texts:
            print(f"\n--- 开始分块处理 ---")
            full_text = "\n\n".join(page_texts)

            # 使用分块管理器进行分块
            chunks = self.chunk_manager.chunk_text(full_text, {'source': pdf_path})

            # 替换原有的行级分块
            all_chunks = chunks

            stats['total_chunks'] = len(chunks)
            chunk_stats = self.chunk_manager.get_chunk_statistics(chunks)
            stats['avg_chunk_size'] = chunk_stats['avg_token_count']

            if verbose:
                print(f"分块结果:")
                print(f"  - 总块数: {len(chunks)}")
                print(f"  - 平均块大小: {stats['avg_chunk_size']:.1f} tokens")
                print(f"  - 最小块: {chunk_stats['min_token_count']} tokens")
                print(f"  - 最大块: {chunk_stats['max_token_count']} tokens")

        return all_chunks, all_tables, stats


# ==================== 辅助函数 ====================

def parse_pdf_with_tables(pdf_path: str, from_page: int = 0, to_page: int = 100000):
    """使用 pdfplumber 解析 PDF，支持表格提取（原始版本）"""
    print(f"\n{'=' * 60}")
    print(f"解析 PDF: {pdf_path}")
    print(f"{'=' * 60}")

    if not os.path.exists(pdf_path):
        print(f"错误: 文件不存在 - {pdf_path}")
        return [], []

    all_sections = []
    all_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"总页数: {total_pages}")

        start_page = max(0, from_page)
        end_page = min(total_pages, to_page)

        for page_num in range(start_page, end_page):
            print(f"\n--- 第 {page_num + 1} / {total_pages} 页 ---")
            page = pdf.pages[page_num]

            text = page.extract_text()
            if text:
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line:
                        all_sections.append((line, ""))
                print(f"  提取文本: {len(lines)} 行, {len(text)} 字符")

            tables = page.extract_tables()
            if tables:
                print(f"  找到 {len(tables)} 个表格")
                for i, table in enumerate(tables):
                    if table and len(table) > 0:
                        try:
                            headers = table[0] if table[0] else None
                            data = table[1:] if len(table) > 1 else []
                            df = pd.DataFrame(data, columns=headers)
                            print(f"    表格 {i + 1}: {df.shape[0]} 行 x {df.shape[1]} 列")
                            all_tables.append({
                                'page': page_num + 1,
                                'table_index': i,
                                'data': table,
                                'dataframe': df
                            })
                        except Exception as e:
                            print(f"    表格 {i + 1}: 解析失败 - {e}")
            else:
                print(f"  未找到表格")

    return all_sections, all_tables


def save_chunks_to_file(chunks: List[Chunk], output_path: str):
    """保存分块结果到文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("PDF 解析分块结果\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"总块数: {len(chunks)}\n\n")

        for i, chunk in enumerate(chunks):
            f.write(f"\n{'=' * 40}\n")
            f.write(f"块 {i + 1} (ID: {chunk.id})\n")
            f.write(f"{'-' * 40}\n")
            f.write(f"页数: {chunk.page_num}\n")
            f.write(f"Token 数: {chunk.token_count}\n")
            f.write(f"元数据: {chunk.metadata}\n")
            f.write(f"内容:\n{chunk.content}\n")
            f.write(f"{'=' * 40}\n")

    print(f"\n分块结果已保存到: {output_path}")


def print_chunk_summary(chunks: List[Chunk]):
    """打印分块摘要"""
    if not chunks:
        print("无分块数据")
        return

    print(f"\n{'=' * 60}")
    print("分块摘要")
    print(f"{'=' * 60}")

    total_tokens = sum(c.token_count for c in chunks)
    avg_tokens = total_tokens / len(chunks) if chunks else 0

    print(f"\n分块统计:")
    print(f"  - 总块数: {len(chunks)}")
    print(f"  - 总 Token 数: {total_tokens}")
    print(f"  - 平均 Token 数: {avg_tokens:.1f}")
    print(f"  - 最小块: {min(c.token_count for c in chunks)} tokens")
    print(f"  - 最大块: {max(c.token_count for c in chunks)} tokens")

    # 打印前几个块的预览
    print(f"\n前 5 个块预览:")
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n块 {i + 1} (Token: {chunk.token_count}):")
        preview = chunk.content[:200].replace('\n', ' ')
        print(f"  {preview}...")


# ==================== 演示函数 ====================

def demo_chunk_strategies():
    """演示1：不同分块策略对比"""
    print("\n" + "=" * 60)
    print("演示1：不同分块策略对比")
    print("=" * 60)

    test_text = """
    第一章：引言

    人工智能是计算机科学的一个重要分支。它旨在创造能够模拟人类智能的系统。
    近年来，深度学习技术取得了巨大的突破，在图像识别、自然语言处理等领域都有广泛应用。

    第二章：核心技术

    自然语言处理是人工智能领域的关键技术之一。
    计算机视觉则专注于让机器理解和处理图像信息。
    """ * 5

    strategies = [
        ('naive', '原有分块'),
        ('fixed_token', '固定Token'),
        ('recursive', '递归分块'),
        ('sentence', '句子级'),
        ('paragraph', '段落级')
    ]

    for strategy, name in strategies:
        manager = ChunkManager({
            'strategy': strategy,
            'chunk_token_num': 64
        })
        chunks = manager.chunk_text(test_text)

        print(f"\n{name} ({strategy}):")
        print(f"  块数量: {len(chunks)}")
        if chunks:
            avg_size = sum(c.token_count for c in chunks) / len(chunks)
            print(f"  平均块大小: {avg_size:.1f} tokens")
            print(f"  块大小范围: {min(c.token_count for c in chunks)} - {max(c.token_count for c in chunks)}")


def demo_enhanced_parser_with_chunking(pdf_path: str):
    """演示2：使用增强版解析器（带清洗和分块）"""
    print("\n" + "=" * 60)
    print("演示2：增强版 PDF 解析器（启用清洗和分块）")
    print("=" * 60)

    # 创建增强解析器
    parser = EnhancedPDFParser(
        clean_config={
            'remove_empty_lines': True,
            'normalize_whitespace': True,
            'min_line_length': 2,
            'remove_urls': True,
            'remove_emails': True,
        },
        chunk_config={
            'strategy': 'recursive',
            'chunk_token_num': 128
        }
    )

    # 解析、清洗、分块
    chunks, tables, stats = parser.parse(
        pdf_path,
        from_page=0,
        to_page=10,
        enable_cleaning=True,
        enable_chunking=True,
        verbose=True
    )

    # 打印摘要
    print(f"\n{'=' * 60}")
    print("最终结果摘要")
    print(f"{'=' * 60}")
    print(f"分块数量: {len(chunks)}")
    print(f"表格数量: {len(tables)}")

    if stats:
        print(f"页面统计: {stats.get('total_pages', 0)} 页")
        print(f"表格统计: {stats.get('total_tables', 0)} 个")
        print(f"总块数: {stats.get('total_chunks', 0)}")
        print(f"平均块大小: {stats.get('avg_chunk_size', 0):.1f} tokens")
        if stats.get('original_chars', 0) > 0:
            reduction = stats['original_chars'] - stats['cleaned_chars']
            print(f"字符减少: {reduction} ({reduction / stats['original_chars'] * 100:.1f}%)")

    # 保存结果
    if chunks:
        save_chunks_to_file(chunks, "pdf_chunks_result.txt")
        print_chunk_summary(chunks)

    return chunks, tables, stats


def demo_strategy_comparison(pdf_path: str):
    """演示3：对比不同分块策略的效果"""
    print("\n" + "=" * 60)
    print("演示3：不同分块策略效果对比")
    print("=" * 60)

    strategies = [
        ('naive', 64),
        ('fixed_token', 64),
        ('recursive', 64),
        ('sentence', 64),
        ('paragraph', 64),
    ]

    results = {}

    for strategy, chunk_size in strategies:
        print(f"\n>>> 正在测试策略: {strategy} (块大小: {chunk_size})")

        parser = EnhancedPDFParser(
            clean_config={'remove_empty_lines': True},
            chunk_config={'strategy': strategy, 'chunk_token_num': chunk_size}
        )

        chunks, tables, stats = parser.parse(
            pdf_path,
            from_page=0,
            to_page=5,
            enable_cleaning=True,
            enable_chunking=True,
            verbose=False
        )

        results[strategy] = {
            'chunk_count': len(chunks),
            'avg_size': stats.get('avg_chunk_size', 0)
        }

        print(f"  块数量: {len(chunks)}, 平均大小: {stats.get('avg_chunk_size', 0):.1f} tokens")

    # 打印对比结果
    print(f"\n{'=' * 60}")
    print("策略对比结果")
    print(f"{'=' * 60}")
    print(f"{'策略':<15} {'块数量':<10} {'平均大小(tokens)':<15}")
    print("-" * 40)
    for strategy, data in results.items():
        print(f"{strategy:<15} {data['chunk_count']:<10} {data['avg_size']:<15.1f}")


# ==================== 主函数 ====================

if __name__ == "__main__":
    # PDF 文件路径
    pdf_file = "【兴证电子】世运电路2023中报点评.pdf"

    print("=" * 60)
    print("RAG 文档处理 - 数据清洗 + 智能分块")
    print("版本: 2.0")
    print("=" * 60)

    print(f"\n当前目录: {os.getcwd()}")

    # 检查文件
    if not os.path.exists(pdf_file):
        print(f"\n错误: 文件不存在 - {pdf_file}")
        print(f"请将 PDF 文件放在当前目录下，或修改 pdf_file 变量为正确路径")
        print("\n" + "-" * 40)
        print("将运行演示模式（不依赖 PDF 文件）")
        print("-" * 40)

        # 演示模式
        demo_chunk_strategies()

        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)

    else:
        print(f"\n找到 PDF 文件: {pdf_file}")

        # 选择运行模式
        print("\n请选择运行模式:")
        print("  1. 分块策略对比")
        print("  2. 增强解析（清洗 + 分块）")
        print("  3. 原始解析（无清洗无分块）")

        choice = input("\n请输入选择 (1-3，默认 2): ").strip() or "2"

        if choice == "1":
            demo_strategy_comparison(pdf_file)

        elif choice == "2":
            demo_enhanced_parser_with_chunking(pdf_file)

        else:
            sections, tables = parse_pdf_with_tables(pdf_file, from_page=0, to_page=10)
            print(f"\n解析完成: {len(sections)} 段文本, {len(tables)} 个表格")

        print(f"\n{'=' * 60}")
        print("处理完成！")
        print(f"{'=' * 60}")