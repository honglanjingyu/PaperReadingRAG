"""
RAG 文档处理 - 文档读取与数据清洗模块
处理顺序：
1. 数据加载 - 加载多种格式文档
2. 布局识别 - 识别文档布局（左右分栏、表格、图片位置）
3. 连接跨页内容 - 合并跨页段落和表格
4. 数据清洗 - 清洗文本、过滤噪声

所有功能通过导入 app/service/core 模块实现
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# 导入 core 模块
from app.service.core.cleaner.data_cleaner import DataCleaner, TableCleaner, NoiseFilter
from app.service.core.cleaner.pipeline import CleaningPipeline
from app.service.core.deepdoc.parser.pdf_parser import PlainParser
from app.service.core.deepdoc.parser.docx_parser import RAGFlowDocxParser
from app.service.core.deepdoc.parser.txt_parser import RAGFlowTxtParser
from app.service.core.deepdoc.parser.excel_parser import RAGFlowExcelParser


# ==================== 数据结构 ====================

class LayoutType(Enum):
    """布局类型"""
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TEXT_WITH_TABLES = "text_with_tables"
    TEXT_WITH_IMAGES = "text_with_images"
    MIXED = "mixed"


@dataclass
class TextBlock:
    """文本块"""
    page_num: int
    content: str
    x0: float = 0
    y0: float = 0
    x1: float = 0
    y1: float = 0
    column: int = 0


@dataclass
class TableBlock:
    """表格块"""
    page_num: int
    data: List[List[str]]
    x0: float = 0
    y0: float = 0
    x1: float = 0
    y1: float = 0
    is_continued: bool = False
    continued_from_page: Optional[int] = None


@dataclass
class PageContent:
    """页面内容"""
    page_num: int
    text_blocks: List[TextBlock] = field(default_factory=list)
    tables: List[TableBlock] = field(default_factory=list)
    layout_type: LayoutType = LayoutType.SINGLE_COLUMN
    columns: int = 1


@dataclass
class ParsedDocument:
    """解析后的文档"""
    file_path: str
    file_name: str
    file_type: str
    pages: List[PageContent]
    total_pages: int
    cleaned_text: str = ""


# ==================== 1. 数据加载模块 ====================

class DataLoader:
    """数据加载器 - 加载文档原始内容"""

    SUPPORTED_TYPES = {
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.txt': 'text',
        '.md': 'text',
        '.markdown': 'text',
        '.xlsx': 'excel',
        '.xls': 'excel',
    }

    def __init__(self):
        self.pdf_parser = PlainParser()
        self.docx_parser = RAGFlowDocxParser()
        self.txt_parser = RAGFlowTxtParser()
        self.excel_parser = RAGFlowExcelParser()

    def load(self, file_path: str, from_page: int = 0, to_page: int = 100000) -> Dict[str, Any]:
        """加载文档"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()
        file_type = self.SUPPORTED_TYPES.get(file_ext, 'text')

        if file_type == 'pdf':
            return self._load_pdf(file_path, from_page, to_page)
        elif file_type == 'docx':
            return self._load_docx(file_path)
        elif file_type == 'text':
            return self._load_text(file_path)
        elif file_type == 'excel':
            return self._load_excel(file_path)
        else:
            return self._load_text(file_path)

    def _load_pdf(self, file_path: str, from_page: int, to_page: int) -> Dict[str, Any]:
        """加载 PDF"""
        import pdfplumber

        pages_raw = []
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            end_page = min(total_pages, to_page)

            for page_num in range(from_page, end_page):
                page = pdf.pages[page_num]
                pages_raw.append({
                    'page_num': page_num + 1,
                    'text': page.extract_text() or "",
                    'chars': page.chars if page.chars else [],
                    'words': page.extract_words() if page.chars else [],
                    'width': page.width,
                    'height': page.height,
                    'tables': page.extract_tables() or [],
                    'images': page.images if hasattr(page, 'images') else []
                })

        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_type': 'pdf',
            'total_pages': len(pages_raw),
            'pages_raw': pages_raw
        }

    def _load_docx(self, file_path: str) -> Dict[str, Any]:
        """加载 DOCX"""
        result = self.docx_parser(file_path)
        sections, tables = result if isinstance(result, tuple) and len(result) == 2 else (result, [])

        text = '\n'.join([s[0] for s in sections if s and s[0] and s[0].strip()]) if sections else ""

        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_type': 'docx',
            'total_pages': 1,
            'text': text,
            'tables': tables
        }

    def _load_text(self, file_path: str) -> Dict[str, Any]:
        """加载文本文件"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
        content = None

        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if content is None:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_type': 'text',
            'total_pages': 1,
            'text': content,
            'tables': []
        }

    def _load_excel(self, file_path: str) -> Dict[str, Any]:
        """加载 Excel"""
        with open(file_path, 'rb') as f:
            binary = f.read()
        lines = self.excel_parser(binary)
        text = '\n'.join(lines) if lines else ""

        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_type': 'excel',
            'total_pages': 1,
            'text': text,
            'tables': []
        }


# ==================== 2. 布局识别模块 ====================

class LayoutRecognizer:
    """布局识别器 - 识别文档布局"""

    def __init__(self):
        pass

    def recognize(self, raw_data: Dict[str, Any]) -> List[PageContent]:
        """识别文档布局"""
        file_type = raw_data.get('file_type')

        if file_type == 'pdf':
            return self._recognize_pdf_layout(raw_data)
        else:
            return self._recognize_simple_layout(raw_data)

    def _recognize_pdf_layout(self, raw_data: Dict[str, Any]) -> List[PageContent]:
        """识别 PDF 布局"""
        pages_content = []
        pages_raw = raw_data.get('pages_raw', [])

        for page_raw in pages_raw:
            page_num = page_raw['page_num']
            chars = page_raw.get('chars', [])
            width = page_raw.get('width', 0)
            tables_raw = page_raw.get('tables', [])

            # 检测分栏
            columns = self._detect_columns(chars, width)
            layout_type = self._detect_layout_type(tables_raw, len(chars) > 0)

            # 提取文本块
            text_blocks = self._extract_text_blocks(chars, page_num, columns)

            # 提取表格
            tables = self._extract_table_blocks(tables_raw, page_num)

            pages_content.append(PageContent(
                page_num=page_num,
                text_blocks=text_blocks,
                tables=tables,
                layout_type=layout_type,
                columns=len(columns) if columns else 1
            ))

        return pages_content

    def _detect_columns(self, chars: List[Dict], page_width: float) -> List[Tuple[float, float]]:
        """检测分栏"""
        if not chars or page_width == 0:
            return [(0, page_width)]

        x_positions = [c.get('x0', 0) for c in chars]
        rounded = [round(x / 50) * 50 for x in x_positions]

        from collections import Counter
        counter = Counter(rounded)

        clusters = []
        for x in sorted(counter.keys()):
            if not clusters or x - clusters[-1][1] > 100:
                clusters.append([x, x])
            else:
                clusters[-1][1] = x

        if len(clusters) <= 1:
            return [(0, page_width)]

        columns = []
        for cluster in clusters:
            left = max(0, cluster[0] - 30)
            right = min(page_width, cluster[1] + 30)
            columns.append((left, right))

        return columns

    def _detect_layout_type(self, tables: List, has_text: bool) -> LayoutType:
        """检测布局类型"""
        has_tables = len(tables) > 0

        if has_tables:
            return LayoutType.TEXT_WITH_TABLES
        else:
            return LayoutType.SINGLE_COLUMN

    def _extract_text_blocks(self, chars: List[Dict], page_num: int, columns: List) -> List[TextBlock]:
        """提取文本块"""
        if not chars:
            return []

        from collections import defaultdict
        lines = defaultdict(list)
        for char in chars:
            y = round(char.get('y0', 0) / 5) * 5
            lines[y].append(char)

        text_blocks = []
        for y, chars_in_line in sorted(lines.items()):
            chars_in_line.sort(key=lambda c: c.get('x0', 0))

            x0 = chars_in_line[0].get('x0', 0) if chars_in_line else 0
            column = 0
            for i, (left, right) in enumerate(columns):
                if left <= x0 <= right:
                    column = i
                    break

            text = ''.join([c.get('text', '') for c in chars_in_line])

            if text.strip():
                text_blocks.append(TextBlock(
                    page_num=page_num,
                    content=text,
                    x0=chars_in_line[0].get('x0', 0),
                    y0=y,
                    x1=chars_in_line[-1].get('x1', 0),
                    y1=y + 10,
                    column=column
                ))

        return text_blocks

    def _extract_table_blocks(self, tables_raw: List, page_num: int) -> List[TableBlock]:
        """提取表格块"""
        tables = []
        for table_data in tables_raw:
            if table_data and len(table_data) > 0:
                tables.append(TableBlock(
                    page_num=page_num,
                    data=table_data
                ))
        return tables

    def _recognize_simple_layout(self, raw_data: Dict[str, Any]) -> List[PageContent]:
        """识别简单布局（非PDF）"""
        text = raw_data.get('text', '')
        tables_raw = raw_data.get('tables', [])

        text_blocks = []
        if text:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    text_blocks.append(TextBlock(
                        page_num=1,
                        content=line,
                        column=0
                    ))

        tables = []
        for table_data in tables_raw:
            if table_data:
                tables.append(TableBlock(
                    page_num=1,
                    data=table_data if isinstance(table_data, list) else []
                ))

        return [PageContent(
            page_num=1,
            text_blocks=text_blocks,
            tables=tables,
            layout_type=LayoutType.SINGLE_COLUMN,
            columns=1
        )]


# ==================== 3. 跨页内容连接模块 ====================

class CrossPageConnector:
    """跨页内容连接器 - 合并跨页段落和表格"""

    def __init__(self):
        self._current_paragraph = ""
        self._current_paragraph_pages = []
        self._stats = {
            'paragraphs_merged': 0,
            'tables_merged': 0
        }

    def connect(self, pages_content: List[PageContent], verbose: bool = False) -> List[PageContent]:
        """连接跨页内容"""
        if not pages_content:
            return pages_content

        # 重置统计
        self._stats = {'paragraphs_merged': 0, 'tables_merged': 0}

        # 连接跨页段落
        pages_content = self._connect_paragraphs(pages_content, verbose)

        # 连接跨页表格
        pages_content = self._connect_tables(pages_content, verbose)

        if verbose:
            print(f"  跨页连接统计:")
            print(f"    合并段落数: {self._stats['paragraphs_merged']}")
            print(f"    合并表格数: {self._stats['tables_merged']}")

        return pages_content

    def _connect_paragraphs(self, pages_content: List[PageContent], verbose: bool = False) -> List[PageContent]:
        """连接跨页段落"""
        self._current_paragraph = ""
        self._current_paragraph_pages = []

        for page in pages_content:
            new_text_blocks = []

            for block in page.text_blocks:
                content = block.content

                is_paragraph_end = self._is_paragraph_end(content)
                is_paragraph_start = self._is_paragraph_start(content)

                if self._current_paragraph:
                    if is_paragraph_start:
                        # 上一页段落结束，保存（这是一个跨页段落）
                        if len(self._current_paragraph_pages) > 1:
                            self._stats['paragraphs_merged'] += 1
                            if verbose:
                                pages_range = f"{self._current_paragraph_pages[0][0]}-{self._current_paragraph_pages[-1][0]}"
                                print(f"    跨页段落: 页 {pages_range}")

                        merged_text = ' '.join([c for _, c in self._current_paragraph_pages])
                        new_text_blocks.append(TextBlock(
                            page_num=self._current_paragraph_pages[0][0],
                            content=merged_text,
                            column=0
                        ))
                        self._current_paragraph = content
                        self._current_paragraph_pages = [(page.page_num, content)]
                    else:
                        # 继续当前段落
                        self._current_paragraph += " " + content
                        self._current_paragraph_pages.append((page.page_num, content))
                else:
                    self._current_paragraph = content
                    self._current_paragraph_pages = [(page.page_num, content)]

                if is_paragraph_end and self._current_paragraph:
                    # 段落结束在当页
                    if len(self._current_paragraph_pages) > 1:
                        self._stats['paragraphs_merged'] += 1
                        if verbose:
                            pages_range = f"{self._current_paragraph_pages[0][0]}-{self._current_paragraph_pages[-1][0]}"
                            print(f"    跨页段落: 页 {pages_range}")

                    merged_text = ' '.join([c for _, c in self._current_paragraph_pages])
                    new_text_blocks.append(TextBlock(
                        page_num=self._current_paragraph_pages[0][0],
                        content=merged_text,
                        column=0
                    ))
                    self._current_paragraph = ""
                    self._current_paragraph_pages = []

            page.text_blocks = new_text_blocks

        # 处理最后剩余的段落
        if self._current_paragraph:
            if len(self._current_paragraph_pages) > 1:
                self._stats['paragraphs_merged'] += 1
                if verbose:
                    pages_range = f"{self._current_paragraph_pages[0][0]}-{self._current_paragraph_pages[-1][0]}"
                    print(f"    跨页段落: 页 {pages_range}")

            merged_text = ' '.join([c for _, c in self._current_paragraph_pages])
            if pages_content:
                pages_content[-1].text_blocks.append(TextBlock(
                    page_num=self._current_paragraph_pages[0][0],
                    content=merged_text,
                    column=0
                ))

        return pages_content

    def _connect_tables(self, pages_content: List[PageContent], verbose: bool = False) -> List[PageContent]:
        """连接跨页表格"""
        for i in range(len(pages_content) - 1):
            current_page = pages_content[i]
            next_page = pages_content[i + 1]

            if not current_page.tables or not next_page.tables:
                continue

            # 检查是否需要合并表格
            for table_idx, (current_table, next_table) in enumerate(zip(current_page.tables, next_page.tables)):
                if self._should_merge_tables(current_table, next_table):
                    # 记录跨页表格信息
                    self._stats['tables_merged'] += 1
                    if verbose:
                        print(f"    跨页表格: 第 {current_table.page_num} 页 -> 第 {next_table.page_num} 页")
                        if current_table.data and next_table.data:
                            print(f"      表头: {current_table.data[0][:5] if len(current_table.data[0]) > 5 else current_table.data[0]}")
                            print(f"      合并前行数: {len(current_table.data)} + {len(next_table.data)}")

                    # 跳过重复的表头
                    start_row = 1 if self._has_same_header(current_table.data[0], next_table.data[0]) else 0

                    if start_row > 0 and verbose:
                        print(f"      跳过了 {start_row} 行重复表头")

                    # 合并数据
                    current_table.data.extend(next_table.data[start_row:])
                    current_table.is_continued = True

                    if verbose:
                        print(f"      合并后行数: {len(current_table.data)}")

                    # 标记为已合并，后续从下一页移除
                    next_page.tables[table_idx] = None

            # 移除已合并的表格
            next_page.tables = [t for t in next_page.tables if t is not None]

        # 额外检查：统计所有表格的跨页情况
        total_tables = 0
        cross_page_tables = 0
        for page in pages_content:
            for table in page.tables:
                total_tables += 1
                if table.is_continued:
                    cross_page_tables += 1

        if verbose and total_tables > 0:
            print(f"  表格统计: 共 {total_tables} 个表格，其中 {cross_page_tables} 个跨页表格")

        return pages_content

    def _should_merge_tables(self, table1: TableBlock, table2: TableBlock) -> bool:
        """判断两个表格是否应该合并"""
        if not table1.data or not table2.data:
            return False
        return self._has_same_header(table1.data[0], table2.data[0])

    def _has_same_header(self, header1: List[str], header2: List[str]) -> bool:
        """判断两个表头是否相同"""
        if not header1 or not header2:
            return False
        if len(header1) != len(header2):
            return False

        # 清理空白后比较
        header1_clean = [str(h).strip() if h else "" for h in header1]
        header2_clean = [str(h).strip() if h else "" for h in header2]

        return header1_clean == header2_clean

    def _is_paragraph_start(self, line: str) -> bool:
        """判断是否是新段落的开始"""
        if not line:
            return False

        patterns = [
            r'^\s+',
            r'^[A-Z]',
            r'^\d+\.',
            r'^[一二三四五六七八九十]、',
            r'^第[一二三四五六七八九十]+章',
        ]

        for pattern in patterns:
            if re.match(pattern, line):
                return True
        return False

    def _is_paragraph_end(self, line: str) -> bool:
        """判断是否是段落结束"""
        if not line:
            return True

        # 以句号、感叹号、问号等结束
        if line and line[-1] in '。！？.!?':
            return True

        return False

    def get_stats(self) -> Dict[str, int]:
        """获取连接统计信息"""
        return self._stats.copy()


# ==================== 4. 文档解析器（整合全部流程）====================

class DocumentParser:
    """
    文档解析器 - 整合全部处理流程
    顺序：数据加载 -> 布局识别 -> 连接跨页内容 -> 数据清洗
    """

    def __init__(self, clean_config: Optional[Dict] = None):
        """
        初始化解析器

        Args:
            clean_config: 清洗配置参数
        """
        self.data_loader = DataLoader()
        self.layout_recognizer = LayoutRecognizer()
        self.cross_page_connector = CrossPageConnector()

        # 初始化清洗组件
        self.cleaner = DataCleaner(clean_config)
        self.table_cleaner = TableCleaner()
        self.noise_filter = NoiseFilter()

        # 使用 CleaningPipeline 进行批量处理
        self.cleaning_pipeline = CleaningPipeline(clean_config)

    def parse(
        self,
        file_path: str,
        from_page: int = 0,
        to_page: int = 100000,
        enable_cleaning: bool = True,
        verbose: bool = False
    ) -> ParsedDocument:
        """
        解析文档

        处理顺序：
        1. 数据加载 - 加载文档原始内容
        2. 布局识别 - 识别分栏、表格位置
        3. 连接跨页内容 - 合并跨页段落和表格
        4. 数据清洗 - 清洗文本、过滤噪声（使用 CleaningPipeline）

        Args:
            file_path: 文件路径
            from_page: 起始页
            to_page: 结束页
            enable_cleaning: 是否启用清洗
            verbose: 是否打印详细信息

        Returns:
            ParsedDocument 对象
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"开始解析文档: {os.path.basename(file_path)}")
            print(f"{'=' * 60}")

        # 步骤1: 数据加载
        if verbose:
            print("\n[1/4] 数据加载...")
        raw_data = self.data_loader.load(file_path, from_page, to_page)
        if verbose:
            print(f"  文件类型: {raw_data['file_type']}")
            print(f"  总页数: {raw_data['total_pages']}")

        # 步骤2: 布局识别
        if verbose:
            print("\n[2/4] 布局识别...")
        pages_content = self.layout_recognizer.recognize(raw_data)
        if verbose:
            total_blocks = sum(len(p.text_blocks) for p in pages_content)
            total_tables = sum(len(p.tables) for p in pages_content)
            print(f"  文本块数: {total_blocks}")
            print(f"  表格数: {total_tables}")

        # 步骤3: 连接跨页内容
        if verbose:
            print("\n[3/4] 连接跨页内容...")
        pages_content = self.cross_page_connector.connect(pages_content, verbose=verbose)
        if verbose:
            total_blocks = sum(len(p.text_blocks) for p in pages_content)
            print(f"  连接后文本块数: {total_blocks}")

        # 步骤4: 数据清洗
        if verbose:
            print("\n[4/4] 数据清洗...")

        # 提取文本
        all_text = self._extract_text_from_pages(pages_content)
        cleaned_text = all_text

        if enable_cleaning:
            # 方式1：使用 CleaningPipeline 处理
            doc_for_pipeline = {
                'content_with_weight': all_text,
                'content_type': raw_data.get('file_type', 'text')
            }

            # 通过 pipeline 处理
            processed_doc = self.cleaning_pipeline.process(doc_for_pipeline)
            cleaned_text = processed_doc.get('content_with_weight', all_text)

            # 方式2：额外使用 NoiseFilter 过滤噪声行
            lines = cleaned_text.split('\n')
            filtered_lines = [line for line in lines if not self.noise_filter.is_noise_line(line)]
            cleaned_text = '\n'.join(filtered_lines)

            # 方式3：清洗表格数据（如果有表格）
            for page in pages_content:
                for table in page.tables:
                    table.data = self.table_cleaner.clean_table_data(table.data)

        if verbose:
            print(f"  原始文本长度: {len(all_text)} 字符")
            print(f"  清洗后长度: {len(cleaned_text)} 字符")
            if len(all_text) > 0:
                reduction = len(all_text) - len(cleaned_text)
                print(f"  减少: {reduction} 字符 ({reduction/len(all_text)*100:.1f}%)")

        return ParsedDocument(
            file_path=file_path,
            file_name=raw_data['file_name'],
            file_type=raw_data['file_type'],
            pages=pages_content,
            total_pages=len(pages_content),
            cleaned_text=cleaned_text
        )

    def _extract_text_from_pages(self, pages_content: List[PageContent]) -> str:
        """从页面内容中提取文本"""
        texts = []
        for page in pages_content:
            page_texts = [block.content for block in page.text_blocks if block.content.strip()]
            if page_texts:
                texts.append('\n'.join(page_texts))
        return '\n\n'.join(texts)

    def parse_to_text(self, file_path: str, from_page: int = 0, to_page: int = 100000, **kwargs) -> str:
        """解析文档并返回清洗后的纯文本"""
        result = self.parse(file_path, from_page, to_page, **kwargs)
        return result.cleaned_text

    def parse_simple(self, file_path: str) -> str:
        """简单解析（不进行清洗和布局识别）"""
        raw_data = self.data_loader.load(file_path)
        return raw_data.get('text', '')


# ==================== 便捷函数 ====================

def parse_document(file_path: str, enable_cleaning: bool = True, verbose: bool = False, **kwargs) -> ParsedDocument:
    """快速解析文档"""
    parser = DocumentParser()
    return parser.parse(file_path, enable_cleaning=enable_cleaning, verbose=verbose, **kwargs)


def parse_document_to_text(file_path: str, enable_cleaning: bool = True, verbose: bool = False, **kwargs) -> str:
    """快速解析文档为纯文本"""
    parser = DocumentParser()
    return parser.parse_to_text(file_path, enable_cleaning=enable_cleaning, verbose=verbose, **kwargs)


def clean_text(text: str, config: Optional[Dict] = None) -> str:
    """快速清洗文本"""
    cleaner = DataCleaner(config)
    return cleaner.clean_text(text)


# ==================== 导出 ====================

__all__ = [
    # 数据结构
    'TextBlock',
    'TableBlock',
    'PageContent',
    'ParsedDocument',
    'LayoutType',
    
    # 模块类
    'DataLoader',
    'LayoutRecognizer',
    'CrossPageConnector',
    'DocumentParser',
    
    # 便捷函数
    'parse_document',
    'parse_document_to_text',
    'clean_text',
]


# ==================== 演示 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("RAG1 - 文档读取与数据清洗模块")
    print("处理顺序: 数据加载 -> 布局识别 -> 连接跨页内容 -> 数据清洗")
    print("=" * 60)

    # 测试清洗器
    cleaner = DataCleaner()
    test_text = "  这是一个  测试文本  \n\n第 1 页\n第二行内容\n\n   "
    cleaned = cleaner.clean_text(test_text)
    print(f"\n清洗测试:")
    print(f"  原始: '{test_text[:50]}...'")
    print(f"  清洗后: '{cleaned[:50]}...'")

    # 测试 CleaningPipeline
    print(f"\nCleaningPipeline 测试:")
    pipeline = CleaningPipeline()
    test_doc = {
        'content_with_weight': "  这是一个测试  \n\n第 1 页\n内容\n\n  ",
        'content_type': 'text'
    }
    processed = pipeline.process(test_doc)
    print(f"  处理后: '{processed['content_with_weight'][:50]}...'")

    # 测试 PDF 解析
    pdf_file = "【兴证电子】世运电路2023中报点评.pdf"

    if os.path.exists(pdf_file):
        print(f"\n测试 PDF 解析: {pdf_file}")
        print("-" * 40)

        result = parse_document(pdf_file, from_page=0, to_page=5, enable_cleaning=True, verbose=True)

        print(f"\n解析结果:")
        print(f"  文件名: {result.file_name}")
        print(f"  文件类型: {result.file_type}")
        print(f"  总页数: {result.total_pages}")
        print(f"  清洗后文本长度: {len(result.cleaned_text)} 字符")

        # 显示前500字符
        if result.cleaned_text:
            print(f"\n文本预览（前500字符）:")
            print("-" * 40)
            print(result.cleaned_text[:500])
            print("-" * 40)
    else:
        print(f"\n测试文件不存在: {pdf_file}")
        print("请将 PDF 文件放在当前目录下")

    print("\n" + "=" * 60)
    print("模块加载完成")
    print("=" * 60)