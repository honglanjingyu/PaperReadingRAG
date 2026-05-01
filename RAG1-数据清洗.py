# RAG1-复杂PDF解析.py
"""
使用 RAGFlow 的 PlainParser 解析复杂 PDF
增强版：包含数据清洗功能
"""

import os
import sys
import re
import unicodedata
import html
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
import pdfplumber
import pandas as pd

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ==================== 数据清洗模块 ====================

class DataCleaner:
    """数据清洗器 - 处理各种文档类型的文本清洗"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化清洗器

        Args:
            config: 配置参数
                - remove_empty_lines: 是否移除空行 (默认 True)
                - remove_special_chars: 是否移除特殊字符 (默认 True)
                - normalize_whitespace: 是否标准化空白字符 (默认 True)
                - max_line_length: 最大行长度 (默认 None)
                - min_line_length: 最小行长度 (默认 2)
                - remove_urls: 是否移除URL (默认 False)
                - remove_emails: 是否移除邮箱 (默认 False)
                - remove_numbers: 是否移除数字 (默认 False)
                - language: 语言 ('zh', 'en', 'auto') 默认 'auto'
        """
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

        # 噪声模式（页码、页眉页脚等）
        self.noise_patterns = [
            re.compile(r'^\s*\d+\s*$'),  # 纯数字
            re.compile(r'^\s*第\s*\d+\s*页\s*$'),  # 第X页
            re.compile(r'^\s*Page\s+\d+\s*$', re.IGNORECASE),  # Page X
            re.compile(r'^\s*\d+\s*/\s*\d+\s*$'),  # X/Y
            re.compile(r'^\s*Copyright\s+©?\s*\d{4}\s*$', re.IGNORECASE),
            re.compile(r'^\s*All\s+Rights\s+Reserved\s*$', re.IGNORECASE),
            re.compile(r'^\s*Confidential\s*$', re.IGNORECASE),
            re.compile(r'^[-=_*]{10,}$'),  # 分隔线
            re.compile(r'^[─━]{10,}$'),
        ]

    def clean_text(self, text: str, verbose: bool = False) -> str:
        """
        主清洗函数 - 执行所有配置的清洗操作

        Args:
            text: 原始文本
            verbose: 是否打印清洗信息

        Returns:
            清洗后的文本
        """
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

        # 7. 移除数字（可选）
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
            print(
                f"  清洗统计: {original_length} -> {len(cleaned_text)} 字符 (减少 {original_length - len(cleaned_text)} 字符)")

        return cleaned_text

    def _normalize_unicode(self, text: str) -> str:
        """标准化 Unicode 字符"""
        text = unicodedata.normalize('NFKC', text)
        text = text.replace('\u3000', ' ')  # 全角空格转半角
        return text

    def _remove_control_chars(self, text: str) -> str:
        """移除控制字符"""
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    def _normalize_whitespace(self, text: str) -> str:
        """标准化空白字符"""
        text = re.sub(r'[ \t]+', ' ', text)
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines)

    def _remove_urls(self, text: str) -> str:
        """移除URL"""
        return self.url_pattern.sub('', text)

    def _remove_emails(self, text: str) -> str:
        """移除邮箱地址"""
        return self.email_pattern.sub('', text)

    def _remove_special_chars(self, text: str) -> str:
        """移除特殊字符"""
        return self.special_chars_pattern.sub('', text)

    def _remove_numbers(self, text: str) -> str:
        """移除纯数字"""
        return re.sub(r'\b\d+(?:\.\d+)?\b', '', text)

    def _clean_line(self, line: str) -> Optional[str]:
        """清理单行文本"""
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
        """清洗表格数据"""
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
        """将表格转换为 Markdown 格式"""
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
        """判断一行是否为噪声内容"""
        if not line:
            return True

        line_lower = line.lower()
        noise_count = sum(1 for kw in self.noise_keywords if kw.lower() in line_lower)
        if noise_count / max(len(line) / 20, 1) > threshold:
            return True

        if re.match(r'^[\W_]+$', line):
            return True

        return False


# ==================== PDF 解析器 ====================

def pdf2_read(buf):
    """读取 PDF 文件"""
    return pdfplumber.open(buf)


class PlainParser:
    """PDF 简单解析器 - 仅提取纯文本"""

    def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
        lines = []
        with open(filename, 'rb') as f:
            pdf = pdf2_read(BytesIO(f.read()))
            pages = pdf.pages[from_page:to_page] if to_page <= len(pdf.pages) else pdf.pages[from_page:]
            for page in pages:
                text = page.extract_text()
                if text:
                    lines.extend([t for t in text.split("\n") if t.strip()])
        return [(line, "") for line in lines], []


class EnhancedPDFParser:
    """增强版 PDF 解析器 - 包含数据清洗功能"""

    def __init__(self, clean_config: Optional[Dict] = None):
        """
        初始化增强解析器

        Args:
            clean_config: 清洗配置参数
        """
        self.text_cleaner = DataCleaner(clean_config)
        self.table_cleaner = TableCleaner()
        self.noise_filter = NoiseFilter()

    def parse(self, pdf_path: str, from_page: int = 0, to_page: int = 100000,
              enable_cleaning: bool = True, verbose: bool = True) -> Tuple[List, List, Dict]:
        """
        解析 PDF 并可选清洗数据

        Args:
            pdf_path: PDF 文件路径
            from_page: 起始页
            to_page: 结束页
            enable_cleaning: 是否启用数据清洗
            verbose: 是否打印详细信息

        Returns:
            (sections, tables, stats): 文本段落、表格数据、统计信息
        """
        print(f"\n{'=' * 60}")
        print(f"增强版 PDF 解析: {pdf_path}")
        print(f"数据清洗: {'启用' if enable_cleaning else '禁用'}")
        print(f"{'=' * 60}")

        if not os.path.exists(pdf_path):
            print(f"错误: 文件不存在 - {pdf_path}")
            return [], [], {}

        file_size = os.path.getsize(pdf_path) / 1024
        print(f"文件大小: {file_size:.2f} KB")

        all_sections = []
        all_tables = []
        stats = {
            'total_pages': 0,
            'text_pages': 0,
            'table_pages': 0,
            'total_tables': 0,
            'original_chars': 0,
            'cleaned_chars': 0,
            'filtered_lines': 0
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
                if text:
                    stats['text_pages'] += 1
                    lines = text.split('\n')

                    for line in lines:
                        line = line.strip()
                        if line:
                            if enable_cleaning:
                                # 应用清洗
                                cleaned_line = self.text_cleaner.clean_text(line)
                                stats['original_chars'] += len(line)
                                stats['cleaned_chars'] += len(cleaned_line)

                                # 过滤噪声
                                if not self.noise_filter.is_noise_line(cleaned_line):
                                    if cleaned_line:
                                        all_sections.append((cleaned_line, ""))
                                else:
                                    stats['filtered_lines'] += 1
                            else:
                                all_sections.append((line, ""))

                    if verbose:
                        print(f"  提取文本: {len(lines)} 行, {len(text)} 字符")
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

                            # 清洗表格数据
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

        return all_sections, all_tables, stats


def parse_pdf_with_tables(pdf_path: str, from_page: int = 0, to_page: int = 100000):
    """
    使用 pdfplumber 解析 PDF，支持表格提取（原始版本）
    """
    print(f"\n{'=' * 60}")
    print(f"解析 PDF: {pdf_path}")
    print(f"{'=' * 60}")

    if not os.path.exists(pdf_path):
        print(f"错误: 文件不存在 - {pdf_path}")
        return [], []

    file_size = os.path.getsize(pdf_path) / 1024
    print(f"文件大小: {file_size:.2f} KB")

    all_sections = []
    all_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"总页数: {total_pages}")

        start_page = max(0, from_page)
        end_page = min(total_pages, to_page)
        print(f"解析页范围: 第 {start_page + 1} - {end_page} 页")

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
                print(f"  文本预览:")
                for i, line in enumerate(lines[:3]):
                    if line.strip():
                        print(f"    {i + 1}. {line.strip()[:80]}...")
            else:
                print(f"  未提取到文本")

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
                            all_tables.append({
                                'page': page_num + 1,
                                'table_index': i,
                                'data': table
                            })
            else:
                print(f"  未找到表格")

    return all_sections, all_tables


def parse_pdf_with_plainparser(pdf_path: str, from_page: int = 0, to_page: int = 100000):
    """使用 PlainParser 解析 PDF（仅文本，更快）"""
    print(f"\n{'=' * 60}")
    print(f"使用 PlainParser 解析: {pdf_path}")
    print(f"{'=' * 60}")

    parser = PlainParser()

    try:
        sections, tables = parser(pdf_path, from_page=from_page, to_page=to_page)
        print(f"\n解析结果:")
        print(f"  - 文本行数: {len(sections)}")
        print(f"  - 表格数量: {len(tables)}")

        if sections:
            print(f"\n文本预览（前10行）:")
            for i, (text, style) in enumerate(sections[:10]):
                if text.strip():
                    print(f"  {i + 1}. {text.strip()[:100]}...")

        return sections, tables

    except Exception as e:
        print(f"PlainParser 解析失败: {e}")
        return [], []


def extract_all_tables_to_dataframes(pdf_path: str):
    """提取 PDF 中所有表格并转换为 DataFrame"""
    print(f"\n{'=' * 60}")
    print(f"提取所有表格: {pdf_path}")
    print(f"{'=' * 60}")

    all_dataframes = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables):
                if table and len(table) > 1:
                    try:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        all_dataframes.append({
                            'page': page_num + 1,
                            'table': table_idx + 1,
                            'dataframe': df
                        })
                        print(f"\n第 {page_num + 1} 页, 表格 {table_idx + 1}:")
                        print(f"形状: {df.shape}")
                        print(df.to_string())
                        print("-" * 40)
                    except Exception as e:
                        print(f"表格转换失败: {e}")

    return all_dataframes


def save_results(sections, tables, output_path: str, cleaned: bool = False):
    """保存解析结果到文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"PDF 解析结果{' (已清洗)' if cleaned else ''}\n")
        f.write("=" * 60 + "\n\n")

        f.write("【文本内容】\n")
        f.write("-" * 40 + "\n")
        if sections:
            for i, (text, style) in enumerate(sections):
                if text.strip():
                    f.write(f"\n段落 {i + 1}:\n")
                    f.write(f"{text}\n")
                    f.write("-" * 40 + "\n")
        else:
            f.write("未提取到文本内容\n")

        f.write("\n\n【表格内容】\n")
        f.write("-" * 40 + "\n")
        if tables:
            for i, table_info in enumerate(tables):
                f.write(f"\n表格 {i + 1} (第 {table_info.get('page', '?')} 页):\n")
                if 'dataframe' in table_info:
                    f.write(table_info['dataframe'].to_string())
                elif 'data' in table_info:
                    table_str = TableCleaner.table_to_markdown(table_info['data'])
                    f.write(table_str if table_str else str(table_info['data']))
                f.write("\n" + "-" * 40 + "\n")
        else:
            f.write("未提取到表格\n")

    print(f"\n结果已保存到: {output_path}")


def print_summary(sections, tables, stats: Dict = None):
    """打印解析摘要"""
    print(f"\n{'=' * 60}")
    print("解析摘要")
    print(f"{'=' * 60}")

    if stats:
        print(f"\n页面统计:")
        print(f"  - 总页数: {stats.get('total_pages', 0)}")
        print(f"  - 有文本页数: {stats.get('text_pages', 0)}")
        print(f"  - 有表格页数: {stats.get('table_pages', 0)}")
        print(f"  - 表格总数: {stats.get('total_tables', 0)}")

        if stats.get('original_chars', 0) > 0:
            reduction = stats['original_chars'] - stats['cleaned_chars']
            reduction_pct = (reduction / stats['original_chars']) * 100
            print(f"\n文本清洗统计:")
            print(f"  - 原始字符数: {stats['original_chars']}")
            print(f"  - 清洗后字符数: {stats['cleaned_chars']}")
            print(f"  - 减少: {reduction} 字符 ({reduction_pct:.1f}%)")
            print(f"  - 过滤行数: {stats.get('filtered_lines', 0)}")

    if sections:
        total_chars = sum(len(text) for text, _ in sections)
        print(f"\n文本统计:")
        print(f"  - 总行数: {len(sections)}")
        print(f"  - 总字符数: {total_chars}")
        print(f"  - 平均行长度: {total_chars // len(sections) if sections else 0} 字符")

    if tables:
        print(f"\n表格统计:")
        print(f"  - 表格数量: {len(tables)}")
        pages_with_tables = set(t.get('page', 0) for t in tables)
        print(f"  - 涉及页面: {sorted(pages_with_tables)}")

        for i, table_info in enumerate(tables):
            if 'dataframe' in table_info:
                df = table_info['dataframe']
                print(f"  - 表格 {i + 1} (第{table_info['page']}页): {df.shape[0]}行 x {df.shape[1]}列")
            elif 'data' in table_info:
                data = table_info['data']
                print(f"  - 表格 {i + 1} (第{table_info['page']}页): {len(data)}行 x {len(data[0]) if data else 0}列")
    else:
        print(f"\n未提取到表格")


# ==================== 示例和使用演示 ====================

def demo_basic_cleaning():
    """演示1：基础文本清洗"""
    print("\n" + "=" * 60)
    print("演示1：基础文本清洗")
    print("=" * 60)

    cleaner = DataCleaner()

    # 测试用例
    test_texts = [
        "这是一段  包含  多个空格  和\n\n\n空行的文本。",
        "包含URL: https://example.com 和邮箱: test@example.com 的文本。",
        "第 1 页\n这是一段正常文本。\nAll Rights Reserved.\ncopyright 2024",
        "   \n\t\n   \n   "
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n测试 {i}:")
        print(f"  原始: {repr(text[:50])}...")
        cleaned = cleaner.clean_text(text, verbose=True)
        print(f"  清洗后: {repr(cleaned[:50])}...")


def demo_enhanced_parser(pdf_path: str):
    """演示2：使用增强版解析器（带数据清洗）"""
    print("\n" + "=" * 60)
    print("演示2：增强版 PDF 解析器（启用数据清洗）")
    print("=" * 60)

    # 创建增强解析器
    parser = EnhancedPDFParser({
        'remove_empty_lines': True,
        'normalize_whitespace': True,
        'min_line_length': 2,
        'remove_urls': True,
        'remove_emails': True,
        'language': 'auto'
    })

    # 解析并清洗
    sections, tables, stats = parser.parse(
        pdf_path,
        from_page=0,
        to_page=5,
        enable_cleaning=True,
        verbose=True
    )

    # 打印摘要
    print_summary(sections, tables, stats)

    return sections, tables, stats


def demo_comparison(pdf_path: str):
    """演示3：对比原始解析和清洗后解析"""
    print("\n" + "=" * 60)
    print("演示3：原始解析 vs 清洗后解析对比")
    print("=" * 60)

    # 原始解析（不清洗）
    print("\n>>> 原始解析（无清洗）<<<")
    sections_raw, tables_raw = parse_pdf_with_tables(pdf_path, from_page=0, to_page=3)

    # 增强解析（带清洗）
    print("\n>>> 增强解析（带清洗）<<<")
    parser = EnhancedPDFParser()
    sections_clean, tables_clean, stats = parser.parse(
        pdf_path,
        from_page=0,
        to_page=3,
        enable_cleaning=True,
        verbose=True
    )

    # 对比结果
    print("\n" + "-" * 40)
    print("对比结果:")
    print("-" * 40)
    print(f"原始解析 - 文本行数: {len(sections_raw)}")
    print(f"清洗解析 - 文本行数: {len(sections_clean)}")
    print(f"过滤/清洗掉: {len(sections_raw) - len(sections_clean)} 行")

    if stats.get('original_chars', 0) > 0:
        reduction = stats['original_chars'] - stats['cleaned_chars']
        print(f"字符减少: {reduction} ({reduction / max(stats['original_chars'], 1) * 100:.1f}%)")


def demo_table_cleaning():
    """演示4：表格清洗"""
    print("\n" + "=" * 60)
    print("演示4：表格数据清洗")
    print("=" * 60)

    # 示例表格数据（包含脏数据）
    dirty_table = [
        [" 姓名  ", "  年龄  ", "  职位  "],
        [" 张三  ", "  25  ", "  工程师  "],
        ["  李四  ", "  30  ", "  经理  "],
        ["  ", None, "  实习生  "],
    ]

    print("\n原始表格:")
    for row in dirty_table:
        print(f"  {row}")

    # 清洗表格
    cleaner = TableCleaner()
    cleaned_table = cleaner.clean_table_data(dirty_table)

    print("\n清洗后表格:")
    for row in cleaned_table:
        print(f"  {row}")

    # 转换为 Markdown
    markdown = cleaner.table_to_markdown(cleaned_table)
    print("\nMarkdown 格式:")
    print(markdown)


def demo_noise_filter():
    """演示5：噪声过滤"""
    print("\n" + "=" * 60)
    print("演示5：噪声内容过滤")
    print("=" * 60)

    filter_tool = NoiseFilter()

    test_lines = [
        "这是正常内容",
        "第 1 页",
        "版权所有 2024",
        "confidential",
        "---",
        "正常内容继续"
    ]

    print("\n噪声检测结果:")
    for line in test_lines:
        is_noise = filter_tool.is_noise_line(line)
        status = "❌ 噪声" if is_noise else "✅ 正常"
        print(f"  {status}: {line}")


def demo_custom_config():
    """演示6：自定义清洗配置"""
    print("\n" + "=" * 60)
    print("演示6：自定义清洗配置")
    print("=" * 60)

    # 不同场景的配置
    configs = {
        "严格模式": {
            'remove_empty_lines': True,
            'remove_special_chars': True,
            'normalize_whitespace': True,
            'min_line_length': 5,
            'remove_urls': True,
            'remove_emails': True,
            'remove_numbers': True,
        },
        "宽松模式": {
            'remove_empty_lines': True,
            'remove_special_chars': False,
            'normalize_whitespace': True,
            'min_line_length': 2,
            'remove_urls': False,
            'remove_emails': False,
            'remove_numbers': False,
        },
        "技术文档模式": {
            'remove_empty_lines': True,
            'remove_special_chars': False,  # 保留代码符号
            'normalize_whitespace': True,
            'min_line_length': 1,
            'remove_urls': False,
            'remove_emails': True,
            'remove_numbers': False,
        }
    }

    test_text = "代码: if x > 0: print(x)   \n\nURL: https://example.com\n第 1 页\n"

    for mode, config in configs.items():
        print(f"\n{mode}:")
        cleaner = DataCleaner(config)
        result = cleaner.clean_text(test_text)
        print(f"  输入: {repr(test_text)}")
        print(f"  输出: {repr(result)}")


# ==================== 主函数 ====================

if __name__ == "__main__":
    # PDF 文件路径
    pdf_file = "【兴证电子】世运电路2023中报点评.pdf"

    print("=" * 60)
    print("PDF 复杂解析器 - 增强版（含数据清洗）")
    print("=" * 60)

    # 显示当前目录
    print(f"\n当前目录: {os.getcwd()}")

    # 检查文件
    if not os.path.exists(pdf_file):
        print(f"\n错误: 文件不存在 - {pdf_file}")
        print(f"请将 PDF 文件放在当前目录下，或修改 pdf_file 变量为正确路径")
        print("\n" + "-" * 40)
        print("将运行演示模式（不依赖 PDF 文件）")
        print("-" * 40)

        # 演示模式 - 不依赖 PDF 文件
        demo_basic_cleaning()
        demo_table_cleaning()
        demo_noise_filter()
        demo_custom_config()

        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)

    else:
        # 有 PDF 文件时，运行完整解析
        print(f"\n找到 PDF 文件: {pdf_file}")

        # 选择解析模式
        print("\n请选择解析模式:")
        print("  1. 原始解析（无清洗）")
        print("  2. 增强解析（带清洗）")
        print("  3. 对比模式")
        print("  4. 运行所有演示")

        choice = input("\n请输入选择 (1-4，默认 2): ").strip() or "2"

        if choice == "1":
            # 原始解析
            sections, tables = parse_pdf_with_tables(pdf_file, from_page=0, to_page=10)
            print_summary(sections, tables)
            if sections or tables:
                save_results(sections, tables, "pdf_parse_result_raw.txt", cleaned=False)

        elif choice == "2":
            # 增强解析
            parser = EnhancedPDFParser()
            sections, tables, stats = parser.parse(
                pdf_file, from_page=0, to_page=10,
                enable_cleaning=True, verbose=True
            )
            print_summary(sections, tables, stats)
            if sections or tables:
                save_results(sections, tables, "pdf_parse_result_cleaned.txt", cleaned=True)
                print(f"\n清洗前结果: pdf_parse_result_raw.txt")
                print(f"清洗后结果: pdf_parse_result_cleaned.txt")

        elif choice == "3":
            # 对比模式
            demo_comparison(pdf_file)

        else:
            # 运行所有演示
            demo_basic_cleaning()
            demo_enhanced_parser(pdf_file)
            demo_comparison(pdf_file)
            demo_table_cleaning()
            demo_noise_filter()
            demo_custom_config()

        print(f"\n{'=' * 60}")
        print("解析完成！")
        print(f"{'=' * 60}")