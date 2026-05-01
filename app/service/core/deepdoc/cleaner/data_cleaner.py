"""
数据清洗模块 - 统一的数据清洗处理
用于文档解析后的文本清洗、规范化、过滤等操作
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import html


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

        # 页码/页眉页脚模式（常见于PDF）
        self.noise_patterns = [
            # 页码
            re.compile(r'^\s*\d+\s*$'),
            re.compile(r'^\s*第\s*\d+\s*页\s*$'),
            re.compile(r'^\s*Page\s+\d+\s*$'),
            re.compile(r'^\s*\d+\s*/\s*\d+\s*$'),
            # 页眉页脚常见内容
            re.compile(r'^\s*Copyright\s+©?\s*\d{4}\s*$', re.IGNORECASE),
            re.compile(r'^\s*All\s+Rights\s+Reserved\s*$', re.IGNORECASE),
            re.compile(r'^\s*Confidential\s*$', re.IGNORECASE),
            re.compile(r'^\s*Draft\s*$', re.IGNORECASE),
            # 分隔线
            re.compile(r'^[-=_*]{10,}$'),
            re.compile(r'^[─━]{10,}$'),
        ]

    def clean_text(self, text: str) -> str:
        """
        主清洗函数 - 执行所有配置的清洗操作

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not text:
            return ""

        # 1. 标准化 Unicode（处理全角/半角、组合字符等）
        text = self._normalize_unicode(text)

        # 2. 移除或替换控制字符
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

        return '\n'.join(cleaned_lines)

    def _normalize_unicode(self, text: str) -> str:
        """标准化 Unicode 字符"""
        # NFKC 标准化：处理全角/半角、兼容字符等
        text = unicodedata.normalize('NFKC', text)

        # 全角空格转半角
        text = text.replace('\u3000', ' ')

        # 全角数字/字母转半角
        fullwidth_patterns = [
            (re.compile(r'[\uFF01-\uFF5E]'), lambda m: chr(ord(m.group(0)) - 0xfee0)),
        ]
        for pattern, repl in fullwidth_patterns:
            text = pattern.sub(repl, text)

        return text

    def _remove_control_chars(self, text: str) -> str:
        """移除控制字符（保留换行和制表符）"""
        # 移除除了 \n, \r, \t 以外的控制字符
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    def _normalize_whitespace(self, text: str) -> str:
        """标准化空白字符"""
        # 将多个连续空格替换为单个空格
        text = re.sub(r'[ \t]+', ' ', text)
        # 移除行首行尾空格（每行单独处理）
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines)

    def _remove_urls(self, text: str) -> str:
        """移除URL"""
        return self.url_pattern.sub('[URL]', text)

    def _remove_emails(self, text: str) -> str:
        """移除邮箱地址"""
        return self.email_pattern.sub('[EMAIL]', text)

    def _remove_special_chars(self, text: str) -> str:
        """移除特殊字符"""
        return self.special_chars_pattern.sub('', text)

    def _remove_numbers(self, text: str) -> str:
        """移除纯数字（保留单词内的数字）"""
        # 移除独立的数字（非字母数字混合）
        return re.sub(r'\b\d+(?:\.\d+)?\b', '', text)

    def _clean_line(self, line: str) -> Optional[str]:
        """清理单行文本"""
        # 去除首尾空白
        line = line.strip()

        if not line:
            return None if self.config['remove_empty_lines'] else ''

        # 检查最小长度
        if len(line) < self.config['min_line_length']:
            return None

        # 检查最大长度（截断）
        if self.config['max_line_length'] and len(line) > self.config['max_line_length']:
            line = line[:self.config['max_line_length']]

        # 检查是否为噪声内容
        for pattern in self.noise_patterns:
            if pattern.match(line):
                return None

        return line

    def clean_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        清洗整个文档对象

        Args:
            document: 文档字典，包含 content_with_weight, content_ltks 等字段

        Returns:
            清洗后的文档字典
        """
        if 'content_with_weight' in document:
            document['content_with_weight_original'] = document['content_with_weight']
            document['content_with_weight'] = self.clean_text(document['content_with_weight'])

        if 'content_ltks' in document:
            # 重新生成分词（将在调用方处理）
            document['content_ltks_original'] = document['content_ltks']

        return document


class HTMLCleaner:
    """HTML 内容清洗器"""

    @staticmethod
    def clean_html(html_content: str) -> str:
        """
        清理 HTML 内容，提取纯文本

        Args:
            html_content: HTML 字符串

        Returns:
            清洗后的纯文本
        """
        if not html_content:
            return ""

        # 使用 BeautifulSoup 解析
        soup = BeautifulSoup(html_content, 'html.parser')

        # 移除脚本和样式
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        # 提取文本
        text = soup.get_text(separator='\n')

        # 清理空白行
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        return '\n'.join(lines)

    @staticmethod
    def unescape_html(text: str) -> str:
        """反转义 HTML 实体"""
        return html.unescape(text)


class TableCleaner:
    """表格数据清洗器"""

    @staticmethod
    def clean_table_data(table_data: List[List[str]]) -> List[List[str]]:
        """
        清洗表格数据

        Args:
            table_data: 表格数据（二维列表）

        Returns:
            清洗后的表格数据
        """
        if not table_data:
            return []

        cleaned_table = []
        for row in table_data:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append('')
                else:
                    # 清洗单元格内容
                    cleaned = str(cell).strip()
                    # 移除多余空白
                    cleaned = re.sub(r'\s+', ' ', cleaned)
                    cleaned_row.append(cleaned)
            cleaned_table.append(cleaned_row)

        return cleaned_table

    @staticmethod
    def table_to_markdown(table_data: List[List[str]]) -> str:
        """
        将表格转换为 Markdown 格式

        Args:
            table_data: 表格数据

        Returns:
            Markdown 表格字符串
        """
        if not table_data or len(table_data) < 2:
            return ""

        lines = []
        # 表头
        header = "| " + " | ".join(str(cell) for cell in table_data[0]) + " |"
        lines.append(header)
        # 分隔线
        separator = "| " + " | ".join(["---"] * len(table_data[0])) + " |"
        lines.append(separator)
        # 数据行
        for row in table_data[1:]:
            line = "| " + " | ".join(str(cell) for cell in row) + " |"
            lines.append(line)

        return "\n".join(lines)


class NoiseFilter:
    """噪声内容过滤器"""

    def __init__(self):
        # 噪声关键词（常见于文档中的无用内容）
        self.noise_keywords = [
            '请勿转载', '版权所有', '翻印必究',
            'confidential', 'proprietary', 'all rights reserved',
            '仅供内部使用', '内部资料', '注意保密',
            'www.', 'http://', 'https://',
            '第1页', 'page 1', '图1', 'Figure 1',
            '如表1', '见上图', '详见下文',
        ]

    def is_noise_line(self, line: str, threshold: float = 0.3) -> bool:
        """
        判断一行是否为噪声内容

        Args:
            line: 文本行
            threshold: 噪声关键词比例阈值

        Returns:
            True 如果是噪声
        """
        if not line:
            return True

        line_lower = line.lower()

        # 检查是否包含噪声关键词
        noise_count = sum(1 for kw in self.noise_keywords if kw.lower() in line_lower)
        if noise_count / max(len(line) / 20, 1) > threshold:
            return True

        # 检查是否为纯符号行
        if re.match(r'^[\W_]+$', line):
            return True

        return False

    def filter_document(self, chunks: List[Dict]) -> List[Dict]:
        """
        过滤文档块中的噪声

        Args:
            chunks: 文档块列表

        Returns:
            过滤后的文档块列表
        """
        filtered = []
        for chunk in chunks:
            content = chunk.get('content_with_weight', '')
            if not self.is_noise_line(content):
                filtered.append(chunk)
        return filtered