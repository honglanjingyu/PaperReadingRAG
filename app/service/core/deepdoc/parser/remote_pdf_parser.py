# app/service/core/deepdoc/parser/remote_pdf_parser.py
"""
远程PDF解析器 - 使用MinerU云端API解析PDF
将 MinerU 返回的 Markdown 正确解析为段落和表格
"""

import os
import re
import time
from io import BytesIO
from typing import List, Tuple, Optional, Dict, Any
import tempfile
import logging

logger = logging.getLogger(__name__)


class RemotePDFParser:
    """远程PDF解析器 - 使用MinerU API解析PDF"""

    def __init__(self, api_token: str = None):
        """初始化远程PDF解析器"""
        self.api_token = api_token or os.getenv("PARSE_API_TOKEN")
        self._client = None
        self._init_client()

    def _init_client(self):
        """初始化MinerU客户端"""
        if not self.api_token:
            logger.warning("PARSE_API_TOKEN未配置，远程PDF解析不可用")
            return

        try:
            os.environ["MINERU_TOKEN"] = self.api_token
            from mineru import MinerU

            self._client = MinerU()
            print(f"  MinerU 客户端初始化成功")
            logger.info("远程PDF解析器初始化成功")
        except ImportError as e:
            logger.error(f"导入 mineru 失败: {e}")
            self._client = None
        except Exception as e:
            logger.error(f"远程PDF解析器初始化失败: {e}")
            self._client = None

    def is_available(self) -> bool:
        """检查远程解析器是否可用"""
        return self._client is not None and self.api_token is not None

    def parse_pdf(
            self,
            file_path_or_binary,
            from_page: int = 0,
            to_page: int = 100000,
            callback=None
    ) -> Tuple[List[Tuple[str, str]], List[List[List[str]]]]:
        """
        解析PDF文件

        Returns:
            (sections, tables):
                sections: [(text, style), ...] 段落列表
                tables: 表格列表，每个表格是 List[List[str]] 二维列表
        """
        if not self.is_available():
            logger.error("远程PDF解析器不可用")
            return [], []

        try:
            # 处理文件路径
            temp_file = None
            if isinstance(file_path_or_binary, (bytes, BytesIO)):
                if isinstance(file_path_or_binary, BytesIO):
                    binary_data = file_path_or_binary.getvalue()
                else:
                    binary_data = file_path_or_binary

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                temp_file.write(binary_data)
                temp_file.close()
                file_path = temp_file.name
            else:
                file_path = file_path_or_binary

            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return [], []

            print(f"  正在调用 MinerU API...")
            print(f"  文件大小: {os.path.getsize(file_path) / 1024:.2f} KB")

            # 调用 MinerU 解析
            os.environ["MINERU_TOKEN"] = self.api_token
            result = self._client.extract(file_path)

            # 清理临时文件
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

            if result is None or not result.markdown:
                logger.error("远程解析返回空结果")
                return [], []

            markdown_content = result.markdown
            print(f"  ✓ API 调用成功，返回内容长度: {len(markdown_content)} 字符")

            # 解析 Markdown 为段落和表格
            sections, tables = self.parse_markdown(markdown_content)

            print(f"  解析完成: {len(sections)}段落, {len(tables)}表格")
            return sections, tables

        except Exception as e:
            logger.error(f"远程PDF解析失败: {e}")
            print(f"  ❌ 解析失败: {e}")
            import traceback
            traceback.print_exc()
            return [], []

    def parse_markdown(self, markdown_content: str) -> Tuple[List[Tuple[str, str]], List[List[List[str]]]]:
        """
        将 MinerU 返回的 Markdown 解析为段落和表格

        Args:
            markdown_content: 完整的 Markdown 字符串

        Returns:
            (sections, tables): 段落列表和表格列表
        """
        if not markdown_content:
            return [], []

        lines = markdown_content.split('\n')
        total_lines = len(lines)

        print(f"  开始解析 Markdown，共 {total_lines} 行")

        # 第一步：提取所有表格及其占用的行范围
        tables = []
        table_ranges = []  # (start_line, end_line, table_index)

        i = 0
        while i < total_lines:
            line = lines[i].strip()

            # 检测 Markdown 表格（以 | 开头和结尾）
            if line.startswith('|') and line.endswith('|'):
                start = i
                table_lines = []

                # 收集连续的表格行
                while i < total_lines:
                    current = lines[i].strip()
                    if current.startswith('|') and current.endswith('|'):
                        table_lines.append(current)
                        i += 1
                    else:
                        break

                # 解析表格
                if len(table_lines) >= 2:
                    table_data = self._parse_md_table(table_lines)
                    if table_data and len(table_data) > 0:
                        tables.append(table_data)
                        table_ranges.append((start, i, len(tables) - 1))
                        print(
                            f"    表格 {len(tables)}: {len(table_data)} 行 x {len(table_data[0])} 列 (行 {start + 1}-{i})")
                continue

            # 检测 HTML 表格
            elif '<table' in line.lower():
                start = i
                html_lines = []

                # 收集到 </table>
                while i < total_lines:
                    html_lines.append(lines[i])
                    if '</table>' in lines[i].lower():
                        i += 1
                        break
                    i += 1

                # 解析 HTML 表格
                html_content = '\n'.join(html_lines)
                table_data = self._parse_html_table(html_content)
                if table_data and len(table_data) > 0:
                    tables.append(table_data)
                    table_ranges.append((start, i, len(tables) - 1))
                    print(
                        f"    表格 {len(tables)}: {len(table_data)} 行 x {len(table_data[0])} 列 (行 {start + 1}-{i})")
                continue

            else:
                i += 1

        # 第二步：解析段落（跳过表格行）
        sections = []
        current_paragraph = []

        i = 0
        while i < total_lines:
            # 检查是否在表格范围内
            in_table = False
            for start, end, _ in table_ranges:
                if start <= i < end:
                    in_table = True
                    break

            if in_table:
                i += 1
                continue

            line = lines[i]
            stripped = line.strip()

            # 空行：结束当前段落
            if not stripped:
                if current_paragraph:
                    text = ' '.join(current_paragraph).strip()
                    if text:
                        sections.append((text, "paragraph"))
                    current_paragraph = []
                i += 1
                continue

            # 标题行（以 # 开头）
            if stripped.startswith('#'):
                if current_paragraph:
                    text = ' '.join(current_paragraph).strip()
                    if text:
                        sections.append((text, "paragraph"))
                    current_paragraph = []

                # 解析标题级别
                level = 0
                for ch in stripped:
                    if ch == '#':
                        level += 1
                    else:
                        break
                level = min(level, 6)
                title = stripped[level:].strip()
                # 清理标题中的 Markdown 标记
                title = self._clean_text(title)
                if title:
                    sections.append((title, f"heading_{level}"))
                i += 1
                continue

            # 普通文本行
            cleaned = self._clean_text(stripped)
            if cleaned:
                current_paragraph.append(cleaned)
            i += 1

        # 处理最后一段
        if current_paragraph:
            text = ' '.join(current_paragraph).strip()
            if text:
                sections.append((text, "paragraph"))

        print(f"  解析结果: {len(sections)}段落, {len(tables)}表格")
        return sections, tables

    def _parse_md_table(self, table_lines: List[str]) -> List[List[str]]:
        """
        解析 Markdown 格式的表格

        Args:
            table_lines: Markdown 表格行，如：
                "| 列1 | 列2 | 列3 |"
                "| --- | --- | --- |"
                "| 值1 | 值2 | 值3 |"

        Returns:
            二维列表
        """
        if not table_lines or len(table_lines) < 2:
            return []

        result = []

        for line_idx, line in enumerate(table_lines):
            # 分割单元格
            cells = line.split('|')
            # 去掉首尾空元素（因为行以 | 开头和结尾）
            cells = cells[1:-1]
            # 清理每个单元格
            cells = [c.strip() for c in cells]

            # 跳过分隔行（包含 --- 或 :--- 的行）
            if line_idx == 1 and all(self._is_separator(c) for c in cells):
                continue

            # 过滤掉全空的行
            if not any(c for c in cells):
                continue

            # 清理单元格内容
            cleaned_row = []
            for cell in cells:
                cleaned = self._clean_cell(cell)
                cleaned_row.append(cleaned)

            if cleaned_row:
                result.append(cleaned_row)

        # 确保每行列数一致
        if result:
            max_cols = max(len(row) for row in result)
            for row in result:
                while len(row) < max_cols:
                    row.append("")

        return result

    def _is_separator(self, cell: str) -> bool:
        """判断是否为表格分隔行单元格"""
        if not cell:
            return False
        # 移除空白和冒号
        cleaned = cell.replace(' ', '').replace(':', '')
        return all(c == '-' for c in cleaned)

    def _parse_html_table(self, html_content: str) -> List[List[str]]:
        """
        解析 HTML 格式的表格

        Args:
            html_content: HTML 表格内容

        Returns:
            二维列表
        """
        if not html_content:
            return []

        result = []

        # 提取所有行
        tr_pattern = r'<tr[^>]*>(.*?)</tr>'
        rows = re.findall(tr_pattern, html_content, re.DOTALL | re.IGNORECASE)

        for row_html in rows:
            # 提取所有单元格（td 和 th）
            cell_pattern = r'<t[dh][^>]*>(.*?)</t[dh]>'
            cells = re.findall(cell_pattern, row_html, re.DOTALL | re.IGNORECASE)

            if not cells:
                continue

            # 清理每个单元格
            cleaned_row = []
            for cell_html in cells:
                # 移除内部 HTML 标签
                text = re.sub(r'<[^>]+>', '', cell_html)
                # 清理空白
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()
                cleaned = self._clean_cell(text)
                cleaned_row.append(cleaned)

            # 跳过全空行
            if any(c for c in cleaned_row):
                result.append(cleaned_row)

        return result

    def _clean_cell(self, text: str) -> str:
        """
        清理单元格内容，移除 Markdown 格式标记

        Args:
            text: 原始单元格文本

        Returns:
            清理后的文本
        """
        if not text:
            return ""

        # 粗体 **text** 或 __text__
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)

        # 斜体 *text* 或 _text_
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)

        # 行内代码 `code`
        text = re.sub(r'`([^`]+)`', r'\1', text)

        # 链接 [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        # 移除 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)

        # 移除 LaTeX 公式 $...$
        text = re.sub(r'\$[^$]+\$', '', text)

        # 移除特殊符号
        text = text.replace('\\', '').replace('*', '').replace('_', '')

        # 标准化空白
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # 限制长度
        if len(text) > 2000:
            text = text[:2000] + "..."

        return text

    def _clean_text(self, text: str) -> str:
        """
        清理普通文本，移除 Markdown 格式标记

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        if not text:
            return ""

        # 移除标题标记
        if text.startswith('#'):
            text = re.sub(r'^#+\s*', '', text)

        # 移除粗体
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)

        # 移除斜体
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)

        # 移除行内代码
        text = re.sub(r'`([^`]+)`', r'\1', text)

        # 移除链接
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        # 移除 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)

        # 标准化空白
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text


def is_remote_parse_enabled() -> bool:
    """检查远程解析是否启用"""
    enabled = os.getenv("ENABLE_REMOTE_PARSE", "false").lower() == "true"
    has_token = bool(os.getenv("PARSE_API_TOKEN"))
    return enabled and has_token


def parse_pdf_remote(
        file_path_or_binary,
        from_page: int = 0,
        to_page: int = 100000,
        api_token: str = None
) -> Tuple[List[Tuple[str, str]], List[List[List[str]]]]:
    """使用远程API解析PDF"""
    parser = RemotePDFParser(api_token)
    return parser.parse_pdf(file_path_or_binary, from_page, to_page)


__all__ = [
    'RemotePDFParser',
    'parse_pdf_remote',
    'is_remote_parse_enabled',
]