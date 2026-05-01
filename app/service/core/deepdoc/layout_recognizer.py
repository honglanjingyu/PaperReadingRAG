# app/service/core/deepdoc/layout_recognizer.py
"""
布局识别模块 - 识别文档布局（分栏、表格位置等）
"""

from typing import List, Dict, Any, Tuple
from collections import Counter

from .models import PageContent, LayoutType, TextBlock, TableBlock


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

        lines = {}
        for char in chars:
            y = round(char.get('y0', 0) / 5) * 5
            if y not in lines:
                lines[y] = []
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


__all__ = ['LayoutRecognizer']