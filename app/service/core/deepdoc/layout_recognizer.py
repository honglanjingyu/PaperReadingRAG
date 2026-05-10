# app/service/core/deepdoc/layout_recognizer.py
"""布局识别模块 - 简化版，只处理远程解析结果"""

from typing import List, Dict, Any, Tuple
from collections import Counter

from .models import PageContent, LayoutType, TextBlock, TableBlock


class LayoutRecognizer:
    """布局识别器 - 简化版"""

    def __init__(self):
        pass

    def recognize(self, raw_data: Dict[str, Any]) -> List[PageContent]:
        """识别文档布局"""
        file_type = raw_data.get('file_type')
        is_remote = raw_data.get('parse_method') == 'remote'

        if file_type == 'pdf' and is_remote:
            return self._recognize_remote_layout(raw_data)
        else:
            return self._recognize_simple_layout(raw_data)

    def _recognize_remote_layout(self, raw_data: Dict[str, Any]) -> List[PageContent]:
        """识别远程解析的 PDF 布局"""
        pages_content = []
        pages_raw = raw_data.get('pages_raw', [])

        for page_raw in pages_raw:
            page_num = page_raw.get('page_num', 1)
            tables_raw = page_raw.get('tables', [])

            # 使用远程解析返回的 text_blocks
            text_blocks = []
            if page_raw.get('text_blocks'):
                for block in page_raw['text_blocks']:
                    text_blocks.append(TextBlock(
                        page_num=page_num,
                        content=block.get('text', block.get('content', '')),
                        column=block.get('column', 0),
                        x0=block.get('x0', 0),
                        y0=block.get('y0', 0),
                        x1=block.get('x1', 0),
                        y1=block.get('y1', 0)
                    ))

            # 提取表格
            tables = self._extract_table_blocks(tables_raw, page_num)

            pages_content.append(PageContent(
                page_num=page_num,
                text_blocks=text_blocks,
                tables=tables,
                layout_type=LayoutType.SINGLE_COLUMN,
                columns=1
            ))

        return pages_content

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
        """识别简单布局（非PDF或非远程解析）"""
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