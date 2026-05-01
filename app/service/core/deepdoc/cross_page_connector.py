# app/service/core/deepdoc/cross_page_connector.py
"""
跨页内容连接模块 - 合并跨页段落和表格
"""

import re
from typing import List, Dict, Tuple

from .models import PageContent, TextBlock, TableBlock


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

                    # 跳过重复的表头
                    start_row = 1 if self._has_same_header(current_table.data[0], next_table.data[0]) else 0

                    # 合并数据
                    current_table.data.extend(next_table.data[start_row:])
                    current_table.is_continued = True

                    # 标记为已合并，后续从下一页移除
                    next_page.tables[table_idx] = None

            # 移除已合并的表格
            next_page.tables = [t for t in next_page.tables if t is not None]

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


__all__ = ['CrossPageConnector']