# app/service/core/deepdoc/loader.py
"""
数据加载模块 - 加载文档原始内容
"""

import os
import re
from typing import Dict, Any, List

from .parser.pdf_parser import PlainParser
from .parser.docx_parser import RAGFlowDocxParser
from .parser.txt_parser import RAGFlowTxtParser
from .parser.excel_parser import RAGFlowExcelParser

# 导入远程PDF解析器
from .parser.remote_pdf_parser import RemotePDFParser, is_remote_parse_enabled


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

        # 初始化远程PDF解析器
        self.remote_pdf_parser = RemotePDFParser()
        self.use_remote_parse = is_remote_parse_enabled()

        if self.use_remote_parse:
            print(f"[DataLoader] 远程PDF解析已启用")
        else:
            print(f"[DataLoader] 使用本地PDF解析")

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
        """加载 PDF - 支持本地和远程解析"""

        # 判断是否使用远程解析
        if self.use_remote_parse and self.remote_pdf_parser.is_available():
            print(f"  使用远程API解析PDF: {os.path.basename(file_path)}")
            return self._load_pdf_remote(file_path, from_page, to_page)
        else:
            print(f"  使用本地解析PDF: {os.path.basename(file_path)}")
            return self._load_pdf_local(file_path, from_page, to_page)

    def _load_pdf_local(self, file_path: str, from_page: int, to_page: int) -> Dict[str, Any]:
        """本地解析PDF"""
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
            'pages_raw': pages_raw,
            'parse_method': 'local'
        }

    # app/service/core/deepdoc/loader.py

    def _load_pdf_remote(self, file_path: str, from_page: int, to_page: int) -> Dict[str, Any]:
        """远程API解析PDF"""
        try:
            # 调用远程解析器
            sections, tables = self.remote_pdf_parser.parse_pdf(
                file_path,
                from_page=from_page,
                to_page=to_page
            )

            # 构建 text_blocks 和完整文本
            text_blocks_for_pages = []
            all_text_parts = []

            for idx, (section, style) in enumerate(sections):
                if section and section.strip():
                    all_text_parts.append(section)
                    text_blocks_for_pages.append({
                        'text': section,
                        'style': style,
                        'x0': 0, 'y0': idx * 100,  # 模拟位置
                        'x1': 0, 'y1': idx * 100 + 50,
                        'column': 0
                    })

            full_text = '\n'.join(all_text_parts)
            print(f"  提取文本长度: {len(full_text)} 字符")
            print(f"  文本块数量: {len(text_blocks_for_pages)}")

            # 转换表格格式
            formatted_tables = []
            for table in tables:
                if table and len(table) > 0:
                    formatted_tables.append(table)

            # 构建页面数据
            pages_raw = [{
                'page_num': 1,
                'text': full_text,
                'chars': [],
                'words': [],
                'width': 0,
                'height': 0,
                'tables': formatted_tables,
                'images': [],
                'text_blocks': text_blocks_for_pages
            }]

            result = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_type': 'pdf',
                'total_pages': 1,
                'pages_raw': pages_raw,
                'parse_method': 'remote',
                'remote_sections': sections,
                'remote_tables': tables
            }

            return result

        except Exception as e:
            print(f"  远程PDF解析失败: {e}，回退到本地解析")
            import traceback
            traceback.print_exc()
            return self._load_pdf_local(file_path, from_page, to_page)

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


__all__ = ['DataLoader']