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


__all__ = ['DataLoader']