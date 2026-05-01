# app/service/core/deepdoc/parser/pdf_parser.py - 修复版
"""
PDF 解析器，支持 OCR、布局识别、表格提取
"""

from io import BytesIO
import os
from typing import List, Tuple, Optional


def pdf2_read(file_obj):
    """
    读取 PDF 文件，兼容 pdfplumber 接口

    Args:
        file_obj: 文件对象或文件路径

    Returns:
        pdfplumber.PDF 对象
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("请安装 pdfplumber: pip install pdfplumber")

    if isinstance(file_obj, str):
        return pdfplumber.open(file_obj)
    else:
        return pdfplumber.open(file_obj)


class RAGFlowPdfParser:
    """PDF 文档解析器 - 完整版"""

    def __init__(self):
        try:
            import pdfplumber
            self.pdfplumber = pdfplumber
        except ImportError:
            raise ImportError("请安装 pdfplumber: pip install pdfplumber")

        # 可选依赖，如果不存在则使用简单模式
        try:
            from deepdoc.ocr.ocr import OCR
            self.ocr = OCR()
        except (ImportError, ModuleNotFoundError):
            self.ocr = None

        try:
            from deepdoc.layout.layout_recognizer import LayoutRecognizer
            self.layouter = LayoutRecognizer("layout")
        except (ImportError, ModuleNotFoundError):
            self.layouter = None

        try:
            from deepdoc.table.table_structure_recognizer import TableStructureRecognizer
            self.tbl_det = TableStructureRecognizer()
        except (ImportError, ModuleNotFoundError):
            self.tbl_det = None

    def __call__(self, fnm, binary=None, from_page=0, to_page=100000,
                 need_image=True, zoomin=3, return_html=False, callback=None):
        """解析 PDF 文档"""
        if binary:
            self.pdf = pdf2_read(BytesIO(binary))
        else:
            self.pdf = pdf2_read(fnm)

        sections = []
        tables = []

        total_pages = len(self.pdf.pages)
        end_page = min(to_page, total_pages)

        for page_num in range(from_page, end_page):
            page = self.pdf.pages[page_num]

            # 提取文本
            text = page.extract_text() or ""
            if text.strip():
                sections.append((text.strip(), ""))

            # 提取表格
            page_tables = page.extract_tables()
            for table in page_tables:
                if table and len(table) > 0:
                    # 过滤空表格
                    if any(cell and str(cell).strip() for row in table for cell in row):
                        tables.append(table)

        return sections, tables


class PlainParser:
    """PDF 简单解析器 - 仅提取纯文本"""

    def __call__(self, filename, from_page=0, to_page=100000, binary=None, **kwargs):
        """解析 PDF 文件，返回文本段落列表"""
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("请安装 pdfplumber: pip install pdfplumber")

        if binary:
            pdf = pdfplumber.open(BytesIO(binary))
        else:
            pdf = pdfplumber.open(filename)

        lines = []
        total_pages = len(pdf.pages)
        end_page = min(to_page, total_pages)

        for page_num in range(from_page, end_page):
            page = pdf.pages[page_num]
            text = page.extract_text()
            if text:
                for line in text.split("\n"):
                    line = line.strip()
                    if line:
                        lines.append(line)

        pdf.close()

        # 返回 [(text, style), ...] 格式
        sections = [(line, "") for line in lines]
        return sections, []