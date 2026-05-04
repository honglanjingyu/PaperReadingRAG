# app/service/core/deepdoc/parser/__init__.py
"""
文档解析器模块
"""

from .pdf_parser import PlainParser, RAGFlowPdfParser
from .docx_parser import RAGFlowDocxParser
from .txt_parser import RAGFlowTxtParser
from .excel_parser import RAGFlowExcelParser
from .html_parser import RAGFlowHtmlParser
from .json_parser import RAGFlowJsonParser
from .markdown_parser import RAGFlowMarkdownParser
from .ppt_parser import RAGFlowPptParser
from .remote_pdf_parser import RemotePDFParser, parse_pdf_remote, is_remote_parse_enabled

__all__ = [
    'PlainParser',
    'RAGFlowPdfParser',
    'RAGFlowDocxParser',
    'RAGFlowTxtParser',
    'RAGFlowExcelParser',
    'RAGFlowHtmlParser',
    'RAGFlowJsonParser',
    'RAGFlowMarkdownParser',
    'RAGFlowPptParser',
    'RemotePDFParser',
    'parse_pdf_remote',
    'is_remote_parse_enabled',
]