# app/service/core/deepdoc/parser/__init__.py
"""
文档解析器模块 - 只使用远程 MinerU API
"""

from .remote_pdf_parser import (
    RemotePDFParser,
    parse_document_remote,
    parse_pdf_remote,
    is_remote_parse_enabled,
    save_chunked_report,
)

__all__ = [
    'RemotePDFParser',
    'parse_document_remote',
    'parse_pdf_remote',
    'is_remote_parse_enabled',
    'save_chunked_report',
]