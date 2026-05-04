# app/service/core/deepdoc/__init__.py
"""
DeepDoc 文档处理模块
包含文档解析器、文档清洗器、布局识别、跨页连接等
"""

from typing import Optional, Dict

from .parser import (
    PlainParser,
    RAGFlowDocxParser,
    RAGFlowTxtParser,
    RAGFlowExcelParser,
    RAGFlowPdfParser,
    RAGFlowHtmlParser,
    RAGFlowJsonParser,
    RAGFlowMarkdownParser,
    RAGFlowPptParser,
    RemotePDFParser,
    parse_pdf_remote,
    is_remote_parse_enabled,
)

from .cleaner import (
    DataCleaner,
    HTMLCleaner,
    TableCleaner,
    NoiseFilter,
    CleaningPipeline,
)

from .models import (
    LayoutType,
    TextBlock,
    TableBlock,
    PageContent,
    ParsedDocument,
)

from .loader import DataLoader
from .layout_recognizer import LayoutRecognizer
from .cross_page_connector import CrossPageConnector
from .document_parser import DocumentParser


# 便捷函数
def parse_document(file_path: str, enable_cleaning: bool = True, verbose: bool = False, **kwargs) -> ParsedDocument:
    """快速解析文档"""
    parser = DocumentParser()
    return parser.parse(file_path, enable_cleaning=enable_cleaning, verbose=verbose, **kwargs)


def parse_document_to_text(file_path: str, enable_cleaning: bool = True, verbose: bool = False, **kwargs) -> str:
    """快速解析文档为纯文本"""
    parser = DocumentParser()
    return parser.parse_to_text(file_path, enable_cleaning=enable_cleaning, verbose=verbose, **kwargs)


def clean_text(text: str, config: Optional[Dict] = None) -> str:
    """快速清洗文本"""
    cleaner = DataCleaner(config)
    return cleaner.clean_text(text)


__all__ = [
    # 数据结构
    'LayoutType',
    'TextBlock',
    'TableBlock',
    'PageContent',
    'ParsedDocument',

    # 解析器
    'PlainParser',
    'RAGFlowDocxParser',
    'RAGFlowTxtParser',
    'RAGFlowExcelParser',
    'RAGFlowPdfParser',
    'RAGFlowHtmlParser',
    'RAGFlowJsonParser',
    'RAGFlowMarkdownParser',
    'RAGFlowPptParser',
    'RemotePDFParser',
    'parse_pdf_remote',
    'is_remote_parse_enabled',

    # 清洗器
    'DataCleaner',
    'HTMLCleaner',
    'TableCleaner',
    'NoiseFilter',
    'CleaningPipeline',

    # 模块类
    'DataLoader',
    'LayoutRecognizer',
    'CrossPageConnector',
    'DocumentParser',

    # 便捷函数
    'parse_document',
    'parse_document_to_text',
    'clean_text',
]