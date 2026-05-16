# app/service/core/deepdoc/__init__.py

import os

from .cleaner import (
    DataCleaner,
    HTMLCleaner,
    TableCleaner,
    NoiseFilter,
    CleaningPipeline,
    clean_document_content,
)

from .remote_parser import (
    RemoteDocumentParser,
    parse_document_remote,
    is_remote_parse_enabled,
    save_chunked_report,
)


# ========== 兼容数据类 ==========
class ParsedDocument:
    """兼容旧版 ParsedDocument 的数据类"""

    def __init__(self, file_path: str, sections: list, total_pages: int = 1):
        self.file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1][1:] if '.' in file_path else 'txt'
        self.file_type = 'pdf' if file_ext == 'pdf' else file_ext
        self.total_pages = total_pages
        self.cleaned_text = "\n".join([text for text, _ in sections]) if sections else ""
        self._sections = sections


# ========== 兼容类：模拟旧的 DocumentParser 接口 ==========
class DocumentParser:
    """兼容类 - 模拟旧版 DocumentParser 接口，内部使用 RemoteDocumentParser"""

    def __init__(self):
        self._parser = RemoteDocumentParser()

    def parse(self, file_path: str, from_page: int = 0, to_page: int = 100000,
              enable_cleaning: bool = True, verbose: bool = False):
        """
        兼容 parse 方法，返回 ParsedDocument 对象

        注意：enable_cleaning 和 verbose 参数被忽略（远程解析已包含清洗）
        """
        sections, tables, total_pages = self._parser.parse_document(file_path, from_page, to_page)
        return ParsedDocument(file_path, sections, total_pages)

    def parse_to_text(self, file_path: str, from_page: int = 0, to_page: int = 100000, **kwargs) -> str:
        """兼容 parse_to_text 方法"""
        sections, _, _ = self._parser.parse_document(file_path, from_page, to_page)
        return "\n".join([text for text, _ in sections]) if sections else ""


# 导出兼容类
__all__ = [
    # 清洗器
    'DataCleaner',
    'HTMLCleaner',
    'TableCleaner',
    'NoiseFilter',
    'CleaningPipeline',
    'clean_document_content',
    # 远程解析器
    'RemoteDocumentParser',
    'parse_document_remote',
    'is_remote_parse_enabled',
    'save_chunked_report',
    # 兼容类
    'DocumentParser',
    'ParsedDocument',
]