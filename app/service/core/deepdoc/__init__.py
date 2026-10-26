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

    # 远程解析不可用时可本地直读的纯文本类型
    _LOCAL_TEXT_EXT = {'.txt', '.md', '.markdown'}

    def _parse_local_text(self, file_path: str):
        """本地直读纯文本文件，返回 (sections, total_pages)"""
        for enc in ('utf-8', 'utf-8-sig', 'gbk', 'latin-1'):
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    text = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            logger.error(f"本地读取失败（编码不支持）: {file_path}")
            return [], 1
        text = text or ''
        return ([(text, 'text')] if text.strip() else []), 1

    def parse(self, file_path: str, from_page: int = 0, to_page: int = 100000,
              enable_cleaning: bool = True, verbose: bool = False):
        """
        兼容 parse 方法，返回 ParsedDocument 对象

        远程解析器不可用时，对 txt/md 等纯文本文件回退到本地直读。
        """
        ext = os.path.splitext(file_path)[1].lower()
        if not self._parser.is_available() and ext in self._LOCAL_TEXT_EXT:
            logger.warning(f"远程解析不可用，回退本地直读: {file_path}")
            sections, total_pages = self._parse_local_text(file_path)
            return ParsedDocument(file_path, sections, total_pages)
        sections, tables, total_pages = self._parser.parse_document(file_path, from_page, to_page)
        return ParsedDocument(file_path, sections, total_pages)

    def parse_to_text(self, file_path: str, from_page: int = 0, to_page: int = 100000, **kwargs) -> str:
        """兼容 parse_to_text 方法"""
        parsed = self.parse(file_path, from_page, to_page)
        return parsed.cleaned_text


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