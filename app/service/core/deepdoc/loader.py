# app/service/core/deepdoc/loader.py
"""数据加载模块 - 只使用远程 MinerU 解析"""

import os
import re
from typing import Dict, Any, List
import logging
import asyncio
from .parser.remote_pdf_parser import RemotePDFParser, is_remote_parse_enabled

logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器 - 只使用远程 MinerU 解析 PDF"""

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

        # 初始化远程PDF解析器
        self.remote_pdf_parser = RemotePDFParser()

        if not is_remote_parse_enabled():
            logger.error("远程PDF解析未启用，请设置 ENABLE_REMOTE_PARSE=true 和 PARSE_API_TOKEN")
            raise RuntimeError("远程PDF解析未启用，MinerU API 是必需的")

        logger.info(f"[DataLoader] 远程PDF解析已启用")

    def load(self, file_path: str, from_page: int = 0, to_page: int = 100000) -> Dict[str, Any]:
        """加载文档"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()
        file_type = self.SUPPORTED_TYPES.get(file_ext, 'text')

        if file_type == 'pdf':
            return self._load_pdf_remote(file_path, from_page, to_page)
        elif file_type == 'docx':
            return self._load_docx(file_path)
        elif file_type == 'text':
            return self._load_text(file_path)
        elif file_type == 'excel':
            return self._load_excel(file_path)
        else:
            return self._load_text(file_path)

    def _load_pdf_remote(self, file_path: str, from_page: int, to_page: int) -> Dict[str, Any]:
        """远程API解析PDF"""
        try:
            logger.info(f"使用远程MinerU API解析PDF: {os.path.basename(file_path)}")

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
                        'x0': 0, 'y0': idx * 100,
                        'x1': 0, 'y1': idx * 100 + 50,
                        'column': 0
                    })

            full_text = '\n'.join(all_text_parts)
            logger.info(f"  提取文本长度: {len(full_text)} 字符")
            logger.info(f"  文本块数量: {len(text_blocks_for_pages)}")

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
            logger.error(f"远程PDF解析失败: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"PDF解析失败，请检查 MinerU API 配置: {e}")

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