# app/service/core/deepdoc/document_parser.py
"""
文档解析器 - 整合全部处理流程
顺序：数据加载 -> 布局识别 -> 连接跨页内容 -> 数据清洗
"""

import os
from typing import List, Dict, Any, Optional

from .models import ParsedDocument, PageContent
from .loader import DataLoader
from .layout_recognizer import LayoutRecognizer
from .cross_page_connector import CrossPageConnector
from .cleaner import DataCleaner, TableCleaner, NoiseFilter, CleaningPipeline


class DocumentParser:
    """
    文档解析器 - 整合全部处理流程
    顺序：数据加载 -> 布局识别 -> 连接跨页内容 -> 数据清洗
    """

    def __init__(self, clean_config: Optional[Dict] = None):
        """
        初始化解析器

        Args:
            clean_config: 清洗配置参数
        """
        self.data_loader = DataLoader()
        self.layout_recognizer = LayoutRecognizer()
        self.cross_page_connector = CrossPageConnector()

        # 初始化清洗组件
        self.cleaner = DataCleaner(clean_config)
        self.table_cleaner = TableCleaner()
        self.noise_filter = NoiseFilter()

        # 使用 CleaningPipeline 进行批量处理
        self.cleaning_pipeline = CleaningPipeline(clean_config)

    def parse(self, file_path: str, from_page: int = 0, to_page: int = 100000,
              enable_cleaning: bool = True, verbose: bool = False) -> ParsedDocument:
        """
        解析文档

        处理顺序：
        1. 数据加载 - 加载文档原始内容
        2. 布局识别 - 识别分栏、表格位置
        3. 连接跨页内容 - 合并跨页段落和表格
        4. 数据清洗 - 清洗文本、过滤噪声

        Args:
            file_path: 文件路径
            from_page: 起始页
            to_page: 结束页
            enable_cleaning: 是否启用清洗
            verbose: 是否打印详细信息

        Returns:
            ParsedDocument 对象
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"开始解析文档: {os.path.basename(file_path)}")
            print(f"{'=' * 60}")

        # 步骤1: 数据加载
        if verbose:
            print("\n[1/12] 数据加载...")
        raw_data = self.data_loader.load(file_path, from_page, to_page)
        is_remote = raw_data.get('parse_method') == 'remote'
        if verbose:
            print(f"  文件类型: {raw_data['file_type']}")
            print(f"  总页数: {raw_data['total_pages']}")

        # 步骤2: 布局识别
        if verbose:
            print("\n[2/12] 布局识别...")
        pages_content = self.layout_recognizer.recognize(raw_data)
        if verbose:
            total_blocks = sum(len(p.text_blocks) for p in pages_content)
            total_tables = sum(len(p.tables) for p in pages_content)
            print(f"  文本块数: {total_blocks}")
            print(f"  表格数: {total_tables}")

        # 步骤3: 连接跨页内容
        if verbose:
            print("\n[3/12] 连接跨页内容...")
        pages_content = self.cross_page_connector.connect(pages_content, verbose=verbose, is_remote=is_remote)

        # 步骤4: 数据清洗
        if verbose:
            print("\n[4/12] 数据清洗...")

        # 提取文本
        all_text = self._extract_text_from_pages(pages_content)
        cleaned_text = all_text

        if enable_cleaning:
            # 方式1：使用 CleaningPipeline 处理
            doc_for_pipeline = {
                'content_with_weight': all_text,
                'content_type': raw_data.get('file_type', 'text')
            }

            # 通过 pipeline 处理
            processed_doc = self.cleaning_pipeline.process(doc_for_pipeline)
            cleaned_text = processed_doc.get('content_with_weight', all_text)

            # 方式2：额外使用 NoiseFilter 过滤噪声行
            lines = cleaned_text.split('\n')
            filtered_lines = [line for line in lines if not self.noise_filter.is_noise_line(line)]
            cleaned_text = '\n'.join(filtered_lines)

            # 方式3：清洗表格数据（如果有表格）
            for page in pages_content:
                for table in page.tables:
                    table.data = self.table_cleaner.clean_table_data(table.data)

        if verbose:
            print(f"  原始文本长度: {len(all_text)} 字符")
            print(f"  清洗后长度: {len(cleaned_text)} 字符")
            if len(all_text) > 0:
                reduction = len(all_text) - len(cleaned_text)
                print(f"  减少: {reduction} 字符 ({reduction/len(all_text)*100:.1f}%)")

        return ParsedDocument(
            file_path=file_path,
            file_name=raw_data['file_name'],
            file_type=raw_data['file_type'],
            pages=pages_content,
            total_pages=len(pages_content),
            cleaned_text=cleaned_text
        )

    def _extract_text_from_pages(self, pages_content: List[PageContent]) -> str:
        """从页面内容中提取文本"""
        texts = []
        for page in pages_content:
            page_texts = [block.content for block in page.text_blocks if block.content.strip()]
            if page_texts:
                texts.append('\n'.join(page_texts))
        return '\n\n'.join(texts)

    def parse_to_text(self, file_path: str, from_page: int = 0, to_page: int = 100000, **kwargs) -> str:
        """解析文档并返回清洗后的纯文本"""
        result = self.parse(file_path, from_page, to_page, **kwargs)
        return result.cleaned_text


__all__ = ['DocumentParser']