# app/service/core/cleaner/pipeline.py - 修复版
"""
数据清洗管道 - 整合所有清洗步骤
"""

from typing import List, Dict, Any, Optional
from .data_cleaner import DataCleaner, HTMLCleaner, TableCleaner, NoiseFilter


class CleaningPipeline:
    """数据清洗管道"""

    def __init__(self, config: Optional[Dict] = None):
        # 修复：如果 config 为 None，使用空字典
        config = config or {}

        self.text_cleaner = DataCleaner(config)
        self.html_cleaner = HTMLCleaner()
        self.table_cleaner = TableCleaner()
        self.noise_filter = NoiseFilter()

        # 管道步骤配置
        self.steps = config.get('steps', ['clean_text', 'filter_noise', 'normalize'])

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个文档

        Args:
            data: 文档数据字典

        Returns:
            处理后的文档
        """
        # 1. 清洗文本内容
        if 'content_with_weight' in data:
            content = data['content_with_weight']

            # 如果是 HTML，先进行 HTML 清洗
            if data.get('content_type') == 'html':
                content = self.html_cleaner.clean_html(content)

            # 普通文本清洗
            content = self.text_cleaner.clean_text(content)
            data['content_with_weight'] = content

        # 2. 过滤噪声
        if self.noise_filter.is_noise_line(data.get('content_with_weight', '')):
            data['is_noise'] = True

        # 3. 清洗表格数据
        if 'table_data' in data:
            data['table_data'] = self.table_cleaner.clean_table_data(data['table_data'])

        return data

    def process_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量处理文档

        Args:
            documents: 文档列表

        Returns:
            处理后的文档列表（自动过滤噪声）
        """
        processed = []
        for doc in documents:
            cleaned = self.process(doc)
            if not cleaned.get('is_noise', False):
                processed.append(cleaned)
        return processed


# 便捷函数
def clean_document_content(content: str, content_type: str = 'text') -> str:
    """
    快速清洗文档内容

    Args:
        content: 原始内容
        content_type: 内容类型 ('text', 'html', 'markdown')

    Returns:
        清洗后的内容
    """
    if content_type == 'html':
        content = HTMLCleaner.clean_html(content)

    cleaner = DataCleaner()
    return cleaner.clean_text(content)