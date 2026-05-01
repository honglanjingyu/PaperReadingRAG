# RAG1.py - 修正版
"""
RAG 文档处理 - 文档读取与数据清洗模块

本文件现在只是一个调用入口，所有功能通过导入 app/service/core/deepdoc 模块实现

处理顺序：
1. 数据加载 - 加载多种格式文档
2. 布局识别 - 识别文档布局（左右分栏、表格、图片位置）
3. 连接跨页内容 - 合并跨页段落和表格
4. 数据清洗 - 清洗文本、过滤噪声
"""

import os
from typing import Optional, Dict

# 从 deepdoc 模块导入所有功能
from app.service.core.deepdoc import (
    # 数据结构
    TextBlock,
    TableBlock,
    PageContent,
    ParsedDocument,
    LayoutType,

    # 模块类
    DataLoader,
    LayoutRecognizer,
    CrossPageConnector,
    DocumentParser,

    # 清洗器类（重要：需要导入这些）
    DataCleaner,
    CleaningPipeline,

    # 便捷函数
    parse_document,
    parse_document_to_text,
    clean_text,
)


# 为了向后兼容，导出所有类和函数
__all__ = [
    # 数据结构
    'TextBlock',
    'TableBlock',
    'PageContent',
    'ParsedDocument',
    'LayoutType',

    # 模块类
    'DataLoader',
    'LayoutRecognizer',
    'CrossPageConnector',
    'DocumentParser',

    # 清洗器
    'DataCleaner',
    'CleaningPipeline',

    # 便捷函数
    'parse_document',
    'parse_document_to_text',
    'clean_text',
]


# ==================== 演示 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("RAG1 - 文档读取与数据清洗模块（调用入口）")
    print("处理顺序: 数据加载 -> 布局识别 -> 连接跨页内容 -> 数据清洗")
    print("=" * 60)

    # 测试清洗器
    cleaner = DataCleaner()
    test_text = "  这是一个  测试文本  \n\n第 1 页\n第二行内容\n\n   "
    cleaned = cleaner.clean_text(test_text)
    print(f"\n清洗测试:")
    print(f"  原始: '{test_text[:50]}...'")
    print(f"  清洗后: '{cleaned[:50]}...'")

    # 测试 CleaningPipeline
    print(f"\nCleaningPipeline 测试:")
    pipeline = CleaningPipeline()
    test_doc = {
        'content_with_weight': "  这是一个测试  \n\n第 1 页\n内容\n\n  ",
        'content_type': 'text'
    }
    processed = pipeline.process(test_doc)
    print(f"  处理后: '{processed['content_with_weight'][:50]}...'")

    # 测试 PDF 解析
    pdf_file = "【兴证电子】世运电路2023中报点评.pdf"

    if os.path.exists(pdf_file):
        print(f"\n测试 PDF 解析: {pdf_file}")
        print("-" * 40)

        result = parse_document(pdf_file, from_page=0, to_page=5, enable_cleaning=True, verbose=True)

        print(f"\n解析结果:")
        print(f"  文件名: {result.file_name}")
        print(f"  文件类型: {result.file_type}")
        print(f"  总页数: {result.total_pages}")
        print(f"  清洗后文本长度: {len(result.cleaned_text)} 字符")

        # 显示前500字符
        if result.cleaned_text:
            print(f"\n文本预览（前500字符）:")
            print("-" * 40)
            print(result.cleaned_text[:500])
            print("-" * 40)
    else:
        print(f"\n测试文件不存在: {pdf_file}")
        print("请将 PDF 文件放在当前目录下")

    print("\n" + "=" * 60)
    print("模块加载完成")
    print("=" * 60)