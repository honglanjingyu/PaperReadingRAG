# app/service/core/deepdoc/parser/markdown_parser.py
import re


class RAGFlowMarkdownParser:
    """Markdown 文档解析器"""

    def __init__(self, chunk_token_num=128):
        self.chunk_token_num = int(chunk_token_num)

    def extract_tables_and_remainder(self, markdown_text):
        """提取 Markdown 中的表格和剩余文本"""
        # 标准 Markdown 表格
        table_pattern = re.compile(
            r'(?:\n|^)(?:\|.*?\|.*?\|.*?\n)(?:\|(?:\s*[:-]+[-| :]*\s*)\|.*?\n)(?:\|.*?\|.*?\|.*?\n)+', re.VERBOSE)
        tables = table_pattern.findall(markdown_text)
        remainder = table_pattern.sub('', markdown_text)

        # 无边框表格
        no_border_table_pattern = re.compile(
            r'(?:\n|^)(?:\S.*?\|.*?\n)(?:(?:\s*[:-]+[-| :]*\s*).*?\n)(?:\S.*?\|.*?\n)+', re.VERBOSE)
        tables.extend(no_border_table_pattern.findall(remainder))
        remainder = no_border_table_pattern.sub('', remainder)

        return remainder, tables

    def __call__(self, fnm, binary=None):
        """解析 Markdown 文件"""
        if binary:
            text = binary.decode('utf-8', errors='ignore')
        else:
            with open(fnm, 'r', encoding='utf-8') as f:
                text = f.read()

        remainder, tables = self.extract_tables_and_remainder(text)

        # 将文本按行分割
        lines = remainder.split('\n')
        sections = [(line, "") for line in lines if line.strip()]

        return sections, tables