# app/service/core/deepdoc/parser/html_parser.py
import readability
import html_text
from .utils import find_codec, get_text

class RAGFlowHtmlParser:
    """HTML 文档解析器"""

    def __call__(self, fnm, binary=None):
        """解析 HTML 文件"""
        if binary:
            encoding = find_codec(binary)
            txt = binary.decode(encoding, errors="ignore")
        else:
            with open(fnm, "r", encoding='utf-8') as f:
                txt = f.read()
        return self.parser_txt(txt)

    @classmethod
    def parser_txt(cls, txt):
        """使用 readability 提取主要内容"""
        html_doc = readability.Document(txt)
        title = html_doc.title()
        content = html_text.extract_text(html_doc.summary(html_partial=True))
        txt = f"{title}\n{content}"
        return txt.split("\n")