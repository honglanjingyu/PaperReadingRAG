# app/service/core/deepdoc/parser/pdf_parser.py
# PDF 解析器，支持 OCR、布局识别、表格提取

class RAGFlowPdfParser:
    """PDF 文档解析器 - 完整版"""

    def __init__(self):
        self.ocr = OCR()  # OCR 识别
        self.layouter = LayoutRecognizer("layout")  # 布局识别
        self.tbl_det = TableStructureRecognizer()  # 表格结构识别

    def __call__(self, fnm, need_image=True, zoomin=3, return_html=False):
        """解析 PDF 文档"""
        self.__images__(fnm, zoomin)  # 提取图像
        self._layouts_rec(zoomin)  # 布局识别
        self._table_transformer_job(zoomin)  # 表格处理
        self._text_merge()  # 文本合并
        self._concat_downward()  # 段落连接
        return self.__filterout_scraps(deepcopy(self.boxes), zoomin), tbls


class PlainParser:
    """PDF 简单解析器 - 仅提取纯文本"""

    def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
        with open(filename, 'rb') as f:
            self.pdf = pdf2_read(BytesIO(f.read()))
            for page in self.pdf.pages[from_page:to_page]:
                lines.extend([t for t in page.extract_text().split("\n")])
        return [(line, "") for line in lines], []