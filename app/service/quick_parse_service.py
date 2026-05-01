# app/service/quick_parse_service.py
# 支持快速解析的轻量级服务

class QuickParseService:
    """快速文档解析服务 - 用于小文档快速处理"""

    def __init__(self):
        self.supported_formats = ['docx', 'pdf', 'txt']
        self.max_pages = 4  # PDF 最大页数
        self.max_characters = 4000  # 最大字符数

    def parse_docx(self, file_content: bytes):
        """解析 DOCX 文件"""
        doc = Document(BytesIO(file_content))
        text = '\n'.join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
        return text, len(text)

    def parse_pdf(self, file_content: bytes):
        """解析 PDF 文件（限制页数）"""
        with pdfplumber.open(BytesIO(file_content)) as pdf:
            if len(pdf.pages) > self.max_pages:
                raise HTTPException(400, f"PDF页数({len(pdf.pages)})超过限制({self.max_pages})")
            text = '\n'.join([page.extract_text() or '' for page in pdf.pages])
        return text, len(pdf.pages)

    def parse_txt(self, file_content: bytes):
        """解析 TXT 文件（自动检测编码）"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'ascii']
        for encoding in encodings:
            try:
                content = file_content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        return content, len(content)


# 在 quick_parse_service.py 中添加清洗功能

from app.service.core.cleaner.data_cleaner import DataCleaner, TableCleaner, NoiseFilter


class QuickParseService:
    """快速文档解析服务 - 增强版，包含数据清洗"""

    def __init__(self):
        self.supported_formats = ['docx', 'pdf', 'txt']
        self.max_pages = 4
        self.max_characters = 4000
        # 初始化清洗器
        self.cleaner = DataCleaner({
            'remove_empty_lines': True,
            'normalize_whitespace': True,
            'min_line_length': 2,
            'remove_special_chars': False,  # 保留部分特殊字符
            'language': 'auto'
        })
        self.noise_filter = NoiseFilter()

    def parse_docx(self, file_content: bytes) -> tuple:
        """解析 DOCX 文件并清洗"""
        doc = Document(BytesIO(file_content))
        text = '\n'.join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
        # 清洗文本
        text = self.cleaner.clean_text(text)
        return text, len(text)

    def parse_pdf(self, file_content: bytes) -> tuple:
        """解析 PDF 文件并清洗"""
        with pdfplumber.open(BytesIO(file_content)) as pdf:
            if len(pdf.pages) > self.max_pages:
                raise HTTPException(400, f"PDF页数({len(pdf.pages)})超过限制({self.max_pages})")
            text = '\n'.join([page.extract_text() or '' for page in pdf.pages])
            # 清洗文本
            text = self.cleaner.clean_text(text)
        return text, len(pdf.pages)