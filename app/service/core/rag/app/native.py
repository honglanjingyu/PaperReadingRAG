# app/service/core/rag/app/naive.py - chunk函数中定义的支持格式

# 支持的文件格式：
# .docx, .pdf, .xlsx/.xls, .txt, .md/.markdown, .htm/.html, .json, .doc

if re.search(r"\.docx$", filename, re.IGNORECASE):
# DOCX 文档处理

elif re.search(r"\.pdf$", filename, re.IGNORECASE):
# PDF 文档处理

elif re.search(r"\.xlsx?$", filename, re.IGNORECASE):
# Excel 文档处理

elif re.search(r"\.(txt|py|js|java|c|cpp|h|php|go|ts|sh|cs|kt|sql)$", filename, re.IGNORECASE):
# 纯文本文件处理

elif re.search(r"\.(md|markdown)$", filename, re.IGNORECASE):
# Markdown 文档处理

elif re.search(r"\.(htm|html)$", filename, re.IGNORECASE):
# HTML 文档处理

elif re.search(r"\.json$", filename, re.IGNORECASE):
# JSON 文档处理

elif re.search(r"\.doc$", filename, re.IGNORECASE):
# 旧版 Word 文档处理（使用 tika）
# app/service/core/rag/app/naive.py - chunk 函数中的 .doc 处理

elif re.search(r"\.doc$", filename, re.IGNORECASE):
    """旧版 Word 文档（使用 Apache Tika）"""
    callback(0.1, "Start to parse.")
    from tika import parser
    from io import BytesIO

    binary = BytesIO(binary)
    doc_parsed = parser.from_buffer(binary)

    if doc_parsed.get('content', None) is not None:
        sections = doc_parsed['content'].split('\n')
        sections = [(_, "") for _ in sections if _]
    else:
        logging.warning(f"tika.parser got empty content from {filename}.")
        return []


# app/service/core/rag/app/naive.py - 主入口函数

def chunk(filename, binary=None, from_page=0, to_page=100000,
          lang="Chinese", callback=None, **kwargs):
    """
    统一的文档分块入口函数
    支持格式: docx, pdf, xlsx, txt, md, html, json, doc

    Returns:
        list: 处理后的文档块，每个块包含内容、分词、向量等信息
    """
    parser_config = kwargs.get("parser_config", {
        "chunk_token_num": 128,
        "delimiter": "\n!?。；！？"
    })

    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
    }

    # 根据文件扩展名选择解析器
    if re.search(r"\.docx$", filename, re.IGNORECASE):
        # DOCX 处理
        sections, tables = Docx()(filename, binary)
        res = tokenize_table(tables, doc, is_english)
        chunks, images = naive_merge_docx(sections, chunk_token_num, delimiter)
        res.extend(tokenize_chunks_docx(chunks, doc, is_english, images))

    elif re.search(r"\.pdf$", filename, re.IGNORECASE):
        # PDF 处理
        pdf_parser = Pdf() if parser_config.get("layout_recognize") == "DeepDOC" else PlainParser()
        sections, tables = pdf_parser(filename, binary, from_page=from_page, to_page=to_page, callback=callback)
        res = tokenize_table(tables, doc, is_english)

    elif re.search(r"\.xlsx?$", filename, re.IGNORECASE):
        # Excel 处理
        excel_parser = ExcelParser()
        if binary is None:
            with open(filename, 'rb') as f:
                binary = f.read()
        sections = [(_, "") for _ in excel_parser(binary) if _]

    elif re.search(r"\.(txt|py|js|java|c|cpp|h|php|go|ts|sh|cs|kt|sql)$", filename, re.IGNORECASE):
        # 纯文本处理
        sections = TxtParser()(filename, binary, chunk_token_num, delimiter)

    elif re.search(r"\.(md|markdown)$", filename, re.IGNORECASE):
        # Markdown 处理
        sections, tables = Markdown(chunk_token_num)(filename, binary)
        res = tokenize_table(tables, doc, is_english)

    elif re.search(r"\.(htm|html)$", filename, re.IGNORECASE):
        # HTML 处理
        sections = HtmlParser()(filename, binary)
        sections = [(_, "") for _ in sections if _]

    elif re.search(r"\.json$", filename, re.IGNORECASE):
        # JSON 处理
        sections = JsonParser(chunk_token_num)(binary)
        sections = [(_, "") for _ in sections if _]

    elif re.search(r"\.doc$", filename, re.IGNORECASE):
        # 旧版 Word 处理 (Tika)
        binary = BytesIO(binary)
        doc_parsed = parser.from_buffer(binary)
        if doc_parsed.get('content'):
            sections = [(_, "") for _ in doc_parsed['content'].split('\n') if _]

    else:
        raise NotImplementedError(f"File type not supported: {filename}")

    # 文本分块
    chunks = naive_merge(sections, chunk_token_num, delimiter)
    res.extend(tokenize_chunks(chunks, doc, is_english, pdf_parser))

    return res


# app/service/core/rag/app/naive.py

from service.core.chunking import ChunkManager, ChunkProcessor


def chunk_with_manager(filename, binary=None, from_page=0, to_page=100000,
                       lang="Chinese", callback=None, **kwargs):
    """
    使用分块管理器的统一分块入口
    """
    parser_config = kwargs.get("parser_config", {
        "chunk_token_num": 128,
        "delimiter": "\n!?。；！？",
        "strategy": "recursive"  # 新增策略配置
    })

    # 初始化分块管理器
    chunk_manager = ChunkManager({
        'chunk_token_num': parser_config.get('chunk_token_num', 128),
        'strategy': parser_config.get('strategy', 'recursive')
    })

    # ... 原有的解析逻辑 ...

    # 使用分块管理器进行分块
    chunks = chunk_manager.naive_merge(sections,
                                       chunk_token_num,
                                       delimiter)

    # 或者使用 ChunkProcessor 进行清洗+分块
    processor = ChunkProcessor({
        'chunk_token_num': chunk_token_num,
        'strategy': parser_config.get('strategy', 'recursive')
    })

    chunks = processor.process_sections(sections)

    return res