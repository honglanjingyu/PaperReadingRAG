# app/service/core/deepdoc/parser/utils.py
import re

def find_codec(binary):
    """检测二进制数据的编码"""
    try:
        import chardet
        result = chardet.detect(binary)
        return result['encoding'] or 'utf-8'
    except:
        return 'utf-8'

def get_text(fnm: str, binary=None) -> str:
    """读取文本文件内容，支持编码自动检测"""
    if binary:
        encoding = find_codec(binary)
        txt = binary.decode(encoding, errors="ignore")
    else:
        with open(fnm, "r", encoding='utf-8') as f:
            txt = f.read()
    return txt

def num_tokens_from_string(text):
    """简单估算 token 数量"""
    return len(text) // 2