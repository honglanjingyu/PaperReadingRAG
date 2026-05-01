# app/service/core/deepdoc/parser/txt_parser.py
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


class RAGFlowTxtParser:
    """纯文本文件解析器 (TXT, PY, JS, JAVA, C, CPP, etc.)"""

    def __call__(self, fnm, binary=None, chunk_token_num=128, delimiter="\n!?;。；！？"):
        """解析文本文件"""
        txt = get_text(fnm, binary)
        return self.parser_txt(txt, chunk_token_num, delimiter)

    @classmethod
    def parser_txt(cls, txt, chunk_token_num=128, delimiter="\n!?;。；！？"):
        """文本分块处理"""
        # 按分隔符分割
        dels = "|".join([re.escape(d) for d in delimiter])
        secs = re.split(r"(%s)" % dels, txt)

        cks = [""]
        tk_nums = [0]

        for sec in secs:
            tnum = num_tokens_from_string(sec)
            if tk_nums[-1] + tnum > chunk_token_num and cks[-1]:
                cks.append(sec)
                tk_nums.append(tnum)
            else:
                cks[-1] += sec
                tk_nums[-1] += tnum

        return [[c, ""] for c in cks if c]