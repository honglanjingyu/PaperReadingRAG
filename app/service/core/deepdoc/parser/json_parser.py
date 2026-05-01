# app/service/core/deepdoc/parser/json_parser.py
import json
import re
from typing import Dict, List, Any


# 添加缺失的函数
def find_codec(binary):
    """检测二进制数据的编码"""
    try:
        import chardet
        result = chardet.detect(binary)
        return result['encoding'] or 'utf-8'
    except:
        return 'utf-8'


class RAGFlowJsonParser:
    """JSON 文档解析器"""

    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size * 2
        self.min_chunk_size = max(max_chunk_size - 200, 50)

    def __call__(self, binary):
        """解析 JSON 文件，支持大 JSON 分块"""
        encoding = find_codec(binary)
        txt = binary.decode(encoding, errors="ignore")
        json_data = json.loads(txt)
        chunks = self.split_json(json_data, True)
        sections = [json.dumps(line, ensure_ascii=False) for line in chunks if line]
        return sections

    def split_json(self, json_data, convert_lists: bool = False):
        """将大 JSON 按大小分块"""
        if convert_lists:
            preprocessed_data = self._list_to_dict_preprocessing(json_data)
            chunks = self._json_split(preprocessed_data)
        else:
            chunks = self._json_split(json_data)
        return chunks

    def _list_to_dict_preprocessing(self, data):
        """将列表转换为字典预处理"""
        if isinstance(data, dict):
            return {k: self._list_to_dict_preprocessing(v) for k, v in data.items()}
        elif isinstance(data, list):
            return {f"item_{i}": self._list_to_dict_preprocessing(item) for i, item in enumerate(data)}
        else:
            return data

    def _json_split(self, data, current_depth=0):
        """递归分割JSON数据"""
        if isinstance(data, (str, int, float, bool)) or data is None:
            return [data]

        if isinstance(data, dict):
            chunks = []
            current_chunk = {}
            current_size = 0

            for key, value in data.items():
                value_str = json.dumps(value, ensure_ascii=False)
                if current_size + len(value_str) > self.max_chunk_size and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = {}
                    current_size = 0

                current_chunk[key] = value
                current_size += len(value_str)

            if current_chunk:
                chunks.append(current_chunk)

            return chunks

        return [data]