"""
RAG 文档处理 - 智能分块 + 向量化
处理指定 PDF 文件： 【兴证电子】世运电路2023中报点评.pdf
支持远程API和本地Embedding模型
"""

import os
import re
import json
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import xxhash

# 加载 .env 文件
load_dotenv()

# ==================== 配置 ====================
# 从环境变量读取配置
EMBEDDING_TYPE = os.getenv("EMBEDDING_TYPE", "local")  # remote 或 local

# 远程API配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
REMOTE_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
REMOTE_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

# 本地模型配置
LOCAL_MODEL_PATH = os.getenv("LOCAL_EMBEDDING_PATH", "")
LOCAL_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# PDF处理配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))  # 分块大小（token数）
FROM_PAGE = int(os.getenv("FROM_PAGE", "0"))  # 起始页
TO_PAGE = int(os.getenv("TO_PAGE", "10")) if os.getenv("TO_PAGE") else None  # 结束页
ENABLE_VECTORIZATION = os.getenv("ENABLE_VECTORIZATION", "true").lower() == "true"

# ES配置
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "infini_rag_flow")
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "rag_documents")


# ==================== 数据结构 ====================

@dataclass
class VectorChunk:
    """带向量的分块数据结构"""
    id: str
    content: str
    vector: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    chunk_index: int = 0

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'content': self.content,
            'vector': self.vector[:10] if self.vector else [],  # 只保存前10维用于预览
            'vector_dim': len(self.vector),
            'metadata': self.metadata,
            'token_count': self.token_count,
            'chunk_index': self.chunk_index
        }

    def to_es_document(self, kb_id: str = None, doc_name: str = None) -> Dict:
        """转换为Elasticsearch文档格式"""
        doc = {
            "id": self.id,
            "content_with_weight": self.content,
            "metadata": self.metadata,
            "token_count": self.token_count,
            "created_at": datetime.now().isoformat()
        }

        if self.vector:
            doc[f"q_{len(self.vector)}_vec"] = self.vector

        if kb_id:
            doc["kb_id"] = kb_id
        if doc_name:
            doc["docnm_kwd"] = doc_name

        return doc


# ==================== Embedding 模型 ====================

class EmbeddingModel:
    """统一的Embedding模型接口"""

    def __init__(self, model_type: str = None):
        # 如果未指定，从环境变量获取
        if model_type is None:
            model_type = EMBEDDING_TYPE

        self.model_type = model_type
        self._model = None
        self._dimension = 0

        if model_type == "remote":
            self._init_remote_model()
        else:
            self._init_local_model()

    def _init_remote_model(self):
        """初始化远程API模型"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        if not DASHSCOPE_API_KEY:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

        self._client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
        self._model_name = REMOTE_MODEL_NAME
        self._dimension = REMOTE_DIMENSIONS
        print(f"初始化远程Embedding模型: {self._model_name}, 维度={self._dimension}")

    def _init_local_model(self):
        """初始化本地模型"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")

        if not LOCAL_MODEL_PATH:
            raise ValueError("请设置 LOCAL_EMBEDDING_PATH 环境变量")

        if not os.path.exists(LOCAL_MODEL_PATH):
            raise FileNotFoundError(f"本地模型路径不存在: {LOCAL_MODEL_PATH}")

        print(f"加载本地Embedding模型: {LOCAL_MODEL_PATH}")
        print(f"设备: {LOCAL_DEVICE}")

        self._model = SentenceTransformer(LOCAL_MODEL_PATH, device=LOCAL_DEVICE)
        self._dimension = self._model.get_sentence_embedding_dimension()
        print(f"模型加载完成: 维度={self._dimension}")

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """生成单个文本的向量"""
        if not text:
            return None

        try:
            if self.model_type == "remote":
                completion = self._client.embeddings.create(
                    model=self._model_name,
                    input=text,
                    dimensions=self._dimension,
                    encoding_format="float"
                )
                return completion.data[0].embedding
            else:
                embedding = self._model.encode(text, normalize_embeddings=True)
                return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            print(f"向量生成失败: {e}")
            return None

    def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """批量生成文本向量"""
        if not texts:
            return []

        try:
            if self.model_type == "remote":
                all_embeddings = []
                batch_size = 10
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    completion = self._client.embeddings.create(
                        model=self._model_name,
                        input=batch,
                        dimensions=self._dimension,
                        encoding_format="float"
                    )
                    all_embeddings.extend([item.embedding for item in completion.data])
                return all_embeddings
            else:
                embeddings = self._model.encode(
                    texts,
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=True
                )
                return [e.tolist() if hasattr(e, 'tolist') else list(e) for e in embeddings]
        except Exception as e:
            print(f"批量向量生成失败: {e}")
            return [None] * len(texts)

    @property
    def dimension(self) -> int:
        return self._dimension


# ==================== 分块器 ====================

class RecursiveChunker:
    """递归分块器 - 优先在自然边界处切分"""

    def __init__(self, chunk_token_num: int = 256):
        self.chunk_token_num = chunk_token_num
        self.min_chunk_size = 20
        self.separators = ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; ", "，", ", ", " "]

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []
        return self._recursive_split(text)

    def _recursive_split(self, text: str) -> List[str]:
        if self._count_tokens(text) <= self.chunk_token_num:
            return [text]

        for separator in self.separators:
            if separator in text:
                parts = text.split(separator, 1)
                left, right = parts[0], parts[1]
                if self._count_tokens(left) >= self.min_chunk_size:
                    return self._recursive_split(left) + self._recursive_split(right)

        mid = len(text) // 2
        return self._recursive_split(text[:mid]) + self._recursive_split(text[mid:])

    def chunk_to_vector_chunks(self, text: str, metadata: Dict = None) -> List[VectorChunk]:
        """分块并返回 VectorChunk 对象列表"""
        chunk_texts = self.chunk(text)
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            if chunk_text.strip():
                chunk_id = hashlib.md5(f"{i}_{chunk_text[:100]}".encode()).hexdigest()[:16]
                chunks.append(VectorChunk(
                    id=f"chunk_{i}_{chunk_id}",
                    content=chunk_text,
                    metadata=metadata or {},
                    token_count=self._count_tokens(chunk_text),
                    chunk_index=i
                ))
        return chunks


# ==================== 文本清洗器 ====================

class TextCleaner:
    """文本清洗器"""

    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""
        # 标准化空白
        text = re.sub(r'[ \t]+', ' ', text)
        # 移除多余换行
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 移除孤立数字行（页码）
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not re.match(r'^\s*\d+\s*$', line):
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)


# ==================== PDF 解析器 ====================

class PDFParser:
    """PDF 文档解析器"""

    def __init__(self):
        try:
            import pdfplumber
            self.pdfplumber = pdfplumber
        except ImportError:
            raise ImportError("请安装 pdfplumber: pip install pdfplumber")

    def extract_text(self, pdf_path: str, from_page: int = 0, to_page: int = None) -> str:
        """提取PDF文本"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"文件不存在: {pdf_path}")

        all_text = []

        with self.pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            end_page = to_page if to_page else total_pages
            end_page = min(end_page, total_pages)

            print(f"PDF总页数: {total_pages}")
            print(f"解析页范围: 第 {from_page + 1} - {end_page} 页")

            for page_num in range(from_page, end_page):
                page = pdf.pages[page_num]
                text = page.extract_text() or ""
                if text:
                    all_text.append(text)
                if (page_num + 1) % 10 == 0:
                    print(f"  已解析 {page_num + 1}/{end_page} 页")

        return "\n\n".join(all_text)


# ==================== 向量存储 ====================
def store_to_vector_db(chunks: List[VectorChunk], index_name: str, file_name: str) -> int:
    """
    将分块存储到向量数据库（使用 vector_storage_service）
    """
    import sys
    import os

    # 添加项目根目录到路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    from app.service.core.vector_store.vector_storage_service import get_vector_storage_service

    # 从环境变量读取 ES 配置
    es_host = os.getenv("ES_HOST", "http://localhost:9200")
    es_user = os.getenv("ES_USER", "elastic")
    es_password = os.getenv("ES_PASSWORD", "infini_rag_flow")

    storage_service = get_vector_storage_service(es_host)
    # 注意：get_vector_storage_service 不接受参数，需要修改
    # 临时方案：直接创建实例
    from app.service.core.vector_store.vector_storage_service import VectorStorageService
    storage_service = VectorStorageService(es_host, es_user, es_password)

    inserted = storage_service.store_vector_chunks(chunks, index_name, file_name)

    print(f"存储完成: {inserted}/{len(chunks)} 条")
    return inserted

# ==================== 主处理流程 ====================

def process_pdf(
        pdf_path: str,
        chunk_size: int = None,
        enable_vectorization: bool = None,
        model_type: str = None,
        from_page: int = None,
        to_page: int = None
) -> List[VectorChunk]:
    """
    处理PDF文件：解析 -> 清洗 -> 分块 -> 向量化

    Args:
        pdf_path: PDF文件路径
        chunk_size: 分块大小（token数），默认从环境变量读取
        enable_vectorization: 是否启用向量化，默认从环境变量读取
        model_type: 模型类型 ("remote" 或 "local")，默认从环境变量读取
        from_page: 起始页，默认从环境变量读取
        to_page: 结束页，默认从环境变量读取

    Returns:
        VectorChunk列表
    """
    # 使用环境变量中的配置作为默认值
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if enable_vectorization is None:
        enable_vectorization = ENABLE_VECTORIZATION
    if model_type is None:
        model_type = EMBEDDING_TYPE
    if from_page is None:
        from_page = FROM_PAGE
    if to_page is None:
        to_page = TO_PAGE

    print("=" * 60)
    print("RAG2 - PDF 智能分块 + 向量化")
    print("=" * 60)
    print(f"\n文件: {pdf_path}")
    print(f"分块大小: {chunk_size} tokens")
    print(f"向量化: {'启用' if enable_vectorization else '禁用'}")
    if enable_vectorization:
        print(f"模型类型: {model_type}")

    # 1. 解析PDF
    print("\n[1/4] 解析PDF...")
    parser = PDFParser()
    raw_text = parser.extract_text(pdf_path, from_page, to_page)

    if not raw_text:
        print("错误: 未能从PDF提取文本")
        return []

    print(f"提取文本长度: {len(raw_text)} 字符")

    # 2. 清洗文本
    print("\n[2/4] 清洗文本...")
    cleaner = TextCleaner()
    cleaned_text = cleaner.clean(raw_text)
    print(f"清洗后长度: {len(cleaned_text)} 字符")

    # 3. 分块
    print("\n[3/4] 分块处理...")
    chunker = RecursiveChunker(chunk_token_num=chunk_size)
    chunks = chunker.chunk_to_vector_chunks(cleaned_text, metadata={'source': os.path.basename(pdf_path)})

    print(f"生成 {len(chunks)} 个文本块")
    if chunks:
        token_counts = [c.token_count for c in chunks]
        print(f"Token统计: 最小={min(token_counts)}, 最大={max(token_counts)}, 平均={sum(token_counts) / len(token_counts):.1f}")

    # 4. 向量化
    if enable_vectorization and chunks:
        print("\n[4/4] 向量化处理...")
        try:
            embedding_model = EmbeddingModel(model_type=model_type)
            texts = [c.content for c in chunks]
            vectors = embedding_model.generate_embeddings(texts)

            for chunk, vector in zip(chunks, vectors):
                if vector:
                    chunk.vector = vector

            vectorized_count = len([c for c in chunks if c.vector])
            print(f"向量化完成: {vectorized_count}/{len(chunks)} 个块")
            print(f"向量维度: {embedding_model.dimension}")

        except Exception as e:
            print(f"向量化失败: {e}")
            print("将继续保存未向量化的块")

    return chunks

def print_summary(chunks: List[VectorChunk]):
    """打印摘要信息"""
    if not chunks:
        print("\n无数据")
        return

    print("\n" + "=" * 60)
    print("处理结果摘要")
    print("=" * 60)

    vectorized = [c for c in chunks if c.vector]
    total_tokens = sum(c.token_count for c in chunks)

    print(f"\n统计:")
    print(f"  - 总块数: {len(chunks)}")
    print(f"  - 向量化块数: {len(vectorized)}")
    print(f"  - 总 Token 数: {total_tokens}")
    print(f"  - 平均 Token 数: {total_tokens / len(chunks):.1f}")

    if vectorized:
        print(f"  - 向量维度: {len(vectorized[0].vector)}")

    print(f"\n前 3 个块预览:")
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk.content[:150].replace('\n', ' ')
        print(f"\n块 {i + 1} (Token: {chunk.token_count}):")
        print(f"  {preview}...")


def print_config():
    """打印当前配置"""
    print("=" * 60)
    print("当前配置")
    print("=" * 60)
    print(f"\n模型配置:")
    print(f"  EMBEDDING_TYPE: {EMBEDDING_TYPE}")
    if EMBEDDING_TYPE == "remote":
        print(f"  DASHSCOPE_API_KEY: {'已设置' if DASHSCOPE_API_KEY else '未设置'}")
        print(f"  REMOTE_MODEL_NAME: {REMOTE_MODEL_NAME}")
        print(f"  REMOTE_DIMENSIONS: {REMOTE_DIMENSIONS}")
    else:
        print(f"  LOCAL_MODEL_PATH: {LOCAL_MODEL_PATH}")
        print(f"  LOCAL_DEVICE: {LOCAL_DEVICE}")
        print(f"  BATCH_SIZE: {BATCH_SIZE}")

    print(f"\n处理配置:")
    print(f"  CHUNK_SIZE: {CHUNK_SIZE}")
    print(f"  FROM_PAGE: {FROM_PAGE}")
    print(f"  TO_PAGE: {TO_PAGE if TO_PAGE else '全部'}")
    print(f"  ENABLE_VECTORIZATION: {ENABLE_VECTORIZATION}")

    print(f"\nES配置:")
    print(f"  ES_HOST: {ES_HOST}")
    print(f"  ES_INDEX_NAME: {ES_INDEX_NAME}")


# ==================== 主函数 ====================

def main():
    """主函数"""
    # 打印配置
    print_config()

    # PDF文件路径
    pdf_file = "【兴证电子】世运电路2023中报点评.pdf"

    # 检查文件是否存在
    if not os.path.exists(pdf_file):
        print(f"\n错误: 文件不存在 - {pdf_file}")
        print(f"当前目录: {os.getcwd()}")

        # 尝试查找文件
        for f in os.listdir('.'):
            if f.endswith('.pdf'):
                print(f"找到PDF文件: {f}")
                pdf_file = f
                break
        else:
            print("未找到任何PDF文件")
            return

    print(f"\n目标文件: {pdf_file}")

    # 处理PDF（所有参数都使用环境变量中的值）
    chunks = process_pdf(
        pdf_path=pdf_file
        # 不传递任何参数，全部从环境变量读取
    )

    # 打印摘要
    print_summary(chunks)

    # 存储到向量数据库
    print("\n" + "=" * 60)
    print("存储到向量数据库")
    print("=" * 60)
    inserted = store_to_vector_db(chunks, ES_INDEX_NAME, os.path.basename(pdf_file))

    if inserted > 0:
        print(f"\n✅ 成功！已存储 {inserted} 条文档到 {ES_INDEX_NAME}")
    else:
        print(f"\n❌ 存储失败，请检查ES连接和配置")

    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()