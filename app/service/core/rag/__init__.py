"""RAG 模块 - 文档处理、搜索、生成"""

import os
import hashlib
import logging
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ========== 导入子模块 ==========
from app.service.core.deepdoc import DocumentParser
from app.service.core.chunking import ParentChildSplitter, ParentChildDocument, ParentChunk, ChildChunk
from app.service.core.embedding import get_embedding_service, VectorChunk
from app.service.core.vector_store import get_vector_storage_service
from app.service.core.prompt import PromptBuilder
from app.service.core.llm import get_llm_service
from app.service.core.retrieval import get_parent_child_retriever
from .cached_search import CachedSearchService
from .async_processor import AsyncDocumentProcessor, get_async_processor, init_async_processor, shutdown_async_processor


# ========== 文档处理 ==========

class ParentChildVectorChunk:
    """父子向量块"""
    def __init__(self, child: ChildChunk, parent: ParentChunk):
        self.child = child
        self.parent = parent
        self.id = child.id
        self.content = child.content
        self.parent_content = parent.content
        self.parent_id = parent.id
        self.vector = None
        self.token_count = child.token_count
        self.docnm = (child.metadata or {}).get('source') or (parent.metadata or {}).get('source', '')
        self.metadata = {
            **child.metadata,
            'parent_id': parent.id,
            'parent_content_preview': parent.content[:200],
            'chunk_type': 'child',
            'parent_chunk_index': parent.chunk_index,
            'child_chunk_index': child.chunk_index
        }

    def to_dict(self, user_level: str = "normal") -> Dict:
        return {
            "id": self.id, "content": self.content, "content_with_weight": self.content,
            "parent_id": self.parent_id, "parent_content": self.parent_content,
            "vector": self.vector or [], "token_count": self.token_count,
            "user_level": user_level, "docnm": self.docnm, **self.metadata
        }


def process_document(
        file_path: str,
        chunk_size: int = 256,
        enable_vectorization: bool = True,
        enable_storage: bool = True,
        from_page: int = 0,
        to_page: int = None,
        index_name: str = None,
        verbose: bool = False,
        user_level: str = "normal"
) -> List:
    """处理文档：解析 -> 分块 -> 向量化 -> 存储"""
    # 1. 解析文档
    parser = DocumentParser()
    parsed = parser.parse(file_path, from_page=from_page, to_page=to_page or 100000,
                          enable_cleaning=True, verbose=verbose)
    if not parsed.cleaned_text:
        return []

    # 2. 父子分块
    parent_chunk_size = int(os.getenv("PARENT_CHUNK_SIZE", "500"))
    splitter = ParentChildSplitter(parent_chunk_size=parent_chunk_size,
                                    child_chunk_size=chunk_size)
    doc = splitter.split_document(
        text=parsed.cleaned_text,
        metadata={'source': parsed.file_name, 'document_name': parsed.file_name},
        document_id=hashlib.md5(file_path.encode()).hexdigest()[:16],
        document_name=parsed.file_name
    )

    if verbose:
        logger.info(f"分块完成: {len(doc.parent_chunks)} 父块, {len(doc.all_children)} 子块")

    # 3. 向量化
    vector_chunks = []
    if enable_vectorization and doc.all_children:
        embedding = get_embedding_service()
        vectors = embedding.generate_embeddings([c.content for c in doc.all_children])
        parent_map = {p.id: p for p in doc.parent_chunks}

        for child, vec in zip(doc.all_children, vectors):
            if vec and child.parent_id in parent_map:
                vc = ParentChildVectorChunk(child, parent_map[child.parent_id])
                vc.vector = vec
                vector_chunks.append(vc)

    # 4. 存储
    if enable_storage and vector_chunks:
        storage = get_vector_storage_service()
        index = index_name or os.getenv("VECTOR_INDEX_NAME", "rag_documents")
        storage.store_vector_chunks(vector_chunks, index, parsed.file_name, user_level=user_level)

    return vector_chunks


def get_processing_stats(file_path: str) -> dict:
    """获取处理统计"""
    parser = DocumentParser()
    parsed = parser.parse(file_path, enable_cleaning=True, verbose=False)
    splitter = ParentChildSplitter()
    doc = splitter.split_document(parsed.cleaned_text or "")
    stats = splitter.get_statistics(doc)
    return {
        'file_name': parsed.file_name, 'file_type': parsed.file_type,
        'total_pages': parsed.total_pages, 'text_length': len(parsed.cleaned_text or 0),
        'parent_count': stats['parent_count'], 'child_count': stats['child_count'],
        'avg_parent_tokens': stats['avg_parent_tokens'], 'avg_child_tokens': stats['avg_child_tokens']
    }


# ========== 搜索 ==========

def enhanced_search_with_hybrid_and_rerank(
        question: str,
        index_name: str = None,
        top_k: int = 5,
        recall_k: int = 10,
        keyword_weight: float = 0.4,
        vector_weight: float = 0.6,
        enable_rerank: bool = True,
        enable_query_rewrite: bool = True,
        similarity_threshold: float = 0.3,
        rerank_type: str = "auto",
        user_level: str = None,
        use_cache: bool = True,
        verbose: bool = False  # 添加 verbose 参数
) -> dict:
    """增强检索 - 混合检索 + 重排序"""
    index_name = index_name or os.getenv("VECTOR_INDEX_NAME", "rag_documents")
    actual_recall_k = max(recall_k, top_k * 2)
    retriever = get_parent_child_retriever()

    if use_cache:
        return CachedSearchService().search_with_cache(
            search_func=retriever.search_with_query_rewrite,
            question=question,
            index_name=index_name,
            top_k=top_k,
            recall_k=actual_recall_k,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            similarity_threshold=similarity_threshold,
            enable_rerank=enable_rerank,
            enable_query_rewrite=enable_query_rewrite,
            rerank_type=rerank_type,
            user_level=user_level,
            verbose=verbose  # 传递 verbose
        )
    return retriever.search_with_query_rewrite(
        question=question, index_name=index_name, top_k=top_k, recall_k=actual_recall_k,
        keyword_weight=keyword_weight, vector_weight=vector_weight,
        similarity_threshold=similarity_threshold, enable_rerank=enable_rerank,
        rerank_type=rerank_type, enable_query_rewrite=enable_query_rewrite,
        user_level=user_level, verbose=verbose
    )


# ========== 生成 ==========

def generate_answer(
        question: str,
        results: List[Dict],
        history: List[Dict] = None,
        template_name: str = "detailed",
        preview_answer: bool = False,
        verbose: bool = False  # 添加 verbose 参数
) -> dict:
    """生成答案（非流式）"""
    builder = PromptBuilder(max_context_length=4000, include_scores=True)
    messages = builder.build_messages(question=question, results=results,
                                       history=history, template_name=template_name)
    llm = get_llm_service()
    answer = llm.generate(messages, verbose=verbose)  # 传递 verbose
    return {"success": answer is not None, "question": question, "answer": answer,
            "model_info": llm.get_model_info()}


def generate_answer_stream(question: str, results: List[Dict],
                           history: List[Dict] = None, template_name: str = "detailed",
                           verbose: bool = False):  # 添加 verbose 参数
    """生成答案（流式）"""
    builder = PromptBuilder(max_context_length=4000, include_scores=True)
    messages = builder.build_messages(question=question, results=results,
                                       history=history, template_name=template_name)
    yield from get_llm_service().generate_stream(messages)


def process_document_parent_child(
        file_path: str,
        parent_chunk_size: int = 500,
        child_chunk_size: int = 150,
        enable_vectorization: bool = True,
        enable_storage: bool = True,
        from_page: int = 0,
        to_page: int = None,
        index_name: str = None,
        verbose: bool = False,
        user_level: str = "normal"
):
    """兼容旧接口 - 直接调用 process_document"""
    return process_document(
        file_path=file_path,
        chunk_size=child_chunk_size,
        enable_vectorization=enable_vectorization,
        enable_storage=enable_storage,
        from_page=from_page,
        to_page=to_page,
        index_name=index_name,
        verbose=verbose,
        user_level=user_level
    ), None


def process_document_with_text(
        text_content: str,
        file_name: str,
        chunk_size: int = 256,
        enable_vectorization: bool = True,
        enable_storage: bool = True,
        index_name: str = None,
        user_level: str = "normal",
        verbose: bool = False,
        image_path: str = "",
        media_type: str = "document",
        frame_items: Optional[List[Dict[str, Any]]] = None
) -> List:
    """
    处理已提取文字的内容（用于图片/音频/视频的OCR/ASR结果）

    image_path 非空且 embedding 模型支持图片编码时，会额外生成一个「图像向量块」：
    它的向量由图像塔编码原图得到，与文本块同维度、同空间，写进同一个集合的
    同一个 vector 字段，因此一次 ANN 就能同时命中文本块和图片块。

    frame_items 仅视频使用：[{path, index, text}]，每个关键帧各生成一个图像向量块
    （带 image_ref 指向帧图，生成阶段可回填原帧）。给了 frame_items 时
    image_path 不再单独成块，避免同一张图重复入库。

    Returns:
        List: 向量块列表，处理失败返回空列表
    """
    # 视频的视觉载体是帧图，不是视频文件本身；有帧就按帧入库
    frame_items = [f for f in (frame_items or []) if f and f.get("path")]
    has_visual = bool(image_path) or bool(frame_items)

    if not text_content or not text_content.strip():
        # 图片可能本来就没什么可 OCR 的文字（示意图、照片），但视觉特征仍要入库，
        # 否则真一次 ANN 就退化成纯文本检索了。
        if has_visual:
            logger.info(f"文件 {file_name} 未提取到文字，仅按视觉特征入库")
            text_content = ""
        else:
            logger.warning(f"文件 {file_name} 没有提取到文字内容")
            return []

    logger.info(f"开始处理文档: {file_name}, 文字长度={len(text_content)}, 用户等级={user_level}")

    # 1. 父子分块
    parent_chunk_size = int(os.getenv("PARENT_CHUNK_SIZE", "500"))
    splitter = ParentChildSplitter(
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=chunk_size
    )

    doc = splitter.split_document(
        text=text_content,
        metadata={'source': file_name, 'document_name': file_name},
        document_id=hashlib.md5(file_name.encode()).hexdigest()[:16],
        document_name=file_name
    )

    if verbose:
        logger.info(f"分块完成: {len(doc.parent_chunks)} 父块, {len(doc.all_children)} 子块")

    if not doc.all_children:
        logger.warning(f"文件 {file_name} 分块后没有子块")

    # 2. 向量化（文本块）
    vector_chunks = []
    embedding = get_embedding_service() if enable_vectorization else None
    if enable_vectorization and doc.all_children:
        vectors = embedding.generate_embeddings([c.content for c in doc.all_children])
        parent_map = {p.id: p for p in doc.parent_chunks}

        for child, vec in zip(doc.all_children, vectors):
            if vec and child.parent_id in parent_map:
                vc = ParentChildVectorChunk(child, parent_map[child.parent_id])
                vc.vector = vec
                vector_chunks.append(vc)

        logger.info(f"向量化完成: {len(vector_chunks)}/{len(doc.all_children)} 个块")

    # 2.1 图片块（真一次 ANN 的关键）
    #     图片用图像塔编码，向量维度/空间与文本块完全一致，写进同一个集合的
    #     同一个 vector 字段。因此一次 ANN 就能同时命中文本块和图片块。
    #     即使 OCR 没出文字（all_children 为空），图片块也要照常入库。
    # 视频走 frame_items 分支，image_path 不再单独成块（避免同一张图重复入库）
    if image_path and not frame_items and enable_vectorization and embedding is not None and getattr(embedding, "supports_image", False):
        try:
            import types
            img_vec = embedding.encode_image(image_path)
            if img_vec:
                text_part = (text_content or "").strip()
                snippet = text_part[:2000] if text_part else "（该图片未识别出文字，仅保留视觉特征）"
                # BM25 只对 content / content_with_weight 的字面词打分。图片块光有
                # 「[图片] 文件名 + OCR 文本」时，示意图、照片这类 OCR 出不了字的图
                # 在 BM25 里等于空文档。这里用多模态大模型生成一段中文描述拼进
                # content，图片块就能被文字检索到。失败时返回原文，不影响入库。
                image_caption = ""
                try:
                    from app.service.core.multimodal.image_caption import build_image_payload
                    image_content, image_caption = build_image_payload(image_path, file_name, text_part)
                except Exception as e:
                    logger.warning(f"图片描述生成失败，退回原文本 {file_name}: {e}")
                    image_content = f"[图片] {file_name}\n{snippet}"
                vector_chunks.append(types.SimpleNamespace(
                    id=hashlib.md5(f"img:{file_name}".encode()).hexdigest(),
                    content=image_content,
                    vector=img_vec,
                    metadata={
                        "media_type": "image",
                        "chunk_type": "image",
                        "image_ref": image_path,
                        # 描述单独留一份，写进 ES 的 caption 字段单独加权
                        "caption": image_caption,
                        "doc_id": hashlib.md5(file_name.encode()).hexdigest()[:16],
                    },
                    token_count=0,
                    chunk_index=0,
                    user_level=user_level,
                    parent_id="",
                    parent_content="",
                ))
                logger.info(f"图片向量块已生成: {file_name} dim={len(img_vec)}")
            else:
                logger.warning(f"图片编码返回空，跳过图片块: {file_name}")
        except Exception as e:
            logger.error(f"图片块生成失败 {file_name}: {e}", exc_info=True)

    # 2.2 视频关键帧块（每个关键帧一个图像向量块）
    #     视频没法直接进 VISTA 的视觉塔，视觉载体是抽出来的关键帧。每帧都用
    #     图像塔编码，向量与文本块同维度、写同一个 vector 字段，一次 ANN 就能
    #     命中「某段画面」；image_ref 指向持久化的帧图，生成阶段可回填原帧。
    if frame_items and enable_vectorization and embedding is not None and getattr(embedding, "supports_image", False):
        try:
            import types
            doc_id = hashlib.md5(file_name.encode()).hexdigest()[:16]
            ok_frames = 0
            for item in frame_items:
                frame_path = item.get("path") or ""
                idx = item.get("index", ok_frames)
                if not frame_path or not os.path.exists(frame_path):
                    logger.warning(f"跳过缺失的视频帧: {frame_path}")
                    continue
                try:
                    frame_vec = embedding.encode_image(frame_path)
                except Exception as e:
                    logger.warning(f"视频帧编码失败 {frame_path}: {e}")
                    continue
                if not frame_vec:
                    logger.warning(f"视频帧编码返回空，跳过: {frame_path}")
                    continue

                frame_text = (item.get("text") or "").strip()
                frame_name = f"{file_name}#frame{idx}"
                snippet = frame_text[:2000] if frame_text else "（该帧未识别出文字，仅保留视觉特征）"
                # 与图片块同一套逻辑：生成一段中文描述拼进 content，让 BM25 也能
                # 检索到画面语义（纯 OCR 文本对示意图/照片几乎是空文档）。
                frame_caption = ""
                try:
                    from app.service.core.multimodal.image_caption import build_image_payload
                    frame_content, frame_caption = build_image_payload(frame_path, frame_name, frame_text)
                except Exception as e:
                    logger.warning(f"视频帧描述生成失败，退回原文本 {frame_name}: {e}")
                    frame_content = f"[视频画面] {file_name} 第 {idx + 1} 帧\n{snippet}"

                vector_chunks.append(types.SimpleNamespace(
                    id=hashlib.md5(f"frame:{file_name}:{idx}".encode()).hexdigest(),
                    content=frame_content,
                    vector=frame_vec,
                    metadata={
                        "media_type": "image",
                        "chunk_type": "image",
                        # 生成阶段按它把原帧转 base64 回填给多模态 LLM
                        "image_ref": frame_path,
                        "caption": frame_caption,
                        "doc_id": doc_id,
                        "source_video": file_name,
                        "frame_index": idx,
                    },
                    token_count=0,
                    chunk_index=idx,
                    user_level=user_level,
                    parent_id="",
                    parent_content="",
                ))
                ok_frames += 1

            logger.info(f"视频帧向量块已生成: {file_name} {ok_frames}/{len(frame_items)} 帧")
        except Exception as e:
            logger.error(f"视频帧块生成失败 {file_name}: {e}", exc_info=True)

    if not vector_chunks:
        logger.warning(f"文件 {file_name} 向量化后没有有效块")
        return []

    # 3. 存储
    if enable_storage and vector_chunks:
        storage = get_vector_storage_service()
        index = index_name or os.getenv("VECTOR_INDEX_NAME", "rag_documents")
        inserted = storage.store_vector_chunks(vector_chunks, index, file_name, user_level=user_level)
        logger.info(f"文档 {file_name} 存储完成，插入 {inserted} 个向量块 (期望 {len(vector_chunks)})")

        # 验证存储是否成功
        if inserted == 0:
            logger.error(f"文档 {file_name} 存储失败，插入数量为0")
            return []

    logger.info(f"✅ 文档 {file_name} 处理完成，共 {len(vector_chunks)} 个向量块")
    return vector_chunks

__all__ = [
    'process_document',
    'get_processing_stats',
    'enhanced_search_with_hybrid_and_rerank',
    'generate_answer',
    'generate_answer_stream',
    'ParentChildVectorChunk',
    'process_document_parent_child',
]