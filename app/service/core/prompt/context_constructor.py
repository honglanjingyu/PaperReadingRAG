# app/service/core/prompt/context_constructor.py
"""
上下文构造器 - 整合召回文档和用户问题
"""

from typing import List, Dict, Any, Optional


class ContextConstructor:
    """上下文构造器 - 将召回文档格式化为上下文"""

    def __init__(self, max_context_length: int = 4000, include_scores: bool = False):
        """
        初始化上下文构造器

        Args:
            max_context_length: 最大上下文字符数
            include_scores: 是否包含相似度分数
        """
        self.max_context_length = max_context_length
        self.include_scores = include_scores

    # P0-1：图片块 caption 的呈现方式
    IMAGE_CAPTION_MODES = ("full", "brief", "drop")
    BRIEF_CAPTION_CHARS = 80

    @staticmethod
    def _is_image_result(result: Dict[str, Any]) -> bool:
        """判断一个检索结果是否为图片块（有 image_ref 或 chunk_type=image）。"""
        if result.get("image_ref") or result.get("image_path"):
            return True
        return str(result.get("chunk_type", "")).lower() == "image"

    @staticmethod
    def _result_image_ref(result: Dict[str, Any]) -> str:
        return str(result.get("image_ref") or result.get("image_path") or "")

    @staticmethod
    def _format_image_placeholder(doc_name: str) -> str:
        return f"[图片] {doc_name}（见附图）"

    def format_documents(
            self,
            results: List[Dict[str, Any]],
            include_scores: bool = None,
            image_caption_mode: str = "full",
            injected_image_refs: Optional[List[str]] = None,
    ) -> str:
        """将检索结果格式化为上下文字符串，使用真实文件名。

        Args:
            image_caption_mode: 图片块 caption 的呈现方式。
                full  —— 保留 VLM 生成的完整 caption（旧行为，默认）
                brief —— 已注入原图时只留占位 + 前 80 字 caption
                drop  —— 已注入原图时只留「[图片] 文件名（见附图）」
            injected_image_refs: 本次实际注入原图的 image_ref 列表。
                只有这些图片块才能去 caption —— 原图没进去却把 caption
                删掉，等于把信息全丢了。
        """
        if not results:
            return "（未找到相关文档内容）"

        if image_caption_mode not in self.IMAGE_CAPTION_MODES:
            image_caption_mode = "full"
        injected = set(injected_image_refs or [])

        include = include_scores if include_scores is not None else self.include_scores
        formatted_parts = []

        for result in results:
            content = result.get("content", result.get("content_with_weight", ""))
            if not content:
                continue

            # ========== 关键修改：获取真实文件名 ==========
            doc_name = (
                    result.get("document_name") or
                    result.get("docnm") or
                    result.get("docnm_kwd") or
                    "未知文档"
            )

            # 清理文件名（去除路径）
            if doc_name and '/' in doc_name:
                doc_name = doc_name.split('/')[-1]
            if doc_name and '\\' in doc_name:
                doc_name = doc_name.split('\\')[-1]

            # P0-1：已注入原图的图片块，去掉（或压缩）caption，逼模型看图
            if image_caption_mode != "full" and self._is_image_result(result):
                ref = self._result_image_ref(result)
                if ref and ref in injected:
                    placeholder = self._format_image_placeholder(doc_name)
                    if image_caption_mode == "drop":
                        content = placeholder
                    else:
                        brief = content[:self.BRIEF_CAPTION_CHARS].strip()
                        content = f"{placeholder}\n{brief}..." if brief else placeholder

            # 截断过长的单个文档
            if len(content) > self.max_context_length // max(len(results), 1):
                content = content[:self.max_context_length // max(len(results), 1)] + "..."

            if include:
                score = result.get("score", result.get("similarity", result.get("_score", 0)))
                formatted_parts.append(f"[{doc_name}] (相关度: {score:.3f})\n{content}\n")
            else:
                formatted_parts.append(f"[{doc_name}]\n{content}\n")

        # 确保总长度不超过限制
        context = "\n".join(formatted_parts)
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "..."

        return context

    def format_with_metadata(
        self,
        results: List[Dict[str, Any]],
        metadata_fields: List[str] = None
    ) -> str:
        """
        格式化文档，包含更多元数据

        Args:
            results: 检索结果列表
            metadata_fields: 需要包含的元数据字段

        Returns:
            包含元数据的上下文字符串
        """
        if not results:
            return "（未找到相关文档内容）"

        metadata_fields = metadata_fields or ["document_name", "page_num", "chunk_id"]
        formatted_parts = []

        for i, result in enumerate(results, 1):
            content = result.get("content", result.get("content_with_weight", ""))
            if not content:
                continue

            # 构建元数据信息
            meta_info = []
            for field in metadata_fields:
                value = None
                if field == "document_name":
                    value = result.get("document_name", result.get("docnm", "未知"))
                elif field == "page_num":
                    value = result.get("page_num", result.get("page_num_int", "未知"))
                elif field == "chunk_id":
                    value = result.get("chunk_id", result.get("_id", "未知"))
                elif field in result:
                    value = result[field]

                if value:
                    meta_info.append(f"{field}: {value}")

            meta_str = f" ({', '.join(meta_info)})" if meta_info else ""
            formatted_parts.append(f"[文档{i}]{meta_str}\n{content}\n")

        return "\n".join(formatted_parts)

    def build_context(
        self,
        question: str,
        results: List[Dict[str, Any]],
        history: Optional[List[Dict]] = None,
        template_name: str = "simple"
    ) -> Dict[str, str]:
        """
        构建完整的上下文（包含格式化文档和问题）

        Args:
            question: 用户问题
            results: 检索结果
            history: 对话历史
            template_name: 模板名称

        Returns:
            包含 context 和 question 的字典
        """
        from .templates import get_template

        context = self.format_documents(results)
        template = get_template(template_name)

        # 处理对话历史
        history_str = ""
        if history:
            history_str = "\n".join([
                f"{'用户' if msg['role'] == 'user' else '助手'}: {msg['content']}"
                for msg in history[-6:]  # 最多保留6条
            ])

        # 根据模板填充
        if template_name == "conversation":
            filled_template = template.format(
                context=context,
                history=history_str or "（无历史记录）",
                question=question
            )
        else:
            filled_template = template.format(context=context, question=question)

        return {
            "context": context,
            "question": question,
            "prompt": filled_template,
            "history": history_str
        }


__all__ = ['ContextConstructor']