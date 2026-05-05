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

    def format_documents(
        self,
        results: List[Dict[str, Any]],
        include_scores: bool = None
    ) -> str:
        """
        将检索结果格式化为上下文字符串

        Args:
            results: 检索结果列表，格式如 [{"rank": 1, "content": "...", "score": 0.85}, ...]
            include_scores: 是否包含分数信息

        Returns:
            格式化后的上下文字符串
        """
        if not results:
            return "（未找到相关文档内容）"

        include = include_scores if include_scores is not None else self.include_scores
        formatted_parts = []

        for i, result in enumerate(results, 1):
            content = result.get("content", result.get("content_with_weight", ""))
            if not content:
                continue

            # 截断过长的单个文档
            if len(content) > self.max_context_length // len(results):
                content = content[:self.max_context_length // len(results)] + "..."

            if include:
                score = result.get("score", result.get("similarity", result.get("_score", 0)))
                doc_name = result.get("document_name", result.get("docnm", "未知文档"))
                formatted_parts.append(f"[文档{i}] (来源: {doc_name}, 相关度: {score:.3f})\n{content}\n")
            else:
                formatted_parts.append(f"[文档{i}]\n{content}\n")

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