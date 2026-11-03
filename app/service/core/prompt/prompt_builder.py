# app/service/core/prompt/prompt_builder.py
"""
Prompt构造器 - 构建结构化Prompt
支持系统指令、对话历史、角色设定等
"""

from typing import List, Dict, Any, Optional
import logging
import os

from .templates import get_system_instruction, get_template
from .context_constructor import ContextConstructor
from .multimodal import build_image_parts_detailed, build_multimodal_user_content

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Prompt构造器 - 构建可用于LLM的结构化Prompt"""

    def __init__(
        self,
        system_instruction: str = None,
        max_context_length: int = 4000,
        include_scores: bool = False
    ):
        """
        初始化Prompt构造器

        Args:
            system_instruction: 系统指令，默认使用模板中的指令
            max_context_length: 最大上下文字符数
            include_scores: 是否在上下文中包含相似度分数
        """
        self.system_instruction = system_instruction or get_system_instruction()
        self.context_constructor = ContextConstructor(max_context_length, include_scores)

    def build_messages(
        self,
        question: str,
        results: List[Dict[str, Any]],
        history: Optional[List[Dict]] = None,
        template_name: str = "detailed",
        system_instruction: str = None
    ) -> List[Dict[str, str]]:
        """
        构建LLM API使用的消息列表

        Args:
            question: 用户问题
            results: 检索结果列表
            history: 对话历史 [{"role": "user", "content": "..."}, ...]
            template_name: Prompt模板名称 (simple/detailed/conversation/concise)
            system_instruction: 自定义系统指令（覆盖默认）

        Returns:
            messages列表，格式：[{"role": "system", "content": "..."}, ...]
        """
        # P0-1 + P0-2：先算出实际注入成功的原图（编码只做一次），再据此决定图片块
        # caption 的去留。注入成功的图片块上下文里只留「[图片] 文件名（见附图）」，
        # 让模型只能看图作答；注入失败（文件丢失/超预算/开关关闭）的仍保留完整
        # caption，作为图没进去时的兜底。
        injected_refs, image_parts = build_image_parts_detailed(results)
        caption_mode = os.getenv("MULTIMODAL_IMAGE_CAPTION_MODE", "drop").strip().lower()
        if caption_mode not in ("full", "brief", "drop"):
            caption_mode = "drop"
        if injected_refs and caption_mode != "full":
            logger.info(
                f"P0-1 图片 caption 去重: 注入成功 {len(injected_refs)} 张，"
                f"模式 {caption_mode}"
            )

        # 格式化文档上下文
        context = self.context_constructor.format_documents(
            results,
            image_caption_mode=caption_mode,
            injected_image_refs=injected_refs,
        )

        # 获取模板
        template = get_template(template_name)

        # 构建用户消息
        if template_name == "conversation":
            history_str = self._format_history(history)
            user_content = template.format(
                context=context,
                history=history_str or "（无历史记录）",
                question=question
            )
        else:
            user_content = template.format(context=context, question=question)

        messages = []

        # 添加系统指令
        sys_inst = system_instruction or self.system_instruction
        messages.append({"role": "system", "content": sys_inst})

        # 添加对话历史
        if history:
            for msg in history:
                if msg.get("role") in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg.get("content", "")
                    })

        # 添加当前用户消息
        # 若检索结果里含图片块，把原图按 OpenAI 多模态格式一并注入；
        # 没有可用图片时 content 仍是纯字符串，行为与原来完全一致。
        # 复用上面已经算好的 image_parts，避免重复读盘 / 降采样 / 编码。
        user_payload = build_multimodal_user_content(user_content, results, parts=image_parts)
        messages.append({"role": "user", "content": user_payload})

        return messages

    def build_prompt_text(
        self,
        question: str,
        results: List[Dict[str, Any]],
        history: Optional[List[Dict]] = None,
        template_name: str = "simple"
    ) -> str:
        """
        构建纯文本Prompt（不含system角色，适用于不支持system的模型）

        Args:
            question: 用户问题
            results: 检索结果
            history: 对话历史
            template_name: 模板名称

        Returns:
            纯文本Prompt字符串
        """
        context = self.context_constructor.format_documents(results)
        template = get_template(template_name)

        # 构建系统指令部分
        system_part = f"{self.system_instruction}\n\n" if self.system_instruction else ""

        if template_name == "conversation":
            history_str = self._format_history(history)
            prompt = system_part + template.format(
                context=context,
                history=history_str or "（无历史记录）",
                question=question
            )
        else:
            prompt = system_part + template.format(context=context, question=question)

        return prompt

    def _format_history(self, history: Optional[List[Dict]]) -> str:
        """格式化对话历史"""
        if not history:
            return ""

        formatted = []
        for msg in history[-6:]:  # 最多保留6条
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("content", "")
            if content:
                formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    def build_with_custom_roles(
        self,
        question: str,
        results: List[Dict[str, Any]],
        role_description: str = None,
        output_format: str = None
    ) -> List[Dict[str, str]]:
        """
        构建带角色设定和输出格式的Prompt

        Args:
            question: 用户问题
            results: 检索结果
            role_description: 角色描述，如"你是一位金融分析师"
            output_format: 输出格式要求，如"请用JSON格式输出"

        Returns:
            messages列表
        """
        context = self.context_constructor.format_documents(results)

        # 构建系统消息
        system_content = self.system_instruction
        if role_description:
            system_content = f"{role_description}\n\n{system_content}"
        if output_format:
            system_content = f"{system_content}\n\n输出格式要求：\n{output_format}"

        # 构建用户消息
        user_content = f"""请根据以下文档内容回答问题。

## 参考文档
{context}

## 问题
{question}

请回答："""

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]


# 全局单例
_prompt_builder = None


def get_prompt_builder(
    system_instruction: str = None,
    max_context_length: int = 4000
) -> PromptBuilder:
    """获取Prompt构造器实例"""
    global _prompt_builder
    if _prompt_builder is None:
        _prompt_builder = PromptBuilder(system_instruction, max_context_length)
    return _prompt_builder


__all__ = ['PromptBuilder', 'get_prompt_builder']