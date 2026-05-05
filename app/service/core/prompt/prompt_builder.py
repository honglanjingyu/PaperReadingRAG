# app/service/core/prompt/prompt_builder.py
"""
Prompt构造器 - 构建结构化Prompt
支持系统指令、对话历史、角色设定等
"""

from typing import List, Dict, Any, Optional
from .templates import get_system_instruction, get_template
from .context_constructor import ContextConstructor


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
        # 格式化文档上下文
        context = self.context_constructor.format_documents(results)

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
        messages.append({"role": "user", "content": user_content})

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