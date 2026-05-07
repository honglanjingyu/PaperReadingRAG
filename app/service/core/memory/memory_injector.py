# app/service/core/memory/memory_injector.py
"""
记忆注入器 - 将对话历史注入到Prompt中
"""

from typing import List, Dict, Optional, Any
import logging
from .session_memory import SessionMemory, get_memory_manager

logger = logging.getLogger(__name__)


class MemoryInjector:
    """
    记忆注入器
    负责将对话历史注入到Prompt构造流程中
    """

    # 对话历史格式化模板
    HISTORY_TEMPLATE = """## 对话历史
{history}

## 当前问题
{question}

请基于对话历史和文档内容回答当前问题。如果当前问题涉及历史对话中的内容，请结合上下文理解。"""

    def __init__(self, memory_manager: SessionMemory = None):
        """
        初始化记忆注入器

        Args:
            memory_manager: 记忆管理器实例
        """
        self.memory_manager = memory_manager or get_memory_manager()

    def format_history(
            self,
            session_id: str,
            max_turns: int = 10,
            max_tokens: int = 2000,
            include_timestamps: bool = False
    ) -> str:
        """
        格式化对话历史为文本

        Args:
            session_id: 会话ID
            max_turns: 最大轮次
            max_tokens: 最大token数
            include_timestamps: 是否包含时间戳

        Returns:
            格式化的历史文本
        """
        history = self.memory_manager.get_conversation_history(
            session_id=session_id,
            max_turns=max_turns,
            max_tokens=max_tokens
        )

        if not history:
            return ""

        # 格式化历史记录
        formatted_lines = []
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            content = msg["content"]
            formatted_lines.append(f"{role}: {content}")

        return "\n".join(formatted_lines)

    def build_messages_with_history(
            self,
            session_id: str,
            user_question: str,
            system_instruction: str = None,
            max_history_turns: int = 10,
            max_history_tokens: int = 2000
    ) -> List[Dict[str, str]]:
        """
        构建包含历史的完整消息列表

        Args:
            session_id: 会话ID
            user_question: 用户当前问题
            system_instruction: 系统指令
            max_history_turns: 最大历史轮次
            max_history_tokens: 最大历史token数

        Returns:
            消息列表
        """
        messages = []

        # 添加系统指令
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        # 添加历史消息
        history = self.memory_manager.get_conversation_history(
            session_id=session_id,
            max_turns=max_history_turns,
            max_tokens=max_history_tokens
        )
        messages.extend(history)

        # 添加当前问题
        messages.append({"role": "user", "content": user_question})

        return messages

    def inject_into_prompt(
            self,
            session_id: str,
            question: str,
            context: str,
            template_name: str = "detailed",
            max_history_turns: int = 10
    ) -> Dict[str, Any]:
        """
        将历史注入到Prompt中

        Args:
            session_id: 会话ID
            question: 当前问题
            context: 检索到的上下文
            template_name: 模板名称
            max_history_turns: 最大历史轮次

        Returns:
            包含注入后prompt的字典
        """
        history_text = self.format_history(
            session_id=session_id,
            max_turns=max_history_turns
        )

        from app.service.core.prompt.templates import get_template

        base_template = get_template(template_name)

        if history_text:
            # 使用带历史的模板
            if template_name == "conversation":
                template = base_template
                prompt = template.format(
                    context=context,
                    history=history_text,
                    question=question
                )
            else:
                # 在详细模板中添加历史
                prompt = f"""## 文档内容
{context}

{self.HISTORY_TEMPLATE.format(history=history_text, question=question)}"""
        else:
            # 没有历史时使用原模板
            if template_name == "conversation":
                prompt = base_template.format(
                    context=context,
                    history="（无历史记录）",
                    question=question
                )
            else:
                prompt = base_template.format(context=context, question=question)

        return {
            "prompt": prompt,
            "has_history": bool(history_text),
            "history_length": len(history_text),
            "context": context,
            "question": question
        }

    def update_memory(
            self,
            session_id: str,
            question: str,
            answer: str
    ) -> bool:
        """
        更新会话记忆

        Args:
            session_id: 会话ID
            question: 用户问题
            answer: 助手回答

        Returns:
            是否更新成功
        """
        success = True

        if not self.memory_manager.add_message(session_id, "user", question):
            success = False

        if answer:
            if not self.memory_manager.add_message(session_id, "assistant", answer):
                success = False

        return success

    def clear_memory(self, session_id: str) -> bool:
        """清除会话记忆"""
        return self.memory_manager.clear_session(session_id)


__all__ = ['MemoryInjector']