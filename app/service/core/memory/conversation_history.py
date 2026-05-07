# app/service/core/memory/conversation_history.py
"""
对话历史管理器 - 支持历史总结和压缩
"""

import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """单轮对话"""
    user_question: str
    assistant_answer: str
    timestamp: float = field(default_factory=time.time)
    summarized: bool = False  # 是否已被总结


class ConversationHistory:
    """
    对话历史管理器
    支持自动总结和压缩
    """

    def __init__(
            self,
            max_turns: int = 20,
            max_tokens: int = 4000,
            enable_summarization: bool = False
    ):
        """
        初始化对话历史管理器

        Args:
            max_turns: 最大保留轮次
            max_tokens: 最大token数
            enable_summarization: 是否启用自动总结
        """
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.enable_summarization = enable_summarization

        self._turns: List[ConversationTurn] = []
        self._summary: Optional[str] = None

    def add_turn(self, user_question: str, assistant_answer: str):
        """添加一轮对话"""
        self._turns.append(ConversationTurn(
            user_question=user_question,
            assistant_answer=assistant_answer
        ))

        # 维护大小限制
        self._maintain_size()

    def _maintain_size(self):
        """维护历史大小"""
        # 按轮次限制
        if len(self._turns) > self.max_turns:
            # 移除最早的轮次
            self._turns.pop(0)

    def get_history_text(
            self,
            max_turns: int = None,
            max_tokens: int = None,
            include_summary: bool = True
    ) -> str:
        """
        获取历史文本

        Args:
            max_turns: 最大轮次
            max_tokens: 最大token数
            include_summary: 是否包含总结

        Returns:
            历史文本
        """
        max_turns = max_turns or self.max_turns
        max_tokens = max_tokens or self.max_tokens

        parts = []

        # 添加总结
        if include_summary and self._summary:
            parts.append(f"[历史概要]\n{self._summary}")

        # 获取最近的对话
        recent_turns = self._turns[-max_turns:] if max_turns > 0 else self._turns

        for i, turn in enumerate(recent_turns):
            parts.append(f"[第{i + 1}轮]\n用户: {turn.user_question}\n助手: {turn.assistant_answer}")

        history_text = "\n\n".join(parts)

        # 按token限制裁剪
        if max_tokens and self._estimate_tokens(history_text) > max_tokens:
            history_text = self._trim_history(history_text, max_tokens)

        return history_text

    def get_plain_history(self, max_turns: int = None) -> List[Dict[str, str]]:
        """获取纯文本历史（用于LLM）"""
        max_turns = max_turns or self.max_turns
        recent_turns = self._turns[-max_turns:] if max_turns > 0 else self._turns

        messages = []
        for turn in recent_turns:
            messages.append({"role": "user", "content": turn.user_question})
            messages.append({"role": "assistant", "content": turn.assistant_answer})

        return messages

    def _estimate_tokens(self, text: str) -> int:
        """估算token数"""
        if not text:
            return 0
        chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        others = len(text) - chinese
        return int(chinese / 1.5 + others / 4)

    def _trim_history(self, text: str, max_tokens: int) -> str:
        """裁剪历史文本"""
        # 简单实现：按字符裁剪
        chars_per_token = 2
        max_chars = max_tokens * chars_per_token
        if len(text) > max_chars:
            return "...\n" + text[-(max_chars - 100):]
        return text

    def clear(self):
        """清空历史"""
        self._turns.clear()
        self._summary = None

    def get_turn_count(self) -> int:
        """获取对话轮次数"""
        return len(self._turns)

    def summarize(self, llm_service=None) -> Optional[str]:
        """
        使用LLM总结对话历史

        Args:
            llm_service: LLM服务实例

        Returns:
            总结文本
        """
        if not self._turns:
            return None

        if llm_service is None:
            # 简单总结
            return self._simple_summarize()

        # 使用LLM总结
        try:
            history_text = self.get_history_text(max_turns=10, include_summary=False)

            messages = [
                {"role": "system", "content": "请用简洁的语言总结以下对话的核心内容。"},
                {"role": "user", "content": f"请总结：\n{history_text}"}
            ]

            summary = llm_service.generate(messages)
            if summary:
                self._summary = summary
                return summary
        except Exception as e:
            logger.error(f"LLM总结失败: {e}")

        return self._simple_summarize()

    def _simple_summarize(self) -> str:
        """简单总结"""
        if not self._turns:
            return ""

        # 提取关键信息
        all_questions = [t.user_question for t in self._turns[-5:]]
        all_answers = [t.assistant_answer[:100] for t in self._turns[-5:]]

        summary = f"用户询问了以下问题: {'、'.join(all_questions)}。"

        return summary


__all__ = ['ConversationHistory', 'ConversationTurn']