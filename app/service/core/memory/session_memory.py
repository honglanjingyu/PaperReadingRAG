# app/service/core/memory/session_memory.py
"""
会话记忆管理器 - 管理不同会话的记忆
"""

import uuid
import time
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """单条记忆条目"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)
    turn_id: int = 0  # 对话轮次


@dataclass
class ConversationMemory:
    """会话记忆"""
    session_id: str
    messages: List[MemoryEntry] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    turn_count: int = 0  # 对话轮次数


class SessionMemory:
    """
    会话记忆存储
    支持多个会话，自动过期清理
    """

    # 默认配置
    DEFAULT_MAX_TURNS = 20  # 最大保留轮次
    DEFAULT_MAX_TOKENS = 4000  # 最大保留token数
    DEFAULT_SESSION_TTL = 3600  # 会话过期时间（秒），默认1小时

    def __init__(
            self,
            max_turns: int = None,
            max_tokens: int = None,
            session_ttl: int = None,
            enable_cleanup: bool = True
    ):
        """
        初始化会话记忆

        Args:
            max_turns: 单个会话最大保留轮次
            max_tokens: 单个会话最大保留token数
            session_ttl: 会话过期时间（秒）
            enable_cleanup: 是否启用自动清理
        """
        self.max_turns = max_turns or self.DEFAULT_MAX_TURNS
        self.max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        self.session_ttl = session_ttl or self.DEFAULT_SESSION_TTL

        # 会话存储: session_id -> ConversationMemory
        self._sessions: Dict[str, ConversationMemory] = {}

        # 启用自动清理
        self._cleanup_enabled = enable_cleanup
        self._last_cleanup = time.time()

        logger.info(f"SessionMemory初始化: max_turns={self.max_turns}, "
                    f"max_tokens={self.max_tokens}, session_ttl={self.session_ttl}s")

    def get_or_create_session(self, session_id: str = None) -> str:
        """
        获取或创建会话

        Args:
            session_id: 会话ID，如果为空则自动生成

        Returns:
            session_id
        """
        if session_id and session_id in self._sessions:
            # 更新最后访问时间
            self._sessions[session_id].last_accessed = time.time()
            return session_id

        # 创建新会话
        if not session_id:
            session_id = self._generate_session_id()

        self._sessions[session_id] = ConversationMemory(
            session_id=session_id,
            created_at=time.time(),
            last_accessed=time.time()
        )

        logger.debug(f"创建新会话: {session_id}")
        self._cleanup_if_needed()

        return session_id

    def _generate_session_id(self) -> str:
        """生成唯一会话ID"""
        return hashlib.md5(f"{uuid.uuid4()}_{time.time()}".encode()).hexdigest()[:16]

    def add_message(
            self,
            session_id: str,
            role: str,
            content: str,
            timestamp: float = None
    ) -> bool:
        """
        添加消息到会话记忆

        Args:
            session_id: 会话ID
            role: 角色 ('user' 或 'assistant')
            content: 消息内容
            timestamp: 时间戳

        Returns:
            是否添加成功
        """
        session = self._sessions.get(session_id)
        if not session:
            logger.warning(f"会话不存在: {session_id}")
            return False

        timestamp = timestamp or time.time()

        # 用户消息：增加对话轮次
        if role == 'user':
            session.turn_count += 1

        # 添加消息
        entry = MemoryEntry(
            role=role,
            content=content,
            timestamp=timestamp,
            turn_id=session.turn_count
        )
        session.messages.append(entry)

        # 维护大小限制
        self._trim_session(session)

        logger.debug(f"添加消息到会话 {session_id}: {role}, 轮次={session.turn_count}")
        return True

    def get_conversation_history(
            self,
            session_id: str,
            max_turns: int = None,
            max_tokens: int = None,
            include_system: bool = True
    ) -> List[Dict[str, str]]:
        """
        获取对话历史（用于LLM）

        Args:
            session_id: 会话ID
            max_turns: 最大轮次
            max_tokens: 最大token数
            include_system: 是否包含系统消息

        Returns:
            消息列表，格式: [{"role": "user", "content": "..."}, ...]
        """
        session = self._sessions.get(session_id)
        if not session or not session.messages:
            return []

        # 更新最后访问时间
        session.last_accessed = time.time()

        # 转换为字典格式
        messages = []
        for entry in session.messages:
            messages.append({
                "role": entry.role,
                "content": entry.content,
                "timestamp": entry.timestamp
            })

        # 按轮次限制裁剪
        max_turns = max_turns or self.max_turns
        if max_turns and len(messages) > max_turns * 2:
            # 保留最近的轮次
            messages = messages[-(max_turns * 2):]

        # 按token限制裁剪（预估）
        max_tokens = max_tokens or self.max_tokens
        if max_tokens:
            messages = self._trim_by_tokens(messages, max_tokens)

        # 转换为LLM可用的格式
        result = []
        for msg in messages:
            result.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return result

    def _trim_by_tokens(
            self,
            messages: List[Dict],
            max_tokens: int
    ) -> List[Dict]:
        """按token数裁剪消息"""

        # 简单的token估算（中英文混合）
        def estimate_tokens(text: str) -> int:
            chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            others = len(text) - chinese
            return int(chinese / 1.5 + others / 4)

        total_tokens = 0
        # 从最新的消息开始计算
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            content = msg.get("content", "")
            msg_tokens = estimate_tokens(content)

            if total_tokens + msg_tokens > max_tokens:
                # 超出限制，返回后续消息
                return messages[i + 1:]
            total_tokens += msg_tokens

        return messages

    def _trim_session(self, session: ConversationMemory):
        """裁剪会话记忆"""
        # 按轮次裁剪
        if len(session.messages) > self.max_turns * 2:
            # 保留最近的轮次
            session.messages = session.messages[-(self.max_turns * 2):]

        # 按token裁剪
        max_tokens = self.max_tokens
        if max_tokens:
            # 转换为字典格式以便裁剪
            messages = [
                {"role": e.role, "content": e.content, "timestamp": e.timestamp}
                for e in session.messages
            ]
            trimmed = self._trim_by_tokens(messages, max_tokens)
            if len(trimmed) != len(messages):
                # 重新构建消息列表
                trimmed_set = {(t.get("timestamp"), t.get("role")) for t in trimmed}
                session.messages = [
                    e for e in session.messages
                    if (e.timestamp, e.role) in trimmed_set
                ]

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """获取会话信息"""
        session = self._sessions.get(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "turn_count": session.turn_count,
            "message_count": len(session.messages),
            "created_at": session.created_at,
            "last_accessed": session.last_accessed,
            "is_active": self._is_session_active(session)
        }

    def clear_session(self, session_id: str) -> bool:
        """清除会话记忆"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"清除会话: {session_id}")
            return True
        return False

    def _is_session_active(self, session: ConversationMemory) -> bool:
        """检查会话是否活跃"""
        return (time.time() - session.last_accessed) < self.session_ttl

    def _cleanup_expired_sessions(self):
        """清理过期会话"""
        now = time.time()
        expired = []

        for session_id, session in self._sessions.items():
            if now - session.last_accessed > self.session_ttl:
                expired.append(session_id)

        for session_id in expired:
            del self._sessions[session_id]

        if expired:
            logger.info(f"清理过期会话: {len(expired)} 个")

    def _cleanup_if_needed(self):
        """按需清理"""
        if not self._cleanup_enabled:
            return

        now = time.time()
        if now - self._last_cleanup > 300:  # 每5分钟清理一次
            self._cleanup_expired_sessions()
            self._last_cleanup = now

    def get_active_sessions_count(self) -> int:
        """获取活跃会话数"""
        return sum(1 for s in self._sessions.values() if self._is_session_active(s))

    def get_total_sessions_count(self) -> int:
        """获取总会话数"""
        return len(self._sessions)


# 全局单例
_memory_manager = None


def get_memory_manager() -> SessionMemory:
    """获取会话记忆管理器单例"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = SessionMemory()
    return _memory_manager


__all__ = ['SessionMemory', 'get_memory_manager', 'ConversationMemory', 'MemoryEntry']