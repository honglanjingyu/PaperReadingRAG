# app/service/core/memory/redis_session_memory.py
"""
Redis会话记忆管理器 - 支持持久化、分布式、自动过期
"""

import os
import json
import time
import hashlib
import uuid
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("警告: redis模块未安装，请运行: pip install redis")

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """单条记忆条目"""
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    turn_id: int = 0


class RedisSessionMemory:
    """
    Redis会话记忆存储
    支持多实例共享、自动过期、持久化
    """

    # 默认配置
    DEFAULT_MAX_TURNS = 20  # 最大保留轮次
    DEFAULT_MAX_TOKENS = 4000  # 最大保留token数
    DEFAULT_SESSION_TTL = 604800  # 会话过期时间（秒），默认7天

    def __init__(
            self,
            redis_client: redis.Redis = None,
            max_turns: int = None,
            max_tokens: int = None,
            session_ttl: int = None
    ):
        """
        初始化Redis会话记忆

        Args:
            redis_client: Redis客户端实例（可选，不传则自动创建）
            max_turns: 最大保留轮次
            max_tokens: 最大保留token数
            session_ttl: 会话过期时间（秒）
        """
        self.max_turns = max_turns or int(os.getenv("MEMORY_MAX_TURNS", self.DEFAULT_MAX_TURNS))
        self.max_tokens = max_tokens or int(os.getenv("MEMORY_MAX_TOKENS", self.DEFAULT_MAX_TOKENS))
        self.session_ttl = session_ttl or int(os.getenv("REDIS_SESSION_TTL", self.DEFAULT_SESSION_TTL))

        # 初始化Redis连接
        if redis_client:
            self.redis_client = redis_client
        elif REDIS_AVAILABLE:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                password=os.getenv("REDIS_PASSWORD") or None,
                db=int(os.getenv("REDIS_DB", 0)),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # 测试连接
            try:
                self.redis_client.ping()
                logger.info("Redis连接成功，会话记忆将使用Redis存储")
            except Exception as e:
                logger.error(f"Redis连接失败: {e}")
                self.redis_client = None
        else:
            self.redis_client = None
            logger.warning("redis模块未安装，会话记忆将使用内存存储")

        # 降级模式：内存存储
        self._fallback_storage: Dict[str, List] = {}
        self._fallback_meta: Dict[str, Dict] = {}

        # Redis key前缀
        self._key_prefix = "rag:session:"
        self._meta_prefix = "rag:session:meta:"
        self._list_prefix = "rag:session:list:"  # 用户会话列表

        logger.info(f"RedisSessionMemory初始化: max_turns={self.max_turns}, "
                    f"max_tokens={self.max_tokens}, session_ttl={self.session_ttl}s")

    def _is_redis_available(self) -> bool:
        """检查Redis是否可用"""
        return self.redis_client is not None

    def _get_session_key(self, session_id: str) -> str:
        """获取会话消息的Redis key"""
        return f"{self._key_prefix}{session_id}"

    def _get_meta_key(self, session_id: str) -> str:
        """获取会话元数据的Redis key"""
        return f"{self._meta_prefix}{session_id}"

    def _get_list_key(self, user_id: str) -> str:
        """获取用户会话列表的Redis key"""
        return f"{self._list_prefix}{user_id}"

    def get_or_create_session(self, session_id: str = None, user_id: str = "default") -> str:
        """
        获取或创建会话

        Args:
            session_id: 会话ID，如果为空则自动生成
            user_id: 用户ID

        Returns:
            session_id
        """
        if session_id and self._session_exists(session_id, user_id):
            # 更新最后访问时间
            self._update_last_accessed(session_id, user_id)
            return session_id

        # 创建新会话
        if not session_id:
            session_id = self._generate_session_id()
        else:
            # 检查是否已存在（但可能属于其他用户）
            if self._session_exists(session_id, None):
                # 会话存在但属于其他用户，创建新ID
                existing_user = self.redis_client.hget(self._get_meta_key(session_id),
                                                       "user_id") if self._is_redis_available() else None
                if existing_user and existing_user != user_id:
                    logger.warning(f"会话 {session_id} 属于用户 {existing_user}，为 {user_id} 创建新会话")
                    session_id = self._generate_session_id()

        # 初始化会话元数据
        meta = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": time.time(),
            "last_accessed": time.time(),
            "turn_count": 0,
            "message_count": 0
        }

        if self._is_redis_available():
            self.redis_client.hset(self._get_meta_key(session_id), mapping=meta)
            self.redis_client.expire(self._get_meta_key(session_id), self.session_ttl)

            # 添加到用户会话列表
            self.redis_client.lpush(self._get_list_key(user_id), session_id)
            self.redis_client.ltrim(self._get_list_key(user_id), 0, 99)  # 最多保留100个会话
        else:
            # 降级模式
            if session_id not in self._fallback_storage:
                self._fallback_storage[session_id] = []
            self._fallback_meta[session_id] = meta

        logger.debug(f"创建新会话: {session_id}")
        self._cleanup_if_needed()

        return session_id

    def _session_exists(self, session_id: str, user_id: str = None) -> bool:
        """检查会话是否存在"""
        if self._is_redis_available():
            if not self.redis_client.exists(self._get_meta_key(session_id)):
                return False
            if user_id:
                stored_user = self.redis_client.hget(self._get_meta_key(session_id), "user_id")
                return stored_user == user_id
            return True
        else:
            return session_id in self._fallback_meta

    def _update_last_accessed(self, session_id: str, user_id: str = "default"):
        """更新最后访问时间"""
        if self._is_redis_available():
            self.redis_client.hset(self._get_meta_key(session_id), "last_accessed", time.time())
            # 刷新TTL
            self.redis_client.expire(self._get_session_key(session_id), self.session_ttl)
            self.redis_client.expire(self._get_meta_key(session_id), self.session_ttl)

    def _generate_session_id(self) -> str:
        """生成唯一会话ID"""
        return hashlib.md5(f"{uuid.uuid4()}_{time.time()}".encode()).hexdigest()[:16]

    def add_message(self, session_id: str, role: str, content: str,
                    user_id: str = "default", timestamp: float = None) -> bool:
        """
        添加消息到会话记忆

        Args:
            session_id: 会话ID
            role: 角色 ('user' 或 'assistant')
            content: 消息内容
            user_id: 用户ID
            timestamp: 时间戳

        Returns:
            是否添加成功
        """
        if not self._session_exists(session_id, user_id):
            logger.warning(f"会话不存在: {session_id}")
            return False

        timestamp = timestamp or time.time()

        # 构建消息
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }

        if self._is_redis_available():
            # Redis存储
            key = self._get_session_key(session_id)

            # 添加消息到列表尾部
            self.redis_client.rpush(key, json.dumps(message, ensure_ascii=False))

            # 更新元数据
            if role == 'user':
                self.redis_client.hincrby(self._get_meta_key(session_id), "turn_count", 1)
            self.redis_client.hincrby(self._get_meta_key(session_id), "message_count", 1)
            self.redis_client.hset(self._get_meta_key(session_id), "last_accessed", time.time())

            # 设置过期时间
            self.redis_client.expire(key, self.session_ttl)
            self.redis_client.expire(self._get_meta_key(session_id), self.session_ttl)

            # 维护大小限制
            self._trim_session_redis(session_id)
        else:
            # 降级内存存储
            if session_id not in self._fallback_storage:
                self._fallback_storage[session_id] = []
            self._fallback_storage[session_id].append(message)

            # 更新元数据
            if session_id in self._fallback_meta:
                if role == 'user':
                    self._fallback_meta[session_id]["turn_count"] += 1
                self._fallback_meta[session_id]["message_count"] += 1
                self._fallback_meta[session_id]["last_accessed"] = time.time()

            # 维护大小限制
            self._trim_session_fallback(session_id)

        logger.debug(f"添加消息到会话 {session_id}: {role}")
        return True

    def get_conversation_history(self, session_id: str, user_id: str = "default",
                                 max_turns: int = None, max_tokens: int = None) -> List[Dict[str, str]]:
        """
        获取对话历史（用于LLM）

        Args:
            session_id: 会话ID
            user_id: 用户ID
            max_turns: 最大轮次
            max_tokens: 最大token数

        Returns:
            消息列表，格式: [{"role": "user", "content": "..."}, ...]
        """
        if not self._session_exists(session_id, user_id):
            return []

        # 更新最后访问时间
        self._update_last_accessed(session_id, user_id)

        # 获取消息
        if self._is_redis_available():
            messages = self._get_messages_redis(session_id)
        else:
            messages = self._fallback_storage.get(session_id, [])

        if not messages:
            return []

        # 按轮次限制裁剪
        max_turns = max_turns or self.max_turns
        if max_turns and len(messages) > max_turns * 2:
            messages = messages[-(max_turns * 2):]

        # 按token限制裁剪
        max_tokens = max_tokens or self.max_tokens
        if max_tokens:
            messages = self._trim_by_tokens(messages, max_tokens)

        # 转换为LLM格式
        result = []
        for msg in messages:
            result.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return result

    def _get_messages_redis(self, session_id: str) -> List[Dict]:
        """从Redis获取消息列表"""
        key = self._get_session_key(session_id)
        messages_json = self.redis_client.lrange(key, 0, -1)
        return [json.loads(msg) for msg in messages_json]

    def get_history_text(self, session_id: str, user_id: str = "default",
                         max_turns: int = 10, max_tokens: int = 2000) -> str:
        """
        获取格式化的历史文本

        Args:
            session_id: 会话ID
            user_id: 用户ID
            max_turns: 最大轮次
            max_tokens: 最大token数

        Returns:
            格式化的历史文本
        """
        history = self.get_conversation_history(session_id, user_id, max_turns, max_tokens)

        if not history:
            return ""

        formatted_lines = []
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            content = msg["content"]
            formatted_lines.append(f"{role}: {content}")

        return "\n".join(formatted_lines)

    def _trim_by_tokens(self, messages: List[Dict], max_tokens: int) -> List[Dict]:
        """按token数裁剪消息"""

        def estimate_tokens(text: str) -> int:
            if not text:
                return 0
            chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            others = len(text) - chinese
            return int(chinese / 1.5 + others / 4)

        total_tokens = 0
        for i in range(len(messages) - 1, -1, -1):
            msg_tokens = estimate_tokens(messages[i]["content"])
            if total_tokens + msg_tokens > max_tokens:
                return messages[i + 1:]
            total_tokens += msg_tokens

        return messages

    def _trim_session_redis(self, session_id: str):
        """裁剪Redis中的会话记忆"""
        key = self._get_session_key(session_id)

        # 获取消息数量
        msg_count = self.redis_client.llen(key)

        # 按轮次裁剪
        if msg_count > self.max_turns * 2:
            # 删除最早的消息
            excess = msg_count - (self.max_turns * 2)
            self.redis_client.ltrim(key, excess, -1)

        # 按token裁剪（获取当前消息）
        messages = self._get_messages_redis(session_id)
        trimmed = self._trim_by_tokens(messages, self.max_tokens)

        if len(trimmed) != len(messages):
            # 重新设置消息列表
            self.redis_client.delete(key)
            for msg in trimmed:
                self.redis_client.rpush(key, json.dumps(msg, ensure_ascii=False))

    def _trim_session_fallback(self, session_id: str):
        """裁剪内存中的会话记忆"""
        messages = self._fallback_storage.get(session_id, [])

        # 按轮次裁剪
        if len(messages) > self.max_turns * 2:
            self._fallback_storage[session_id] = messages[-(self.max_turns * 2):]

        # 按token裁剪
        trimmed = self._trim_by_tokens(self._fallback_storage[session_id], self.max_tokens)
        self._fallback_storage[session_id] = trimmed

    def get_session_history(self, session_id: str, user_id: str = "default",
                            limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        获取会话的完整历史（用于前端恢复）

        Args:
            session_id: 会话ID
            user_id: 用户ID
            limit: 返回消息数量限制
            offset: 偏移量

        Returns:
            消息列表，包含id, role, content, created_at等字段
        """
        if not self._session_exists(session_id, user_id):
            return []

        # 获取消息
        if self._is_redis_available():
            messages = self._get_messages_redis(session_id)
        else:
            messages = self._fallback_storage.get(session_id, [])

        # 应用分页
        start = offset
        end = offset + limit if limit > 0 else len(messages)
        messages = messages[start:end]

        # 转换为前端需要的格式
        result = []
        for i, msg in enumerate(messages):
            result.append({
                "id": i,
                "role": msg["role"],
                "content": msg["content"],
                "created_at": datetime.fromtimestamp(msg["timestamp"]).isoformat() if "timestamp" in msg else None
            })

        return result

    def get_session_info(self, session_id: str, user_id: str = "default") -> Optional[Dict]:
        """
        获取会话信息

        Args:
            session_id: 会话ID
            user_id: 用户ID

        Returns:
            会话信息字典
        """
        if not self._session_exists(session_id, user_id):
            return None

        if self._is_redis_available():
            meta = self.redis_client.hgetall(self._get_meta_key(session_id))
            if not meta:
                return None
            msg_count = self.redis_client.llen(self._get_session_key(session_id))
            return {
                "session_id": session_id,
                "user_id": meta.get("user_id"),
                "turn_count": int(meta.get("turn_count", 0)),
                "message_count": msg_count,
                "created_at": float(meta.get("created_at", 0)),
                "last_accessed": float(meta.get("last_accessed", 0)),
                "is_active": (time.time() - float(meta.get("last_accessed", 0))) < self.session_ttl
            }
        else:
            meta = self._fallback_meta.get(session_id, {})
            msg_count = len(self._fallback_storage.get(session_id, []))
            return {
                "session_id": session_id,
                "user_id": meta.get("user_id", user_id),
                "turn_count": meta.get("turn_count", 0),
                "message_count": msg_count,
                "created_at": meta.get("created_at", 0),
                "last_accessed": meta.get("last_accessed", 0),
                "is_active": (time.time() - meta.get("last_accessed", 0)) < self.session_ttl
            }

    def clear_session(self, session_id: str, user_id: str = "default") -> bool:
        """
        清除会话记忆

        Args:
            session_id: 会话ID
            user_id: 用户ID

        Returns:
            是否清除成功
        """
        if self._is_redis_available():
            self.redis_client.delete(self._get_session_key(session_id))
            self.redis_client.delete(self._get_meta_key(session_id))
            self.redis_client.lrem(self._get_list_key(user_id), 1, session_id)
        else:
            self._fallback_storage.pop(session_id, None)
            self._fallback_meta.pop(session_id, None)

        logger.info(f"清除会话: {session_id}")
        return True

    def get_active_sessions_count(self) -> int:
        """获取活跃会话数"""
        if self._is_redis_available():
            # 获取所有元数据key
            keys = self.redis_client.keys(f"{self._meta_prefix}*")
            active = 0
            now = time.time()
            for key in keys:
                last_accessed = self.redis_client.hget(key, "last_accessed")
                if last_accessed and (now - float(last_accessed)) < self.session_ttl:
                    active += 1
            return active
        else:
            return len(self._fallback_meta)

    def get_total_sessions_count(self) -> int:
        """获取总会话数"""
        if self._is_redis_available():
            return len(self.redis_client.keys(f"{self._meta_prefix}*"))
        else:
            return len(self._fallback_meta)

    def list_user_sessions(self, user_id: str = "default", limit: int = 20) -> List[Dict]:
        """列出用户的所有会话"""
        sessions = []
        if self._is_redis_available():
            session_ids = self.redis_client.lrange(self._get_list_key(user_id), 0, limit - 1)
            for sid in session_ids:
                info = self.get_session_info(sid, user_id)
                if info:
                    sessions.append(info)
        else:
            for sid, meta in self._fallback_meta.items():
                if meta.get("user_id") == user_id:
                    sessions.append(self.get_session_info(sid, user_id))
        return sessions

    def _cleanup_if_needed(self):
        """按需清理（Redis有自动过期，无需手动）"""
        pass


# 全局单例
_redis_memory_manager = None


def get_memory_manager() -> RedisSessionMemory:
    """获取会话记忆管理器单例"""
    global _redis_memory_manager
    if _redis_memory_manager is None:
        _redis_memory_manager = RedisSessionMemory()
    return _redis_memory_manager


__all__ = ['RedisSessionMemory', 'get_memory_manager', 'MemoryEntry']