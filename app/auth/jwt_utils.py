# app/auth/jwt_utils.py
import jwt
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from app.db.config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRE_HOURS

logger = logging.getLogger(__name__)


def create_token(user_id: int, username: str) -> str:
    """创建 JWT token"""
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS),
        "iat": datetime.utcnow(),
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """解码并验证 JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token 已过期")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"无效的 Token: {e}")
        return None


def get_user_id_from_token(token: str) -> Optional[int]:
    """从 token 中获取用户ID"""
    payload = decode_token(token)
    if payload:
        return payload.get("user_id")
    return None


def get_username_from_token(token: str) -> Optional[str]:
    """从 token 中获取用户名"""
    payload = decode_token(token)
    if payload:
        return payload.get("username")
    return None