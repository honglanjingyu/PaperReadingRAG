# app/db/models.py
"""数据库模型 - 用户认证表"""

from datetime import datetime
from sqlalchemy import Column, String, DateTime, Boolean, Integer, create_engine
from sqlalchemy.ext.declarative import declarative_base
import hashlib
import secrets

Base = declarative_base()


def hash_password(password: str, salt: str = None) -> tuple:
    """使用 SHA256 哈希密码"""
    if salt is None:
        salt = secrets.token_hex(16)
    hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return salt, hash_obj.hex()


def verify_password(password: str, salt: str, password_hash: str) -> bool:
    """验证密码"""
    _, new_hash = hash_password(password, salt)
    return new_hash == password_hash


class User(Base):
    """用户表"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_salt = Column(String(64), nullable=False)
    password_hash = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    is_active = Column(Boolean, default=True)

    def set_password(self, password: str):
        """设置密码"""
        self.password_salt, self.password_hash = hash_password(password)

    def verify_password(self, password: str) -> bool:
        """验证密码"""
        return verify_password(password, self.password_salt, self.password_hash)


class UserSession(Base):
    """用户-会话关联表"""
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    session_id = Column(String(50), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.now)
    last_accessed = Column(DateTime, default=datetime.now, onupdate=datetime.now)


def get_engine(database_url: str):
    """获取数据库引擎"""
    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False
    )
    return engine