# app/db/models.py - 更新版本
"""数据库模型 - 用户认证表（支持用户等级）"""

from datetime import datetime
from sqlalchemy import Column, String, DateTime, Boolean, Integer, create_engine, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
import hashlib
import secrets
import enum

Base = declarative_base()


class UserRole(str, enum.Enum):
    """用户等级枚举"""
    NORMAL = "normal"  # 普通用户
    ADMIN = "admin"  # 管理员
    OWNER = "owner"  # 所有者


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


def determine_user_role(username: str) -> UserRole:
    """根据用户名确定用户等级"""
    username_lower = username.lower()

    # 所有者用户: root* 或 system*
    if username_lower.startswith('root') or username_lower.startswith('system'):
        return UserRole.OWNER

    # 管理员用户: admin*
    if username_lower.startswith('admin'):
        return UserRole.ADMIN

    # 普通用户
    return UserRole.NORMAL


class User(Base):
    """用户表"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_salt = Column(String(64), nullable=False)
    password_hash = Column(String(128), nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.NORMAL, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    is_active = Column(Boolean, default=True)

    def set_password(self, password: str):
        """设置密码"""
        self.password_salt, self.password_hash = hash_password(password)

    def verify_password(self, password: str) -> bool:
        """验证密码"""
        return verify_password(password, self.password_salt, self.password_hash)

    def set_role_from_username(self):
        """根据用户名设置用户等级"""
        self.role = determine_user_role(self.username)

    def can_access_level(self, doc_level: str) -> bool:
        """检查用户是否有权访问指定等级的文档"""
        level_priority = {
            UserRole.NORMAL: 1,
            UserRole.ADMIN: 2,
            UserRole.OWNER: 3,
        }

        user_priority = level_priority.get(self.role, 1)
        doc_priority = level_priority.get(
            UserRole(doc_level) if isinstance(doc_level, str) else doc_level,
            1
        )

        return user_priority >= doc_priority


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