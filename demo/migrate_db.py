# scripts/migrate_db.py
"""数据库迁移脚本 - 添加 role 列"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    db_url = f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:{os.getenv('POSTGRES_PASSWORD', 'postgres')}@" \
             f"{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'rag_db')}"

    engine = create_engine(db_url)

    try:
        with engine.connect() as conn:
            # 检查 role 列是否存在
            result = conn.execute(text("""
                                       SELECT column_name
                                       FROM information_schema.columns
                                       WHERE table_name = 'users'
                                         AND column_name = 'role'
                                       """))

            if result.fetchone():
                logger.info("role 列已存在，跳过迁移")
            else:
                logger.info("正在添加 role 列...")
                # 在 SQLAlchemy 2.0 中，需要显式开启事务
                with conn.begin():  # 使用 with conn.begin() 自动管理事务
                    conn.execute(text("""
                                      ALTER TABLE users
                                          ADD COLUMN role VARCHAR(20) DEFAULT 'normal' NOT NULL
                                      """))
                logger.info("✓ role 列添加成功")

            # 验证
            result = conn.execute(text("SELECT COUNT(*) FROM users"))
            user_count = result.fetchone()[0]
            logger.info(f"users 表中共有 {user_count} 个用户，role 列默认值为 'normal'")

            # 查看现有用户的 role 值
            result = conn.execute(text("SELECT username, role FROM users LIMIT 5"))
            logger.info("现有用户示例:")
            for row in result:
                logger.info(f"  - {row[0]}: {row[1]}")

    except Exception as e:
        logger.error(f"迁移失败: {e}")
        raise


if __name__ == "__main__":
    main()