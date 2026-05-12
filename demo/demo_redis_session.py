# clean_all_invalid_sessions.py
"""
清理所有无效会话 - 删除只有元数据没有消息的会话
"""

import redis
import os
from dotenv import load_dotenv

load_dotenv()


def clean_all_invalid_sessions():
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD") or None,
        db=int(os.getenv("REDIS_DB", 0)),
        decode_responses=True
    )

    print("=" * 60)
    print("清理所有无效会话")
    print("=" * 60)

    # 获取所有元数据 key
    meta_keys = r.keys("rag:session:meta:*")
    print(f"找到 {len(meta_keys)} 个会话元数据\n")

    cleaned = 0

    for meta_key in meta_keys:
        session_id = meta_key.replace("rag:session:meta:", "")
        msg_key = f"rag:session:{session_id}"

        # 检查消息 key 是否存在
        msg_exists = r.exists(msg_key)
        metadata = r.hgetall(meta_key)

        if not msg_exists:
            print(f"🗑️ 删除无效会话: {session_id}")
            print(f"   元数据: user_id={metadata.get('user_id')}, "
                  f"message_count={metadata.get('message_count')}")

            # 删除元数据
            r.delete(meta_key)

            # 从用户列表中移除
            user_id = metadata.get('user_id', 'default')
            list_key = f"rag:session:list:{user_id}"
            r.lrem(list_key, 1, session_id)

            cleaned += 1
            print(f"   ✅ 已删除\n")

    print(f"=" * 60)
    print(f"清理完成，共删除 {cleaned} 个无效会话")
    print(f"剩余有效会话: {len(r.keys('rag:session:meta:*'))}")


if __name__ == "__main__":
    clean_all_invalid_sessions()