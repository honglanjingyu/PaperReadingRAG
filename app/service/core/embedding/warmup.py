"""embedding 冷启动预热（报告 2.4）

VISTA 子进程首次加载模型要 45~78 s，热态后 embedding 只占端到端 5.5%，
但第一个真实用户会吃满这段延迟。服务启动时在后台跑一次 dummy embedding，
把子进程和模型提前拉起来。

调用方式（app/api/main.py 的 lifespan 里）：
    task = asyncio.create_task(warmup_embedding_async())
不 await，避免阻塞启动；失败只记日志，不影响服务可用性。
"""

import asyncio
import logging
import os
import time

logger = logging.getLogger(__name__)

WARMUP_TEXT = "warmup"


def warmup_embedding(timeout: float = None) -> bool:
    """同步预热：跑一次 dummy embedding。成功返回 True。"""
    timeout = timeout or float(os.getenv("EMBEDDING_WARMUP_TIMEOUT", "180"))
    start = time.time()
    try:
        from app.service.core.embedding import get_embedding_service

        service = get_embedding_service()
        vector = service.generate_embedding(WARMUP_TEXT)
        cost = time.time() - start
        if vector:
            logger.info("embedding 预热完成: dim=%d, 耗时 %.1fs", len(vector), cost)
            return True
        logger.warning("embedding 预热返回空向量（耗时 %.1fs），后续真实请求可能仍需冷启动", cost)
        return False
    except Exception as e:
        logger.warning("embedding 预热失败（耗时 %.1fs）: %s", time.time() - start, e)
        return False


async def warmup_embedding_async(timeout: float = None) -> bool:
    """异步预热：放到线程池里跑，不阻塞事件循环。

    VISTA 的 encode 内部有锁、且要等子进程 stdout，属于阻塞 IO，必须丢线程。
    """
    timeout = timeout or float(os.getenv("EMBEDDING_WARMUP_TIMEOUT", "180"))
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(warmup_embedding, timeout), timeout=timeout + 10
        )
    except asyncio.TimeoutError:
        logger.warning("embedding 预热超时（>%.0fs），放弃等待，服务继续启动", timeout)
        return False
    except Exception as e:
        logger.warning("embedding 预热异常: %s", e)
        return False
